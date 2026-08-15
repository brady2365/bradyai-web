import base64
import html
import re
import time
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/122 Safari/537.36"
)

SEARCH_URL = "https://www.bing.com/search"
TIMEOUT = 10
MAX_RESULTS_PER_SEARCH = 8
MAX_SOURCES = 4
MAX_CHARS_PER_SOURCE = 450

# Sources that are generally useful for factual explanations.
QUALITY_DOMAINS = {
    "ibm.com": 18,
    "microsoft.com": 15,
    "google.com": 12,
    "nist.gov": 22,
    "noaa.gov": 20,
    "nasa.gov": 20,
    "nih.gov": 20,
    "cdc.gov": 20,
    "wikipedia.org": 7,
    "britannica.com": 10,
    "nature.com": 16,
    "scientificamerican.com": 12,
    "arxiv.org": 10,
    "mit.edu": 17,
    "stanford.edu": 17,
    "harvard.edu": 17,
    "edu": 14,
    "gov": 16,
}

# These are usually poor results for explanatory questions.
LOW_QUALITY_DOMAINS = {
    "merriam-webster.com",
    "dictionary.com",
    "vocabulary.com",
    "thesaurus.com",
    "wiktionary.org",
    "urbandictionary.com",
    "quizlet.com",
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
}


session = requests.Session()
session.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)


def normalize_text(text):
    """Collapse whitespace and decode HTML entities."""
    text = html.unescape(text or "")
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    """Useful lowercase words for simple relevance scoring."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can",
        "do", "does", "for", "from", "how", "i", "in", "is", "it",
        "of", "on", "or", "the", "to", "what", "when", "where",
        "which", "who", "why", "with", "work", "works",
    }

    return {
        word
        for word in re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower())
        if word not in stop_words
    }


def domain_score(url):
    hostname = urlparse(url).netloc.lower().replace("www.", "")

    if hostname in LOW_QUALITY_DOMAINS:
        return -30

    score = 0

    for domain, value in QUALITY_DOMAINS.items():
        if hostname == domain or hostname.endswith("." + domain):
            score = max(score, value)

    return score


def is_dictionary_or_irrelevant(url, title, snippet, query):
    hostname = urlparse(url).netloc.lower().replace("www.", "")
    text = f"{title} {snippet}".lower()
    query_lower = query.lower()

    if hostname in LOW_QUALITY_DOMAINS:
        # Dictionary sources are okay only when the user is explicitly
        # asking for a definition or meaning.
        definition_words = {"define", "definition", "meaning", "means"}
        if not any(word in query_lower for word in definition_words):
            return True

    bad_patterns = [
        "word of the day",
        "scrabble",
        "synonym",
        "antonym",
        "pronunciation",
        "crossword",
    ]

    return any(pattern in text for pattern in bad_patterns)


def decode_bing_url(url):
    """
    Try to extract a real target URL from Bing's tracking redirect.
    Bing often uses /ck/a?...&u=a1<base64-url>.
    """
    if not url:
        return ""

    url = html.unescape(url)
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    # Sometimes Bing exposes a direct destination parameter.
    for key in ("url", "r", "target"):
        if params.get(key):
            candidate = unquote(params[key][0])
            if candidate.startswith(("http://", "https://")):
                return candidate

    encoded_values = params.get("u", [])
    if encoded_values:
        encoded = encoded_values[0]

        # Bing's base64 value often begins with "a1".
        if encoded.startswith("a1"):
            encoded = encoded[2:]

        try:
            encoded += "=" * (-len(encoded) % 4)
            decoded = base64.urlsafe_b64decode(encoded).decode(
                "utf-8", errors="ignore"
            )
            if decoded.startswith(("http://", "https://")):
                return decoded
        except Exception:
            pass

    return url


def resolve_url(url):
    """Follow redirects when possible, without failing the whole search."""
    url = decode_bing_url(url)

    if not url.startswith(("http://", "https://")):
        return ""

    try:
        response = session.get(
            url,
            timeout=TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
        final_url = response.url
        response.close()

        if final_url.startswith(("http://", "https://")):
            return final_url
    except requests.RequestException:
        pass

    return url


def search_bing(query):
    """Return raw Bing result records."""
    try:
        response = session.get(
            SEARCH_URL,
            params={"q": query, "setlang": "en-US"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for item in soup.select("li.b_algo"):
        link = item.select_one("h2 a")
        if not link:
            continue

        title = normalize_text(link.get_text(" ", strip=True))
        href = link.get("href", "")
        snippet_node = item.select_one(".b_caption p") or item.select_one("p")
        snippet = normalize_text(
            snippet_node.get_text(" ", strip=True) if snippet_node else ""
        )

        if not title or not href:
            continue

        results.append(
            {
                "title": title,
                "url": href,
                "snippet": snippet,
            }
        )

        if len(results) >= MAX_RESULTS_PER_SEARCH:
            break

    return results


def make_search_queries(query):
    """
    Search the exact question first, then add one focused fallback query.
    This helps Bing avoid matching a short word such as 'do'.
    """
    query = normalize_text(query)
    cleaned = re.sub(
        r"^(research|search|look up|find out about)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )

    query_words = list(words(cleaned))
    topic = " ".join(query_words[:6])

    queries = [cleaned]

    if topic and topic.lower() != cleaned.lower():
        queries.append(f"{topic} explained")
    elif len(query_words) >= 2:
        queries.append(f"{topic} overview")

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(q for q in queries if q))


def relevance_score(query, title, snippet, url):
    query_terms = words(query)
    title_terms = words(title)
    snippet_terms = words(snippet)

    if not query_terms:
        return domain_score(url)

    title_matches = len(query_terms & title_terms)
    snippet_matches = len(query_terms & snippet_terms)

    score = title_matches * 12
    score += snippet_matches * 5
    score += domain_score(url)

    combined = f"{title} {snippet}".lower()

    # Search results with none of the topic words are almost always noise.
    if title_matches == 0 and snippet_matches == 0:
        score -= 35

    # Prefer actual explanatory pages when the question asks how/what/why.
    if any(word in query.lower() for word in ("how", "what", "why", "explain")):
        if any(word in combined for word in ("explained", "overview", "guide", "introduction")):
            score += 4

    return score


def extract_page_text(url, query):
    """Fetch a page and choose paragraphs relevant to the query."""
    try:
        response = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
    except requests.RequestException:
        return ""

    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
    ):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    query_terms = words(query)

    boilerplate_phrases = (
        "official websites use .gov",
        "secure .gov websites use https",
        "cookie policy",
        "privacy policy",
        "subscribe to",
        "sign up for",
        "all rights reserved",
    )

    candidates = []

    for element in main.find_all(["p", "li"]):
        text = normalize_text(element.get_text(" ", strip=True))
        text_lower = text.lower()

        if len(text) < 70:
            continue

        if any(phrase in text_lower for phrase in boilerplate_phrases):
            continue

        paragraph_terms = words(text)
        matches = len(query_terms & paragraph_terms)

        # Prefer explanatory paragraphs, not page navigation.
        score = matches * 20

        if any(
            word in text_lower
            for word in (
                "because", "means", "uses", "works", "qubit",
                "superposition", "entanglement", "computing",
                "computer", "process"
            )
        ):
            score += 5

        if matches > 0:
            candidates.append((score, text))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0], reverse=True)

    chosen = []
    total_length = 0

    for _, text in candidates:
        if total_length + len(text) > MAX_CHARS_PER_SOURCE:
            continue

        chosen.append(text)
        total_length += len(text)

        if len(chosen) >= 2:
            break

    text = normalize_text(" ".join(chosen))

    if len(text) > MAX_CHARS_PER_SOURCE:
        text = text[:MAX_CHARS_PER_SOURCE].rstrip() + "..."

    return text

def research(query):
    """
    Public interface used by the chat program.

    Returns:
        {
            "query": original query,
            "sources": [{"title": ..., "url": ..., "snippet": ...}, ...],
            "research": concise text ready to send to BradyAI
        }
    """
    original_query = normalize_text(query)
    candidates = []
    seen_urls = set()

    for search_query in make_search_queries(original_query):
        for result in search_bing(search_query):
            real_url = resolve_url(result["url"])

            if not real_url:
                continue

            parsed = urlparse(real_url)
            canonical_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".lower()

            if canonical_url in seen_urls:
                continue

            seen_urls.add(canonical_url)

            if is_dictionary_or_irrelevant(
                real_url,
                result["title"],
                result["snippet"],
                original_query,
            ):
                continue

            score = relevance_score(
                original_query,
                result["title"],
                result["snippet"],
                real_url,
            )

            # Reject clearly unrelated results before downloading pages.
            if score < -5:
                continue

            candidates.append(
                {
                    "title": result["title"],
                    "url": real_url,
                    "snippet": result["snippet"],
                    "score": score,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    sources = []
    for candidate in candidates[:MAX_SOURCES]:
        page_text = extract_page_text(
            candidate["url"],
            original_query
        )

        # Bing's own snippet is a useful fallback if a page blocks requests.
        useful_snippet = page_text or candidate["snippet"]
        useful_snippet = normalize_text(useful_snippet)

        if len(useful_snippet) < 40:
            continue

        sources.append(
            {
                "title": candidate["title"],
                "url": candidate["url"],
                "snippet": useful_snippet,
            }
        )

    if not sources:
        research_text = (
            f"No reliable sources were found for: {original_query}. "
            "Answer cautiously and do not invent facts."
        )
    else:
        sections = []
        for index, source in enumerate(sources, start=1):
            sections.append(
                f"Source {index}: {source['title']}\n"
                f"{source['snippet']}"
            )

        research_text = (
            f"Research topic: {original_query}\n\n"
            + "\n\n".join(sections)
            + "\n\nUse only the useful facts above. "
              "Give a short, clear answer. Do not invent facts."
        )

    return {
        "query": original_query,
        "sources": sources,
        "research": research_text,
    }


if __name__ == "__main__":
    import sys

    test_query = " ".join(sys.argv[1:]) or "how do quantum computers work"
    result = research(test_query)

    print("\nRESEARCH:\n")
    print(result["research"])

    print("\nSOURCES:\n")
    for source in result["sources"]:
        print(f"- {source['title']}")
        print(f"  {source['url']}")