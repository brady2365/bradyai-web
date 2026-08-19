import json
import os
import re

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_from_directory, redirect
from flask_cors import CORS
from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
import psycopg2

from model import BradyAI
from research import research
from tokenizer import BPETokenizer


MODEL_FILE = "bradyai_v3.pt"
MAX_NEW_TOKENS = 12
TEMPERATURE = 0.25
TOP_K = 5
PUBLIC_MODE = os.environ.get("PUBLIC_MODE", "false").lower() == "true"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "https://nexuronai.github.io"
        }
    },
    supports_credentials=True
)


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    return psycopg2.connect(DATABASE_URL)

def init_database():
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            # =========================================
            # USER MEMORIES
            # =========================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    user_id TEXT PRIMARY KEY,
                    memory JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMP WITH TIME ZONE
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # =========================================
            # CONVERSATIONS
            # =========================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    created_at TIMESTAMP WITH TIME ZONE
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                ON conversations(user_id)
            """)

            # =========================================
            # MESSAGES
            # =========================================

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGSERIAL PRIMARY KEY,
                    conversation_id BIGINT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_user_id
                ON messages(user_id)
            """)

        connection.commit()

    finally:
        connection.close()
clerk = Clerk(
    bearer_auth=os.environ.get("CLERK_SECRET_KEY")
)

init_database()



# =========================================
# ACCOUNT-SPECIFIC MEMORY
# =========================================

def get_user_memory(user_id):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute("""
                SELECT memory
                FROM user_memories
                WHERE user_id = %s
            """, (user_id,))

            row = cursor.fetchone()

            if row is None:
                memory = {}

                cursor.execute("""
                    INSERT INTO user_memories (user_id, memory)
                    VALUES (%s, %s)
                """, (
                    user_id,
                    json.dumps(memory)
                ))

                connection.commit()

                return memory

            return row[0] or {}

    finally:
        connection.close()


def save_user_memory(user_id, memory):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute("""
                INSERT INTO user_memories
                    (user_id, memory, updated_at)
                VALUES
                    (%s, %s, CURRENT_TIMESTAMP)

                ON CONFLICT (user_id)
                DO UPDATE SET
                    memory = EXCLUDED.memory,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                user_id,
                json.dumps(memory)
            ))

        connection.commit()

    finally:
        connection.close()


# =========================================
# CONVERSATION FUNCTIONS
# =========================================

def create_conversation(user_id, title="New Chat"):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (user_id, title)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, title))

            conversation_id = cursor.fetchone()[0]

        connection.commit()
        return conversation_id

    finally:
        connection.close()


def get_user_conversations(user_id):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = %s
                ORDER BY updated_at DESC
            """, (user_id,))

            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2].isoformat(),
                    "updated_at": row[3].isoformat(),
                }
                for row in rows
            ]

    finally:
        connection.close()


def get_conversation_messages(user_id, conversation_id):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, role, content, created_at
                FROM messages
                WHERE conversation_id = %s
                  AND user_id = %s
                ORDER BY created_at ASC
            """, (conversation_id, user_id))

            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "created_at": row[3].isoformat(),
                }
                for row in rows
            ]

    finally:
        connection.close()


def save_message(user_id, conversation_id, role, content):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            # Make sure this conversation belongs to this user
            cursor.execute("""
                SELECT id
                FROM conversations
                WHERE id = %s
                  AND user_id = %s
            """, (conversation_id, user_id))

            if cursor.fetchone() is None:
                return False

            cursor.execute("""
                INSERT INTO messages
                    (conversation_id, user_id, role, content)
                VALUES
                    (%s, %s, %s, %s)
            """, (
                conversation_id,
                user_id,
                role,
                content
            ))

            cursor.execute("""
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                  AND user_id = %s
            """, (conversation_id, user_id))

        connection.commit()
        return True

    finally:
        connection.close()


def delete_conversation(user_id, conversation_id):
    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                DELETE FROM conversations
                WHERE id = %s
                  AND user_id = %s
            """, (conversation_id, user_id))

            deleted = cursor.rowcount > 0

        connection.commit()
        return deleted

    finally:
        connection.close()

def load_model():
    print("Loading BradyAI web model on", device)
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    tokenizer = BPETokenizer(vocab_size=len(checkpoint["vocab"]))
    tokenizer.vocab = checkpoint["vocab"]
    tokenizer.token_to_id = checkpoint["token_to_id"]
    tokenizer.id_to_token = {
        int(key): value for key, value in checkpoint["id_to_token"].items()
    }
    tokenizer.merges = [tuple(merge) for merge in checkpoint["merges"]]

    config = checkpoint["config"]
    model = BradyAI(
        vocab_size=len(tokenizer),
        embed_size=config["embed_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer, config["block_size"]


model, tokenizer, block_size = load_model()
EOS_ID = tokenizer.token_to_id.get("<EOS>")


def needs_research(user_text):
    text = user_text.lower().strip()
    commands = (
        "research ", "look up ", "look it up ", "search for ",
        "search the web ", "search online ", "find information about ",
        "find info about ",
    )
    current_words = (
        "latest", "today", "tonight", "yesterday", "this week",
        "this month", "currently", "current", "recent", "newest",
        "recently",
    )
    return text.startswith(commands) or any(word in text for word in current_words)


def clean_research_query(user_text):
    text = user_text.strip()
    for prefix in (
        "research ", "look up ", "look it up ", "search for ",
        "search the web ", "search online ", "find information about ",
        "find info about ",
    ):
        if text.lower().startswith(prefix):
            return text[len(prefix):].strip()
    return text


def build_research_answer(result):
    sources = result.get("sources", [])
    if not sources:
        return "I could not find reliable sources for that. Try different wording."

    parts = ["Here is what I found:"]
    for source in sources:
        snippet = re.sub(r"https?://\S+", "", source.get("snippet", "")).strip()
        if not snippet:
            continue

        lower = snippet.lower()
        teaser = snippet.count("?") > 0 and not any(
            word in lower
            for word in ("uses", "qubit", "superposition", "entanglement", "process", "information")
        )
        if teaser:
            continue

        snippet = snippet[:500]
        last_stop = max(snippet.rfind("."), snippet.rfind("!"), snippet.rfind("?"))
        if last_stop > 120:
            snippet = snippet[:last_stop + 1]
        parts.append("- " + snippet)

        if len(parts) >= 4:
            break

    return "\n\n".join(parts) if len(parts) > 1 else "I found sources but no useful summary."


def memory_reply(user_text, user_id):
    
    session_memory = get_user_memory(user_id)
    
    text = user_text.strip()
    lower = text.lower()

    # A deployed service is shared by many visitors. Do not put any visitor's
    # notes in a shared server-side dictionary or file.
    memory_phrases = (
        "my name is", "call me", "favorite color", "i am learning",
        "remember ", "forget ", "show my notes", "list my notes",
        "what notes do you remember", "what do you remember",
        "what is my name", "remember my name", "what am i learning",
    )

    name_match = re.search(r"^(?:my name is|call me)\s+([A-Za-z][A-Za-z'-]*)[.!]?$", text, re.I)
    if name_match:
        name = name_match.group(1)
        session_memory["name"] = name
        save_user_memory(user_id, session_memory)
        return f"Nice to meet you, {name}. I will remember your name."

    color_match = re.search(r"^my favorite colo[u]?r is\s+([A-Za-z]+)[.!]?$", text, re.I)
    if color_match:
        color = color_match.group(1).lower()
        session_memory["favorite_color"] = color
        save_user_memory(user_id, session_memory)
        return f"I will remember that your favorite color is {color}."

    learning_match = re.search(r"^i am learning\s+(.+?)[.!]?$", text, re.I)
    if learning_match:
        subject = learning_match.group(1).strip()
        session_memory["learning"] = subject
        save_user_memory(user_id, session_memory)
        return f"I will remember that you are learning {subject}."

    remember_match = re.search(r"^remember\s+(.+)$", text, re.I)
    if remember_match:
        note = remember_match.group(1).strip()
        if note:
            notes = session_memory.setdefault("notes", [])
            if note not in notes:
                notes.append(note)
                save_user_memory(user_id, session_memory)
            return "I will remember: " + note

    forget_match = re.search(r"^forget\s+(.+)$", text, re.I)
    if forget_match:
        requested = forget_match.group(1).strip()
        notes = session_memory.get("notes", [])
        if requested.lower() in ("all notes", "my notes"):
            session_memory["notes"] = []
            save_user_memory(user_id, session_memory)
            return "I forgot all of your saved notes."
        for note in notes:
            if note.lower() == requested.lower():
                notes.remove(note)
                save_user_memory(user_id, session_memory)
                return "I forgot: " + note
        return "I could not find that note. Use 'show my notes' to see saved notes."

    if "what is my name" in lower or "remember my name" in lower:
        return f"Your name is {session_memory['name']}." if "name" in session_memory else "You have not told me your name yet."

    if "what is my favorite color" in lower or "remember my favorite color" in lower:
        return (
            f"Your favorite color is {session_memory['favorite_color']}."
            if "favorite_color" in session_memory
            else "You have not told me your favorite color yet."
        )

    if "what am i learning" in lower or "remember what i am learning" in lower:
        return (
            f"You are learning {session_memory['learning']}."
            if "learning" in session_memory
            else "You have not told me what you are learning yet."
        )

    if "show my notes" in lower or "list my notes" in lower or "what notes do you remember" in lower:
        notes = session_memory.get("notes", [])
        return "Here are your saved notes:\n- " + "\n- ".join(notes) if notes else "You have no saved notes yet."

    if "what do you remember" in lower:
        details = []
        if "name" in session_memory:
            details.append("your name is " + session_memory["name"])
        if "favorite_color" in session_memory:
            details.append("your favorite color is " + session_memory["favorite_color"])
        if "learning" in session_memory:
            details.append("you are learning " + session_memory["learning"])
        if session_memory.get("notes"):
            details.append("your notes are: " + "; ".join(session_memory["notes"]))
        return "I remember that " + "; ".join(details) + "." if details else "I do not have saved details yet."

    return None


@torch.no_grad()
def generate(user_text):
    import time

    total_start = time.perf_counter()

    # Tokenization
    token_start = time.perf_counter()

    token_ids = tokenizer.encode(
        "User: " + user_text + "\nAssistant:",
        add_special_tokens=False
    )

    token_ids.insert(
        0,
        tokenizer.token_to_id["<BOS>"]
    )

    token_ids = token_ids[-block_size:]

    token_time = time.perf_counter() - token_start

    # Create tensor
    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    generated = []

    # Generation
    generation_start = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):

        logits, _ = model(
            x[:, -block_size:]
        )

        logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        k = min(
            TOP_K,
            logits.size(-1)
        )

        values, indices = torch.topk(
            logits,
            k
        )

        filtered = torch.full_like(
            logits,
            float("-inf")
        )

        filtered.scatter_(
            1,
            indices,
            values
        )

        next_token = torch.multinomial(
            F.softmax(
                filtered,
                dim=-1
            ),
            num_samples=1
        )

        token_id = next_token.item()

        if token_id == EOS_ID:
            break

        generated.append(token_id)

        x = torch.cat(
            [x, next_token],
            dim=1
        )

        if "User:" in tokenizer.decode(generated):
            break

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    response = tokenizer.decode(
        generated
    )

    response = (
        response
        .split("User:")[0]
        .split("\nAssistant:")[0]
        .strip()
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        f"[TIMING] "
        f"tokenization={token_time:.3f}s | "
        f"generation={generation_time:.3f}s | "
        f"total={total_time:.3f}s | "
        f"tokens={len(generated)}",
        flush=True
    )

    return response

def get_authenticated_user():
    """
    Verify the Clerk session token sent by the browser.

    Returns:
        user_id if authenticated
        None if not authenticated
    """

    try:

        result = clerk.authenticate_request(
            request,
            AuthenticateRequestOptions(
                accepts_token=["session_token"]
            )
        )

        print(
            f"[CLERK] signed_in={result.is_signed_in} "
            f"reason={getattr(result, 'reason', None)}",
            flush=True
        )

        if not result.is_signed_in:
            return None

        return result.payload.get("sub")

    except Exception as error:

        print(
            f"[CLERK AUTH ERROR] {error}",
            flush=True
        )

        return None



@app.get("/api/auth")
def auth_status():

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "authenticated": False
        }), 401

    return jsonify({
        "authenticated": True,
        "user_id": user_id
    })

@app.post("/api/conversations")
def create_conversation_api():

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "error": "You must be signed in."
        }), 401

    data = request.get_json(silent=True) or {}

    title = str(
        data.get("title", "New Chat")
    ).strip()

    if not title:
        title = "New Chat"

    conversation_id = create_conversation(
        user_id,
        title
    )

    return jsonify({
        "id": conversation_id,
        "title": title
    })


@app.get("/api/conversations")
def list_conversations():

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "error": "You must be signed in."
        }), 401

    conversations = get_user_conversations(
        user_id
    )

    return jsonify({
        "conversations": conversations
    })


@app.get("/api/conversations/<int:conversation_id>")
def get_conversation(conversation_id):

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "error": "You must be signed in."
        }), 401

    messages = get_conversation_messages(
        user_id,
        conversation_id
    )

    return jsonify({
        "messages": messages
    })

@app.patch("/api/conversations/<int:conversation_id>")
def update_conversation(conversation_id):

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "error": "You must be signed in."
        }), 401

    data = request.get_json(silent=True) or {}

    title = str(
        data.get("title", "")
    ).strip()

    if not title:
        return jsonify({
            "error": "Conversation title cannot be empty."
        }), 400

    title = title[:100]

    connection = get_db_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute("""
                UPDATE conversations
                SET title = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                  AND user_id = %s
            """, (
                title,
                conversation_id,
                user_id
            ))

            if cursor.rowcount == 0:

                connection.rollback()

                return jsonify({
                    "error": "Conversation not found."
                }), 404

        connection.commit()

    finally:
        connection.close()

    return jsonify({
        "success": True,
        "title": title
    })

@app.delete("/api/conversations/<int:conversation_id>")
def delete_conversation_api(conversation_id):

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "error": "You must be signed in."
        }), 401

    deleted = delete_conversation(
        user_id,
        conversation_id
    )

    if not deleted:
        return jsonify({
            "error": "Conversation not found."
        }), 404

    return jsonify({
        "success": True
    })

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.get("/chat")
def chat_page():
    return render_template("chat.html")

@app.post("/api/chat")
def chat():

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "error": "You must be signed in to use BradyAI."
        }), 401

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "error": "Enter a message first."
        }), 400

    # =========================================
    # CLEAR MEMORY
    # =========================================

    if message.lower() == "clear memory":

        if PUBLIC_MODE:
            return jsonify({
                "reply": "Personal memory is disabled on this public demo.",
                "sources": []
            })

        save_user_memory(
            user_id,
            {}
        )

        return jsonify({
            "reply": "Memory cleared.",
            "sources": []
        })

    # =========================================
    # RESEARCH
    # =========================================

    if needs_research(message):

        result = research(
            clean_research_query(message)
        )

        reply = build_research_answer(result)

        return jsonify({
            "reply": reply,
            "sources": result.get("sources", [])
        })

    # =========================================
    # MEMORY / AI
    # =========================================

    saved_reply = memory_reply(
        message,
        user_id
    )

    if saved_reply is not None:
        reply = saved_reply

    else:
        reply = generate(message)

    reply = reply or "I do not know how to respond to that yet."

    # =========================================
    # RETURN RESPONSE
    # =========================================

    return jsonify({
        "reply": reply,
        "sources": []
    })
@app.post("/api/conversations/<int:conversation_id>/messages")
def add_conversation_message(conversation_id):

    user_id = get_authenticated_user()

    if not user_id:
        return jsonify({
            "error": "You must be signed in."
        }), 401

    data = request.get_json(silent=True) or {}

    role = str(
        data.get("role", "")
    ).strip()

    content = str(
        data.get("content", "")
    ).strip()

    if role not in ("user", "assistant"):
        return jsonify({
            "error": "Invalid message role."
        }), 400

    if not content:
        return jsonify({
            "error": "Message content cannot be empty."
        }), 400

    saved = save_message(
        user_id,
        conversation_id,
        role,
        content
    )

    if not saved:
        return jsonify({
            "error": "Conversation not found."
        }), 404

    return jsonify({
        "success": True
    })


@app.route("/sign-in")
def sign_in():
    return render_template("sign-in.html")


@app.route("/sign-up")
def sign_up():
    return render_template("sign-up.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
