from collections import Counter
import re


class BPETokenizer:

    def __init__(self, vocab_size=500):

        self.vocab_size = vocab_size

        self.vocab = []

        self.token_to_id = {}

        self.id_to_token = {}

        self.merges = []


    # ==========================================
    # SPLIT TEXT INTO SAFE PIECES
    # ==========================================

    def _split_text(self, text):

        # Keep words, whitespace, and punctuation separate.
        return re.findall(
            r"\s+|[A-Za-z0-9_]+|[^A-Za-z0-9_\s]",
            text
        )


    # ==========================================
    # TRAIN TOKENIZER
    # ==========================================

    def train(self, text):

        print("Training tokenizer...")

        pieces = self._split_text(text)

        vocabulary = {}

        for piece in pieces:

            sequence = tuple(piece)

            vocabulary[sequence] = (
                vocabulary.get(sequence, 0) + 1
            )


        # Initial character vocabulary
        tokens = set()

        for sequence in vocabulary:

            for token in sequence:

                tokens.add(token)


        # ======================================
        # BPE MERGING
        # ======================================

        while len(tokens) < self.vocab_size:

            pairs = Counter()


            for sequence, frequency in vocabulary.items():

                for i in range(
                    len(sequence) - 1
                ):

                    pair = (
                        sequence[i],
                        sequence[i + 1]
                    )

                    pairs[pair] += frequency


            if not pairs:
                break


            best_pair, count = pairs.most_common(1)[0]


            if count < 2:
                break


            new_token = (
                best_pair[0]
                + best_pair[1]
            )


            self.merges.append(
                best_pair
            )


            new_vocabulary = {}


            for sequence, frequency in vocabulary.items():

                new_sequence = []

                i = 0


                while i < len(sequence):

                    if (
                        i < len(sequence) - 1
                        and sequence[i] == best_pair[0]
                        and sequence[i + 1] == best_pair[1]
                    ):

                        new_sequence.append(
                            new_token
                        )

                        i += 2

                    else:

                        new_sequence.append(
                            sequence[i]
                        )

                        i += 1


                new_sequence = tuple(
                    new_sequence
                )


                new_vocabulary[
                    new_sequence
                ] = frequency


            vocabulary = new_vocabulary

            tokens.add(new_token)


        # ======================================
        # BUILD VOCABULARY
        # ======================================

        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]


        self.vocab = (
            special_tokens
            + sorted(tokens)
        )


        self.token_to_id = {
            token: i
            for i, token in enumerate(
                self.vocab
            )
        }


        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }


        print(
            "Tokenizer vocabulary:",
            len(self.vocab)
        )


    # ==========================================
    # APPLY BPE TO ONE PIECE
    # ==========================================

    def _apply_merges(self, tokens):

        for merge in self.merges:

            new_tokens = []

            i = 0


            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge[0]
                    and tokens[i + 1] == merge[1]
                ):

                    new_tokens.append(
                        merge[0] + merge[1]
                    )

                    i += 2

                else:

                    new_tokens.append(
                        tokens[i]
                    )

                    i += 1


            tokens = new_tokens


        return tokens


    # ==========================================
    # ENCODE
    # ==========================================

    def encode(
        self,
        text,
        add_special_tokens=False
    ):

        ids = []


        if add_special_tokens:

            ids.append(
                self.token_to_id["<BOS>"]
            )


        pieces = self._split_text(text)


        for piece in pieces:

            characters = list(piece)

            tokens = self._apply_merges(
                characters
            )


            for token in tokens:

                if token in self.token_to_id:

                    ids.append(
                        self.token_to_id[token]
                    )

                else:

                    ids.append(
                        self.token_to_id["<UNK>"]
                    )


        if add_special_tokens:

            ids.append(
                self.token_to_id["<EOS>"]
            )


        return ids


    # ==========================================
    # DECODE
    # ==========================================

    def decode(self, ids):

        output = ""


        for token_id in ids:

            token = self.id_to_token.get(
                int(token_id),
                ""
            )


            if token in [
                "<PAD>",
                "<BOS>",
                "<EOS>"
            ]:

                continue


            if token == "<UNK>":

                # Keep unknown tokens visible
                # while debugging.
                output += "<UNK>"

                continue


            output += token


        return output


    # ==========================================
    # LENGTH
    # ==========================================

    def __len__(self):

        return len(
            self.vocab
        )


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    text = """
User: What is Python?
Assistant: Python is a programming language.

User: What is a GPU?
Assistant: A GPU is a processor designed for parallel calculations.

User: Hello!
Assistant: Hello! How can I help you?
"""


    tokenizer = BPETokenizer(
        vocab_size=100
    )


    tokenizer.train(text)


    test_text = (
        "User: What is Python?\n"
        "Assistant: Python is a programming language."
    )


    encoded = tokenizer.encode(
        test_text
    )


    decoded = tokenizer.decode(
        encoded
    )


    print()
    print("Original:")
    print(repr(test_text))


    print()
    print("Decoded:")
    print(repr(decoded))


    print()
    print(
        "Exact match:",
        test_text == decoded
    )


    print()
    print(
        "Original characters:",
        len(test_text)
    )


    print(
        "Token count:",
        len(encoded)
    )