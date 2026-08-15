import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# ATTENTION HEAD
# ==========================================

class Head(nn.Module):

    def __init__(
        self,
        embed_size,
        head_size,
        block_size,
        dropout=0.1
    ):

        super().__init__()

        self.key = nn.Linear(
            embed_size,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            embed_size,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embed_size,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)

        q = self.query(x)


        # ----------------------------------
        # Attention scores
        # ----------------------------------

        attention = (
            q @ k.transpose(-2, -1)
        )

        attention = attention * (
            k.shape[-1] ** -0.5
        )


        # ----------------------------------
        # Causal mask
        # ----------------------------------

        attention = attention.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )


        # ----------------------------------
        # Softmax
        # ----------------------------------

        attention = F.softmax(
            attention,
            dim=-1
        )


        attention = self.dropout(
            attention
        )


        # ----------------------------------
        # Values
        # ----------------------------------

        v = self.value(x)

        output = attention @ v

        return output


# ==========================================
# MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_size,
        num_heads,
        block_size,
        dropout=0.1
    ):

        super().__init__()

        head_size = (
            embed_size // num_heads
        )

        self.heads = nn.ModuleList(
            [
                Head(
                    embed_size,
                    head_size,
                    block_size,
                    dropout
                )
                for _ in range(num_heads)
            ]
        )

        self.projection = nn.Linear(
            embed_size,
            embed_size
        )

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        output = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        output = self.projection(
            output
        )

        output = self.dropout(
            output
        )

        return output


# ==========================================
# FEED FORWARD
# ==========================================

class FeedForward(nn.Module):

    def __init__(
        self,
        embed_size,
        dropout=0.1
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_size,
                4 * embed_size
            ),

            nn.ReLU(),

            nn.Linear(
                4 * embed_size,
                embed_size
            ),

            nn.Dropout(
                dropout
            )
        )


    def forward(self, x):

        return self.network(x)


# ==========================================
# TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_size,
        num_heads,
        block_size,
        dropout=0.1
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            embed_size,
            num_heads,
            block_size,
            dropout
        )

        self.feed_forward = FeedForward(
            embed_size,
            dropout
        )

        self.norm1 = nn.LayerNorm(
            embed_size
        )

        self.norm2 = nn.LayerNorm(
            embed_size
        )


    def forward(self, x):

        # Pre-normalization
        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


# ==========================================
# BRADYAI
# ==========================================

class BradyAI(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_heads=4,
        num_layers=4,
        block_size=128,
        dropout=0.1
    ):

        super().__init__()

        self.block_size = block_size


        # ----------------------------------
        # Token embeddings
        # ----------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_size
        )


        # ----------------------------------
        # Position embeddings
        # ----------------------------------

        self.position_embedding = nn.Embedding(
            block_size,
            embed_size
        )


        # ----------------------------------
        # Transformer blocks
        # ----------------------------------

        self.blocks = nn.Sequential(

            *[
                TransformerBlock(
                    embed_size,
                    num_heads,
                    block_size,
                    dropout
                )

                for _ in range(
                    num_layers
                )
            ]
        )


        # ----------------------------------
        # Final normalization
        # ----------------------------------

        self.final_norm = nn.LayerNorm(
            embed_size
        )


        # ----------------------------------
        # Output layer
        # ----------------------------------

        self.output = nn.Linear(
            embed_size,
            vocab_size
        )


    # ======================================
    # FORWARD
    # ======================================

    def forward(
        self,
        tokens,
        targets=None
    ):

        B, T = tokens.shape


        # ----------------------------------
        # Positions
        # ----------------------------------

        positions = torch.arange(
            T,
            device=tokens.device
        )


        # ----------------------------------
        # Embeddings
        # ----------------------------------

        token_embeddings = (
            self.token_embedding(tokens)
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )


        x = (
            token_embeddings
            + position_embeddings
        )


        # ----------------------------------
        # Transformer
        # ----------------------------------

        x = self.blocks(x)


        # ----------------------------------
        # Final normalization
        # ----------------------------------

        x = self.final_norm(x)


        # ----------------------------------
        # Output logits
        # ----------------------------------

        logits = self.output(x)


        # ==================================
        # LOSS
        # ==================================

        loss = None


        if targets is not None:

            B, T, C = logits.shape


            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )


            # IMPORTANT:
            # Ignore <PAD> tokens.
            #
            # <PAD> has ID 0 because
            # tokenizer.py defines:
            #
            # <PAD> = 0
            #

            loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=0
            )


        return logits, loss