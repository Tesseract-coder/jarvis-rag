"""02: Embeddings and cosine similarity, by hand.

Run: python learn/02_embeddings_cosine.py

An embedding model maps text to a point in a 768-dimensional space so that
texts with similar *meaning* land close together. "Close" is measured with
cosine similarity: the cosine of the angle between two vectors.

    cos(a, b) = (a · b) / (|a| × |b|)      1 = same direction, 0 = unrelated, -1 = opposite
"""

import math

import numpy as np

from ollama_api import EMBED_MODEL, banner, embed


def cosine_by_hand(a: list[float], b: list[float]) -> float:
    """The formula with plain Python loops, so nothing is hidden."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    """All pairs at once: normalise each row to length 1, then one matrix multiply."""
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return unit @ unit.T


def print_matrix(labels: list[str], m: np.ndarray, width: int = 11) -> None:
    short = [lbl[: width - 1] for lbl in labels]
    print(" " * width + "".join(f"{s:>{width}}" for s in short))
    for lbl, row in zip(short, m):
        print(f"{lbl:<{width}}" + "".join(f"{v:>{width}.3f}" for v in row))


def what_is_an_embedding() -> None:
    banner(f"1. What an embedding looks like ({EMBED_MODEL})")
    v = embed(["car"])[0]
    print(f"  embed('car') -> {v.shape[0]} floats: [{', '.join(f'{x:.3f}' for x in v[:6])}, ...]")
    print(f"  length |v| = {np.linalg.norm(v):.4f}")
    print("  No single number 'means' anything. Meaning is in the direction of the whole vector.")


def words() -> None:
    banner("2. Words: car ≈ automobile")
    labels = ["car", "automobile", "vehicle", "truck", "banana", "apple", "democracy"]
    vecs = embed(labels)
    by_hand = cosine_by_hand(vecs[0].tolist(), vecs[1].tolist())
    print(f"  cosine_by_hand(car, automobile) = {by_hand:.4f}   (pure Python loops)")
    print(f"  numpy matrix version            = {cosine_matrix(vecs)[0, 1]:.4f}   (same number, 1 matmul)\n")
    print_matrix(labels, cosine_matrix(vecs))
    print("\n  Note the scale: unrelated words still score ~0.4-0.5, not 0. Only *relative*")
    print("  scores matter, which is why retrieval ranks results instead of using a fixed cutoff.")


def sentences() -> None:
    banner("3. Sentences: meaning beats word overlap")
    query = "How do I reset my password?"
    candidates = [
        "I forgot my login credentials and can't sign in.",   # same meaning, no shared words
        "Steps to change your account passphrase.",           # same meaning, synonyms
        "Password: a secret word, from Middle French.",       # shares 'password', different intent
        "The cafeteria serves lunch from noon.",              # unrelated
    ]
    vecs = embed([query] + candidates)
    scores = cosine_matrix(vecs)[0, 1:]
    print(f"  query: {query!r}")
    for s, c in sorted(zip(scores, candidates), reverse=True):
        print(f"    {s:.3f}  {c}")
    print("\n  A keyword search would rank the dictionary definition first (it contains 'password').")
    print("  Embeddings match intent. This is 'semantic search'.")


def limits() -> None:
    banner("4. What embeddings DON'T capture: negation and facts")
    pairs = [
        ("The robot is safe to use near people.", "The robot is not safe to use near people."),
        ("The battery lasts 6.5 hours.", "The battery lasts 9 hours."),
    ]
    for a, b in pairs:
        print(f"  {cosine_matrix(embed([a, b]))[0, 1]:.3f}  {a!r} vs {b!r}")
    print("\n  Opposite meanings score ~0.95; contradictory numbers still score high. Embeddings")
    print("  encode *topic*, not truth, and can't tell 6.5 from 9 reliably.")
    print("  Retrieval finds the right passage, and the LLM must still read it carefully.")


def prefixes() -> None:
    banner("5. nomic-embed-text task prefixes")
    q = "How many vacation days do employees get?"
    docs = ["Every full-time employee receives 25 days of paid annual leave per year.",
            "The cafeteria serves lunch from 12:00 to 14:30."]
    for qp, dp in [("", ""), ("search_query: ", "search_document: ")]:
        s = cosine_matrix(embed([qp + q] + [dp + d for d in docs]))[0, 1:]
        label = "with prefixes   " if qp else "without prefixes"
        print(f"  {label}: relevant={s[0]:.3f}  irrelevant={s[1]:.3f}  gap={s[0] - s[1]:.3f}")
    print("\n  nomic was trained with 'search_query: ' on questions and 'search_document: ' on")
    print("  passages. Using them is part of the model's contract. Every model has its own")
    print("  conventions (bge uses 'Represent this sentence...'), so read the model card.")


if __name__ == "__main__":
    what_is_an_embedding()
    words()
    sentences()
    limits()
    prefixes()
