"""Shared by dataset building, training and evaluation, so the fine-tuned model sees
at inference exactly the prompt format it was trained on."""

import re

READER_PROMPT = """You answer questions using ONLY the context passages below. Each passage starts with its source label in brackets.
- After the facts you use, cite that passage's label in square brackets, copied exactly as it appears above the passage (the full path, plus any page or lines).
- If the passages do not contain the answer, reply exactly: I don't know based on the available sources.
- Be brief: one to three sentences. Never use outside knowledge."""

ABSTAIN = "I don't know based on the available sources."
ABSTAIN_RE = re.compile(r"don.t know|do not know|not (?:mentioned|available|included|found|present|specified|provided"
                        r"|contain)|no (?:information|data|mention)|(?:does|do|did) not (?:include|contain|cover|"
                        r"mention)|don.t mention", re.I)
CITED_RE = re.compile(r"\[([^\[\]\n]{2,300})\]")


def format_context(passages: list[tuple[str, str]]) -> str:
    """[(label, text)] -> the same layout search_knowledge_base returns to the agent."""
    return "\n\n---\n\n".join(f"[{label}]\n{text}" for label, text in passages)


def user_message(passages: list[tuple[str, str]], question: str) -> str:
    return f"Context:\n{format_context(passages)}\n\nQuestion: {question}"


def citations(answer: str) -> list[str]:
    return [c.strip() for c in CITED_RE.findall(answer)]
