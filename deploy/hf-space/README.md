---
title: JARVIS RAG
emoji: 🤖
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Agentic RAG over docs, code and SQL tables, with checked citations
---

# JARVIS: agentic RAG demo

A tool-calling agent that answers from a small fictional company's knowledge base
(Halcyon Robotics: handbook, product specs, a safety report PDF, a maintenance manual
DOCX, its own Python code, and two CSV tables queried with SQL). It cites every fact,
and each citation is checked against what the tools actually returned. It says
"I don't know" when the documents don't say.

- **Open the app at `/ui`** (the Space root redirects there). Log in with the password
  the owner shared. If the login doesn't stick inside the huggingface.co page, open the
  direct `*.hf.space` URL: browsers block cookies in third-party iframes.
- Try: *"What was the total revenue per region?"*, *"What does error E-217 mean, and how
  many tickets have it?"*, *"Where is the chunk id hash computed?"*, *"Who is the CEO?"*
- This deployment is **read-only** (no uploads) and **ephemeral** (conversations reset
  when the Space restarts).
- Retrieval: BM25 + bge-small vectors, fused, then reranked by a cross-encoder. The LLM
  is a hosted free-tier model, so questions and retrieved passages are sent to that
  provider.

Source, docs and a fully local (Ollama) setup: see the project repository.
