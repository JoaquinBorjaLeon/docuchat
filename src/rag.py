import os

import requests
from dotenv import load_dotenv

from src.db import get_connection, init_db
from src.embeddings import get_embedding

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")


# ── retrieval ───────────────────────────────────────────────────────

RRF_K = 60


def _vector_search(cur, embedding: list[float], n: int) -> list[dict]:
    cur.execute(
        """
        SELECT id, content, source, embedding <=> %s::vector AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s
        """,
        (embedding, n),
    )
    return [
        {"id": r[0], "content": r[1], "source": r[2], "distance": r[3]}
        for r in cur.fetchall()
    ]


def _lexical_search(cur, question: str, n: int) -> list[dict]:
    cur.execute(
        """
        SELECT id, content, source,
               ts_rank(to_tsvector('spanish', content),
                       plainto_tsquery('spanish', %s)) AS rank
        FROM chunks
        WHERE to_tsvector('spanish', content) @@ plainto_tsquery('spanish', %s)
        ORDER BY rank DESC
        LIMIT %s
        """,
        (question, question, n),
    )
    return [
        {"id": r[0], "content": r[1], "source": r[2], "ts_rank": r[3]}
        for r in cur.fetchall()
    ]


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    lexical_results: list[dict],
    k: int,
) -> list[dict]:
    scores: dict[int, float] = {}
    chunks_by_id: dict[int, dict] = {}

    for rank, chunk in enumerate(vector_results, 1):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunks_by_id[cid] = chunk

    for rank, chunk in enumerate(lexical_results, 1):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        if cid not in chunks_by_id:
            chunks_by_id[cid] = chunk

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

    results = []
    for cid in ranked_ids:
        chunk = chunks_by_id[cid]
        chunk["rrf_score"] = scores[cid]
        results.append(chunk)
    return results


def retrieve(question: str, k: int = 8) -> list[dict]:
    embedding = get_embedding(question)
    fetch_n = k * 3

    conn = get_connection()
    init_db(conn)

    with conn.cursor() as cur:
        vec_results = _vector_search(cur, embedding, fetch_n)
        lex_results = _lexical_search(cur, question, fetch_n)

    conn.close()

    return _reciprocal_rank_fusion(vec_results, lex_results, k)


# ── generation ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the context provided below. If the context does not contain enough "
    "information to answer, say 'I don't have enough information to answer "
    "that based on the provided documents.' Do not make up information."
)


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Source: {c['source']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def generate(question: str, chunks: list[dict]) -> str:
    """Build a grounded prompt and call the local LLM via Ollama.

    The prompt structure is deliberate:
      1. A system instruction that forbids inventing facts.
      2. The retrieved chunks as "context".
      3. The user's question.

    This forces the model to ground its answer in the retrieved text.
    Without this constraint, an LLM will happily fill gaps with plausible-
    sounding but fabricated information — a "hallucination".
    """
    context = build_context_block(chunks)

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n\n{context}\n\n"
                        f"---\n\nQuestion: {question}"
                    ),
                },
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── cli ─────────────────────────────────────────────────────────────

def ask(question: str) -> str:
    chunks = retrieve(question)

    print(f"\n🔍 Retrieved {len(chunks)} chunks (hybrid search):")
    for i, c in enumerate(chunks, 1):
        preview = c["content"][:80].replace("\n", " ")
        dist = f"cosine={c['distance']:.4f}" if "distance" in c else "—"
        ts = f"ts_rank={c['ts_rank']:.4f}" if "ts_rank" in c else "—"
        print(f"   {i}. [{c['source']}] rrf={c['rrf_score']:.4f} {dist} {ts} — {preview}…")

    sources = sorted(set(c["source"] for c in chunks))
    answer = generate(question, chunks)

    return f"{answer}\n\n📚 Sources: {', '.join(sources)}"


if __name__ == "__main__":
    print("DocuChat — ask anything about your documents (Ctrl+C to quit)\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        print(f"\n{ask(question)}\n")
