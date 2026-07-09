from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import retrieve, generate

app = FastAPI(
    title="DocuChat",
    description="RAG conversational assistant over your PDF documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["¿Qué es el PMBOK?"])


class ChunkDetail(BaseModel):
    source: str
    content: str
    rrf_score: float
    cosine_distance: float | None = None
    ts_rank: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[ChunkDetail]


# ── endpoints ───────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    raw_chunks = retrieve(req.question)
    answer = generate(req.question, raw_chunks)

    sources = sorted(set(c["source"] for c in raw_chunks))
    chunks = [
        ChunkDetail(
            source=c["source"],
            content=c["content"],
            rrf_score=c["rrf_score"],
            cosine_distance=c.get("distance"),
            ts_rank=c.get("ts_rank"),
        )
        for c in raw_chunks
    ]

    return ChatResponse(answer=answer, sources=sources, chunks=chunks)
