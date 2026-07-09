import os
import sys
from pathlib import Path

import psycopg2
import requests
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


# ── chunking ────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping windows of roughly *size* characters.

    Overlap ensures that a sentence sitting right at a boundary still
    appears complete in at least one chunk, so the retriever can find it.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


# ── embeddings ──────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Call Ollama's local API to embed *text* with nomic-embed-text.

    The model maps text into a 768-dimensional vector where similar
    meanings end up close together — that's what lets us "search by
    meaning" later instead of relying on exact keyword matches.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ── database ────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id       SERIAL PRIMARY KEY,
                source   TEXT NOT NULL,
                content  TEXT NOT NULL,
                embedding VECTOR({EMBED_DIM})
            )
        """)
    conn.commit()


def insert_chunk(cur, source: str, content: str, embedding: list[float]):
    cur.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s)",
        (source, content, embedding),
    )


# ── main ────────────────────────────────────────────────────────────

def ingest():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        sys.exit(1)

    conn = get_connection()
    init_db(conn)
    register_vector(conn)

    total_chunks = 0

    for pdf_path in pdfs:
        print(f"\n📄 Processing: {pdf_path.name}")
        reader = PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        chunks = chunk_text(full_text)
        print(f"   {len(reader.pages)} pages → {len(chunks)} chunks")

        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks, 1):
                embedding = get_embedding(chunk)
                insert_chunk(cur, pdf_path.name, chunk, embedding)
                if i % 10 == 0 or i == len(chunks):
                    print(f"   embedded {i}/{len(chunks)}")
        conn.commit()
        total_chunks += len(chunks)

    conn.close()
    print(f"\nDone — {total_chunks} chunks from {len(pdfs)} PDF(s) stored.")


if __name__ == "__main__":
    ingest()
