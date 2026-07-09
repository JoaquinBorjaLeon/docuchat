import sys
from pathlib import Path

from pypdf import PdfReader

from src.db import get_connection, init_db
from src.embeddings import get_embedding

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def insert_chunk(cur, source: str, content: str, embedding: list[float]):
    cur.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s)",
        (source, content, embedding),
    )


def ingest():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        sys.exit(1)

    conn = get_connection()
    init_db(conn)

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
