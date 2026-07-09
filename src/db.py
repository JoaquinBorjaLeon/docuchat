import os

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()

EMBED_DIM = 768


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )
    return conn


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id        SERIAL PRIMARY KEY,
                source    TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding VECTOR({EMBED_DIM})
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_fts
            ON chunks USING GIN (to_tsvector('spanish', content))
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id                  SERIAL PRIMARY KEY,
                timestamp           TIMESTAMPTZ NOT NULL DEFAULT now(),
                provider            TEXT NOT NULL,
                question            TEXT NOT NULL,
                tools_used          TEXT[] NOT NULL DEFAULT '{}',
                num_chunks_retrieved INTEGER NOT NULL DEFAULT 0,
                latency_ms          INTEGER NOT NULL,
                answer              TEXT
            )
        """)
    conn.commit()
    register_vector(conn)
