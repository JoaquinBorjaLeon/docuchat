from src.db import get_connection, init_db


def log_query(
    provider: str,
    question: str,
    tools_used: list[str],
    num_chunks: int,
    latency_ms: int,
    answer: str | None = None,
):
    conn = get_connection()
    init_db(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO query_logs
                (provider, question, tools_used, num_chunks_retrieved, latency_ms, answer)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (provider, question, tools_used, num_chunks, latency_ms, answer),
        )
    conn.commit()
    conn.close()


def print_stats():
    conn = get_connection()
    init_db(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM query_logs")
        total = cur.fetchone()[0]

        if total == 0:
            print("No queries logged yet.")
            conn.close()
            return

        cur.execute("""
            SELECT provider,
                   count(*) AS queries,
                   round(avg(latency_ms)) AS avg_ms,
                   round(avg(num_chunks_retrieved), 1) AS avg_chunks
            FROM query_logs
            GROUP BY provider
            ORDER BY queries DESC
        """)
        provider_rows = cur.fetchall()

        cur.execute("""
            SELECT tool, count(*) AS uses
            FROM query_logs, unnest(tools_used) AS tool
            GROUP BY tool
            ORDER BY uses DESC
        """)
        tool_rows = cur.fetchall()

    conn.close()

    print(f"\n📊 DocuChat Stats — {total} queries total\n")

    print("By provider:")
    print(f"  {'Provider':<12} {'Queries':>8} {'Avg ms':>8} {'Avg chunks':>11}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*11}")
    for provider, queries, avg_ms, avg_chunks in provider_rows:
        print(f"  {provider:<12} {queries:>8} {avg_ms:>8} {avg_chunks:>11}")

    print("\nTool usage:")
    for tool, uses in tool_rows:
        print(f"  {tool}: {uses}")


if __name__ == "__main__":
    print_stats()
