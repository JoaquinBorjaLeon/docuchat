# DocuChat — a RAG conversational assistant

DocuChat lets you upload PDF documents and have a conversation about their content. It uses retrieval-augmented generation (RAG) to ground every answer in the actual text of your documents, reducing hallucinations and providing source references.

## Stack

- **Language**: Python
- **LLM**: Claude (Anthropic API)
- **Vector store**: PostgreSQL + pgvector
- **PDF parsing**: pypdf
- **API framework**: FastAPI + Uvicorn
