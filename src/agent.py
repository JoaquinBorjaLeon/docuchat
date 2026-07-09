"""
Multi-provider agent with function calling.

Supports Ollama and Gemini as LLM backends. The agent loop sends the
user's question along with available tools, executes any tool the model
requests, feeds the result back, and repeats until the model produces a
final text answer (or we hit MAX_ROUNDS).
"""

import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

from src.rag import retrieve

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MAX_ROUNDS = 5

SYSTEM_INSTRUCTION = (
    "You are DocuChat, a helpful assistant. You have tools available. "
    "Use buscar_documentos to answer questions about the user's PDF "
    "documents. Use fecha_actual when the user asks about the current "
    "date or time. Always respond in the same language the user writes in."
)


# ════════════════════════════════════════════════════════════════════
# 1. TOOL REGISTRY — defined once, used by every provider
# ════════════════════════════════════════════════════════════════════

TOOLS = {
    "buscar_documentos": {
        "description": "Search the user's PDF documents by semantic and keyword relevance.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document chunks.",
                },
            },
            "required": ["query"],
        },
    },
    "fecha_actual": {
        "description": "Return the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def execute_tool(name: str, args: dict) -> str:
    if name == "buscar_documentos":
        chunks = retrieve(args["query"])
        results = [
            {"source": c["source"], "content": c["content"], "rrf_score": round(c["rrf_score"], 4)}
            for c in chunks
        ]
        return json.dumps(results, ensure_ascii=False)

    if name == "fecha_actual":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return json.dumps({"error": f"Unknown tool: {name}"})


# ════════════════════════════════════════════════════════════════════
# 2. PROVIDER ADAPTERS — translate the common schema per provider
# ════════════════════════════════════════════════════════════════════

# ── Ollama ──────────────────────────────────────────────────────────

def _ollama_tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in TOOLS.items()
    ]


def _ollama_chat(messages: list[dict]) -> dict:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": messages,
            "tools": _ollama_tool_specs(),
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _ollama_extract_tool_calls(response: dict) -> list[dict]:
    msg = response.get("message", {})
    raw_calls = msg.get("tool_calls", [])
    return [
        {
            "name": tc["function"]["name"],
            "args": tc["function"]["arguments"],
        }
        for tc in raw_calls
    ]


def _ollama_agent_loop(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_ROUNDS):
        try:
            response = _ollama_chat(messages)
        except requests.ConnectionError:
            return f"[Error] Cannot connect to Ollama at {OLLAMA_URL}. Is it running?"
        except requests.HTTPError as e:
            return f"[Error] Ollama returned an error: {e}"

        tool_calls = _ollama_extract_tool_calls(response)

        if not tool_calls:
            return response["message"].get("content", "")

        messages.append(response["message"])

        for tc in tool_calls:
            _print_tool_call(tc["name"], tc["args"])
            result = execute_tool(tc["name"], tc["args"])
            _print_tool_result(result)
            messages.append({"role": "tool", "content": result})

    return response["message"].get("content", "")


# ── Gemini ──────────────────────────────────────────────────────────

def _gemini_tool_specs():
    from google.genai import types as gtypes

    declarations = []
    for name, spec in TOOLS.items():
        props = spec["parameters"].get("properties", {})
        schema_props = {}
        for pname, pdef in props.items():
            schema_props[pname] = gtypes.Schema(
                type=gtypes.Type.STRING,
                description=pdef.get("description", ""),
            )

        declarations.append(gtypes.FunctionDeclaration(
            name=name,
            description=spec["description"],
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties=schema_props,
                required=spec["parameters"].get("required", []),
            ) if schema_props else None,
        ))
    return [gtypes.Tool(function_declarations=declarations)]


def _gemini_agent_loop(question: str) -> str:
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [gtypes.Content(
        role="user",
        parts=[gtypes.Part.from_text(text=question)],
    )]

    for _ in range(MAX_ROUNDS):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=gtypes.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=_gemini_tool_specs(),
                ),
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                return "[Error] Gemini quota exceeded. Wait a moment or switch to another provider (LLM_PROVIDER=ollama)."
            if "API_KEY" in err or "401" in err or "403" in err:
                return "[Error] Invalid or missing GEMINI_API_KEY. Check your .env file."
            return f"[Error] Gemini API error: {err}"

        part = response.candidates[0].content.parts[0]

        if part.function_call is None:
            return part.text or ""

        fc = part.function_call
        args = dict(fc.args) if fc.args else {}
        _print_tool_call(fc.name, args)
        result = execute_tool(fc.name, args)
        _print_tool_result(result)

        contents.append(response.candidates[0].content)
        contents.append(gtypes.Content(
            role="user",
            parts=[gtypes.Part.from_function_response(
                name=fc.name,
                response={"result": result},
            )],
        ))

    return part.text or ""


# ── Future: Claude ──────────────────────────────────────────────────
# def _claude_tool_specs() -> list[dict]:
#     Anthropic uses {"name", "description", "input_schema"} per tool.
#
# def _claude_agent_loop(question: str) -> str:
#     Use anthropic.Anthropic().messages.create() with tool_use blocks.


# ════════════════════════════════════════════════════════════════════
# 3. PUBLIC INTERFACE
# ════════════════════════════════════════════════════════════════════

PROVIDERS = {
    "ollama": _ollama_agent_loop,
    "gemini": _gemini_agent_loop,
    # "claude": _claude_agent_loop,
}


def agent_chat(question: str) -> str:
    loop_fn = PROVIDERS.get(LLM_PROVIDER)
    if loop_fn is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
            f"Choose from: {', '.join(PROVIDERS)}"
        )
    return loop_fn(question)


# ── CLI helpers ─────────────────────────────────────────────────────

def _print_tool_call(name: str, args: dict):
    args_str = json.dumps(args, ensure_ascii=False)
    print(f"   🔧 Tool call: {name}({args_str})")


def _print_tool_result(result: str):
    preview = result[:200].replace("\n", " ")
    suffix = "…" if len(result) > 200 else ""
    print(f"   ← {preview}{suffix}")


if __name__ == "__main__":
    print(f"DocuChat Agent — provider: {LLM_PROVIDER} (Ctrl+C to quit)\n")
    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            answer = agent_chat(question)
            print(f"\nAssistant: {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n¡Hasta luego!")
