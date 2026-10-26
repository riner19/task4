from __future__ import annotations
import json
import os
import random
import time

import tools as T

BASE_URL = os.getenv("LLM_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_ROUNDS = 8       
MAX_RETRIES = 4 
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

SYSTEM = """You answer questions about a bookstore's order data.

You cannot see the data. The only way to learn anything about it is to call the
tools provided. Follow these rules without exception:

1. Every number, name, date and id in your answer must come from a tool result in
   this conversation. Never calculate, estimate, extrapolate or recall a figure.
2. Never guess a count. To count things matching a condition, call the relevant
   search tool and read its "matched" field, which is the true size of the result
   set. The "rows" list is only a sample and is usually shorter than "matched".
3. If the tools cannot answer the question, say exactly what you could not
   determine and why. Do not fill the gap with a plausible-sounding number.
4. If a tool returns {"error": ...}, either fix the arguments and retry, or tell
   the user what went wrong. Do not invent the answer.
5. Call several tools when a question needs several facts. Do not answer until you
   have the facts.

Conventions in this data:
- Money is US dollars. Prices given in euros were converted at 1 EUR = 1.2 USD.
- Dates are YYYY-MM-DD.
- "Unique users" means real people after reconciling duplicate records. One person
  can own several user_ids; report all of them when identifying someone.
- An "author set" is the exact group of authors credited on a book. A pair writing
  together is a different set from either of them writing alone.

Answer in plain prose, briefly. State the figures plainly; no tables unless asked.
When a question is about one store, say which store the figures are for."""


class NoProvider(RuntimeError):
    pass


def _client():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise NoProvider(
            "No API key found. Set GEMINI_API_KEY in the environment - get one free at "
            "aistudio.google.com - then restart. On Render: Environment -> Add "
            "Environment Variable.")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=BASE_URL)


def _create(client, **kw):
    """One API call, retried with exponential backoff.

    The free tier returns 429 (rate limited) and 503 (model busy) fairly often. Both
    are transient, so retry rather than surfacing a failure - a question that dies on
    a 503 looks like a broken app to whoever is asking it.
    """
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(**kw)
        except Exception as e:                       # SDK raises a typed error per status
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None)
            if status == 429 and "per day" in str(e).lower():
                raise
            if status not in RETRY_STATUS or attempt == MAX_RETRIES - 1:
                raise
            last = e
            time.sleep((2 ** attempt) + random.random())   # 1s, 2s, 4s + jitter
    raise last


def ask(wh: T.Warehouse, question: str, history=None, default_store: str = None):
    """Run one question through the tool loop.

    Returns {"answer": str, "trace": [{"tool":.., "args":.., "result":..}], "error": str|None}
    """
    client = _client()
    schema = T.openai_schema()

    system = SYSTEM
    if default_store:
        system += (f"\n\nThe user is currently looking at {default_store}. If a question "
                   f"does not name a store, assume {default_store} and say so.")

    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in (history or [])]
    messages.append({"role": "user", "content": question})
    trace = []

    for _ in range(MAX_ROUNDS):
        resp = _create(client,
                       model=os.getenv("LLM_MODEL") or DEFAULT_MODEL,
                       messages=messages, tools=schema, max_tokens=1200)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return {"answer": (msg.content or "").strip(), "trace": trace, "error": None}

        messages.append(msg.model_dump(exclude_none=True))

        for c in msg.tool_calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = T.call_tool(wh, c.function.name, args)
            trace.append({"tool": c.function.name, "args": args, "result": out})
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": json.dumps(out, default=str)})

    return {"answer": "", "trace": trace,
            "error": f"Stopped after {MAX_ROUNDS} tool rounds without a final answer."}