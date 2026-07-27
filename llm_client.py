"""
llm_client.py
-------------
Thin wrapper around the Groq API (free tier, OpenAI-compatible) so the app
can query a REAL LLM and compare its self-reported confidence against the
in-house classifier — instead of only scoring claims with a from-scratch
TF-IDF model and calling that "LLM hallucination detection".

If no API key is configured, functions return None and the app falls back
to classifier-only mode without crashing.

Setup:
    1. Get a free API key: https://console.groq.com/keys
    2. Set it as an environment variable: GROQ_API_KEY=your_key_here
       (locally: a .env file or shell export; on Streamlit Cloud: Settings > Secrets)
"""

from __future__ import annotations

import os
import json
import re
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"


def is_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def query_llm_verdict(claim: str, model: str = DEFAULT_MODEL, timeout: int = 15) -> dict | None:
    """
    Asks a real LLM to judge a claim and self-report its confidence.
    Returns {"verdict": "SUPPORTS"/"REFUTES"/"NOT ENOUGH INFO",
             "self_confidence": float 0-1, "raw": str} or None if unavailable.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    prompt = (
        "You are a fact-checking assistant. Judge the following claim.\n"
        f"Claim: \"{claim}\"\n\n"
        "Respond ONLY with a JSON object, no other text, in exactly this form:\n"
        '{"verdict": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO", '
        '"self_confidence": <number between 0 and 1>}'
    )

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {"verdict": "PARSE_ERROR", "self_confidence": None, "raw": content}

        parsed = json.loads(match.group(0))
        return {
            "verdict": parsed.get("verdict", "UNKNOWN"),
            "self_confidence": float(parsed.get("self_confidence", 0.5)),
            "raw": content,
        }
    except Exception as e:
        return {"verdict": "ERROR", "self_confidence": None, "raw": str(e)}


def query_llm_with_evidence(claim: str, evidence_snippets: list[str],
                             model: str = DEFAULT_MODEL, timeout: int = 15) -> dict | None:
    """
    Same as query_llm_verdict, but gives the LLM retrieved evidence to ground
    its answer in — the actual RAG setup. Comparing this against the
    closed-book verdict (no evidence) shows whether retrieval changes the
    model's verdict/confidence, i.e. whether grounding reduces hallucination.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    evidence_text = "\n".join(f"- {s}" for s in evidence_snippets) if evidence_snippets else "(no evidence retrieved)"

    prompt = (
        "You are a fact-checking assistant. Judge the claim using ONLY the "
        "evidence provided below. If the evidence doesn't address the claim, "
        "say NOT ENOUGH INFO.\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Claim: \"{claim}\"\n\n"
        "Respond ONLY with a JSON object, no other text, in exactly this form:\n"
        '{"verdict": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO", '
        '"self_confidence": <number between 0 and 1>}'
    )

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {"verdict": "PARSE_ERROR", "self_confidence": None, "raw": content}

        parsed = json.loads(match.group(0))
        return {
            "verdict": parsed.get("verdict", "UNKNOWN"),
            "self_confidence": float(parsed.get("self_confidence", 0.5)),
            "raw": content,
        }
    except Exception as e:
        return {"verdict": "ERROR", "self_confidence": None, "raw": str(e)}

