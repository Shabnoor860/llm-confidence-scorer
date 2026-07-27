"""
retrieval_utils.py
-------------------
Retrieval-Augmented Generation (RAG) support: fetches real evidence from
Wikipedia for a claim, and measures how well the claim is lexically
"grounded" in that evidence. This lets the app compare:

    closed-book verdict (model/LLM guesses with no evidence)
        vs.
    RAG-grounded verdict (model/LLM answers WITH retrieved evidence)

which is the actual current research direction in hallucination detection,
rather than only scoring claims against a closed-book classifier.

No API key needed — uses Wikipedia's public API.
"""

from __future__ import annotations

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WIKI_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_evidence(claim: str, max_results: int = 3, timeout: int = 10) -> list[dict]:
    """
    Retrieves up to `max_results` Wikipedia page summaries relevant to the claim.
    Returns [] on any failure (no internet, rate limit, blocked host, etc.)
    so callers can degrade gracefully instead of crashing.
    """
    try:
        search_resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": claim,
                "format": "json",
                "srlimit": max_results,
            },
            timeout=timeout,
            headers={"User-Agent": "hallucination-scorer-research-project/1.0"},
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])

        evidence = []
        for hit in hits:
            title = hit["title"]
            extract_resp = requests.get(
                WIKI_API,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": title,
                    "format": "json",
                },
                timeout=timeout,
                headers={"User-Agent": "hallucination-scorer-research-project/1.0"},
            )
            extract_resp.raise_for_status()
            pages = extract_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "").strip()
                if extract:
                    evidence.append({
                        "title": title,
                        "snippet": extract[:800],
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    })
        return evidence
    except Exception:
        return []


def lexical_grounding_score(claim: str, evidence_snippets: list[str]) -> float:
    """
    TF-IDF cosine similarity between the claim and the concatenated evidence.
    0 = claim shares no vocabulary with retrieved evidence (ungrounded / likely
    unsupported by what was retrieved). 1 = very high lexical overlap.
    This is a cheap proxy for groundedness — not a substitute for an entailment
    model, but useful as a fast, dependency-light signal.
    """
    if not evidence_snippets:
        return 0.0
    combined_evidence = " ".join(evidence_snippets)
    if not combined_evidence.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([claim, combined_evidence])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except ValueError:
        # e.g. vocabulary empty after stop-word removal on very short claims
        return 0.0

