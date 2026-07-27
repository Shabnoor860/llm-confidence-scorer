"""
tests/test_retrieval_utils.py — run with: pytest
Note: fetch_wikipedia_evidence requires live internet access to Wikipedia;
these tests check the parts that don't (or gracefully handle no access).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import retrieval_utils as ru


def test_grounding_score_high_overlap():
    claim = "The Eiffel Tower is in Paris."
    evidence = ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France."]
    score = ru.lexical_grounding_score(claim, evidence)
    assert score > 0.3


def test_grounding_score_no_overlap():
    claim = "Dolphins are highly intelligent marine mammals."
    evidence = ["The stock market fluctuated significantly during the fiscal quarter."]
    score = ru.lexical_grounding_score(claim, evidence)
    assert score < 0.2


def test_grounding_score_empty_evidence():
    score = ru.lexical_grounding_score("Any claim.", [])
    assert score == 0.0


def test_fetch_wikipedia_evidence_never_raises():
    # Should return [] rather than raise, regardless of network availability
    result = ru.fetch_wikipedia_evidence("Eiffel Tower", max_results=1)
    assert isinstance(result, list)

