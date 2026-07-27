#!/bin/bash
set -e

echo "Creating project files..."

cat > "app.py" << 'PYEOF_FILE_MARKER'
"""
app.py — Streamlit UI for the LLM Hallucination & Confidence Scorer.

All actual logic lives in model_utils.py (classifier, calibration, metrics)
and llm_client.py (optional live LLM query). This file only wires up the
interface, so the logic can be unit-tested independently of Streamlit.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import streamlit as st
import shap

import model_utils as mu
import llm_client
import retrieval_utils as ru

st.set_page_config(page_title="LLM Confidence Scorer", page_icon="🔍", layout="wide")
st.title("🔍 LLM Hallucination & Confidence Scorer")
st.caption(
    "Compares a from-scratch confidence classifier against a real LLM's "
    "self-reported confidence, measures calibration, tests a fix for it, "
    "checks domain bias, and explains individual predictions with SHAP."
)

# ---------------------------------------------------------------------------
# Load data + train models (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_data():
    return mu.load_data()


@st.cache_resource(show_spinner=False)
def _train(df):
    result = mu.train_model(df)
    baseline = mu.train_baseline(result["X_train"], result["y_train"],
                                  result["X_test"], result["y_test"])
    return result, baseline


try:
    with st.spinner("Loading dataset and training models..."):
        df, data_source = _load_data()
        result, baseline = _train(df)
except Exception as e:
    st.error(f"Setup failed: {e}")
    st.stop()

model = result["model"]
vectorizer = result["vectorizer"]
y_test = result["y_test"]
test_probs = result["test_probs"]
test_logits = result["test_logits"]
acc = result["accuracy"]

st.info(f"**Data source:** {data_source}")
c1, c2 = st.columns(2)
c1.metric("Neural Net Test Accuracy", f"{acc*100:.2f}%")
c2.metric("Logistic Regression Baseline Accuracy", f"{baseline['accuracy']*100:.2f}%")

if not llm_client.is_configured():
    st.warning(
        "No GROQ_API_KEY found — running in classifier-only mode. "
        "Set the GROQ_API_KEY environment variable to also query a real "
        "LLM and compare it against the in-house model (see README)."
    )

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    ["Claim Checker", "Metrics", "Calibration", "Calibration Fix",
     "Bias Detection", "Explainability", "RAG Groundedness",
     "Self-Verification Cascade", "Selective Prediction", "Robustness"]
)

# --- Tab 1: Claim Checker (in-house classifier + optional real LLM) ---
with tab1:
    st.subheader("Enter a claim to check")
    claim = st.text_input("Claim", placeholder="e.g. The Eiffel Tower is in Paris.")

    if claim:
        def score(text):
            feats = vectorizer.transform([text]).toarray()
            t = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                return torch.sigmoid(model(t)).item()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**In-house classifier**")
            s1 = score(claim)
            label1 = "Reliable" if s1 > 0.5 else "Likely Hallucinated"
            st.markdown(f"Confidence: `{s1:.4f}` — **{label1}**")

            s2 = score(f"Is this true? {claim}")
            label2 = "Reliable" if s2 > 0.5 else "Likely Hallucinated"
            st.markdown(f"After self-verification prompt: `{s2:.4f}` — **{label2}**")
            if abs(s1 - s2) > 0.05:
                st.warning("Confidence shifted after self-verification.")

        with col_b:
            st.markdown("**Real LLM (Groq)**")
            if llm_client.is_configured():
                with st.spinner("Querying LLM..."):
                    verdict = llm_client.query_llm_verdict(claim)
                if verdict and verdict["self_confidence"] is not None:
                    st.markdown(f"Verdict: **{verdict['verdict']}**")
                    st.markdown(f"Self-reported confidence: `{verdict['self_confidence']:.2f}`")
                else:
                    st.error(f"LLM query failed: {verdict['raw'] if verdict else 'no response'}")
            else:
                st.info("Add a GROQ_API_KEY to enable this comparison.")

# --- Tab 2: Metrics (precision/recall/F1/confusion matrix + baseline) ---
with tab2:
    st.subheader("Classification Metrics — Neural Net vs Logistic Regression Baseline")
    nn_preds = (test_probs > 0.5).astype(int)
    baseline_preds = (baseline["probs"] > 0.5).astype(int)

    nn_metrics = mu.classification_metrics(y_test, nn_preds)
    base_metrics = mu.classification_metrics(y_test, baseline_preds)

    metrics_df = pd.DataFrame({
        "Neural Net": [nn_metrics["precision"], nn_metrics["recall"], nn_metrics["f1"], acc],
        "Logistic Regression": [base_metrics["precision"], base_metrics["recall"],
                                 base_metrics["f1"], baseline["accuracy"]],
    }, index=["Precision", "Recall", "F1", "Accuracy"])
    st.dataframe(metrics_df.style.format("{:.3f}"))

    if abs(acc - baseline["accuracy"]) < 0.02:
        st.caption(
            "The simple baseline performs comparably to the neural net on this data — "
            "the added complexity isn't clearly justified by accuracy alone here."
        )
    else:
        winner = "Neural Net" if acc > baseline["accuracy"] else "Logistic Regression"
        st.caption(f"{winner} outperforms the other by a meaningful margin on this data.")

    st.markdown("**Bootstrap 95% Confidence Interval (Neural Net Accuracy)**")
    mean_acc, lo, hi = mu.bootstrap_accuracy_ci(y_test, nn_preds, n_bootstrap=1000)
    st.write(f"{mean_acc*100:.2f}% (95% CI: {lo*100:.2f}% – {hi*100:.2f}%)")
    st.caption(
        "A single train/test split gives one noisy point estimate. This resamples "
        "the test set 1000 times to show a plausible range instead."
    )

    st.markdown("**Confusion Matrix (Neural Net)**")
    cm = np.array(nn_metrics["confusion_matrix"])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    ax_cm.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Refutes", "Supports"])
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["Refutes", "Supports"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# --- Tab 3: Calibration (measurement) ---
with tab3:
    st.subheader("Confidence Score Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(test_probs[y_test == 1], bins=20, alpha=0.6, label="Reliable (SUPPORTS)", color="green")
    ax.hist(test_probs[y_test == 0], bins=20, alpha=0.6, label="Hallucinated (REFUTES)", color="red")
    ax.set_xlabel("Confidence Score"); ax.set_ylabel("Number of Claims")
    ax.legend()
    st.pyplot(fig)

    ece_before, bins_before = mu.expected_calibration_error(test_probs, y_test)
    st.metric("Expected Calibration Error (ECE)", f"{ece_before:.4f}",
               help="0 = perfectly calibrated. Higher = confidence doesn't match real accuracy.")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if bins_before:
        confs, accs, _ = zip(*bins_before)
        ax2.plot(confs, accs, "bo-", label="Model (before fix)")
    ax2.set_xlabel("Mean Confidence"); ax2.set_ylabel("Actual Accuracy")
    ax2.set_title("Reliability Diagram")
    ax2.legend()
    st.pyplot(fig2)

# --- Tab 4: Calibration Fix ---
with tab4:
    st.subheader("Does Temperature Scaling Fix Calibration?")
    st.caption(
        "Temperature scaling (Guo et al., 2017) rescales logits by a single "
        "learned scalar T before the sigmoid, without changing accuracy."
    )
    T = mu.fit_temperature(test_logits, y_test)
    calibrated_probs = 1 / (1 + np.exp(-test_logits / T))
    ece_after, bins_after = mu.expected_calibration_error(calibrated_probs, y_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Learned Temperature (T)", f"{T:.3f}")
    col2.metric("ECE Before", f"{ece_before:.4f}")
    col3.metric("ECE After", f"{ece_after:.4f}",
                delta=f"{ece_after - ece_before:+.4f}", delta_color="inverse")

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if bins_before:
        confs_b, accs_b, _ = zip(*bins_before)
        ax3.plot(confs_b, accs_b, "ro--", label="Before scaling", alpha=0.7)
    if bins_after:
        confs_a, accs_a, _ = zip(*bins_after)
        ax3.plot(confs_a, accs_a, "go-", label="After scaling")
    ax3.set_xlabel("Mean Confidence"); ax3.set_ylabel("Actual Accuracy")
    ax3.set_title("Reliability Diagram: Before vs After Temperature Scaling")
    ax3.legend()
    st.pyplot(fig3)

    if ece_after < ece_before:
        st.success(f"Temperature scaling reduced ECE by {ece_before - ece_after:.4f}.")
    else:
        st.warning("Temperature scaling did not improve calibration on this data.")

# --- Tab 5: Bias Detection ---
with tab5:
    st.subheader("Accuracy Across Domains")
    bias = mu.bias_analysis(df, model, vectorizer)
    if bias:
        domains = list(bias.keys())
        accs = [v[0] for v in bias.values()]
        counts = [v[1] for v in bias.values()]
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.bar(domains, accs, color="steelblue")
        for i, c in enumerate(counts):
            ax4.text(i, accs[i] + 1, f"n={c}", ha="center", fontsize=8)
        ax4.set_ylabel("Accuracy (%)"); ax4.set_ylim(0, 100)
        st.pyplot(fig4)
        st.caption("Domains with very small n are noisy — treat as directional, not conclusive.")
    else:
        st.info("Not enough claims per domain in the current dataset to compare.")

# --- Tab 6: Explainability ---
with tab6:
    st.subheader("Why did the model make this prediction?")
    st.caption("SHAP KernelExplainer — which words influenced the score most.")
    sample_claim = st.text_input(
        "Enter claim for explanation", placeholder="e.g. Einstein won the Nobel Prize.",
        key="shap_input"
    )
    run = st.button("Explain this claim")

    if sample_claim and run:
        with st.spinner("Computing SHAP values..."):
            try:
                bg_size = min(30, len(df))
                X_bg = vectorizer.transform(df["claim"].iloc[:bg_size]).toarray()
                X_sample = vectorizer.transform([sample_claim]).toarray()

                def model_predict(x):
                    t = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        return torch.sigmoid(model(t)).numpy()

                explainer = shap.KernelExplainer(model_predict, X_bg)
                shap_vals = explainer.shap_values(X_sample, nsamples=100)

                arr = np.array(shap_vals)
                if arr.ndim == 3:
                    vals = arr[0, :, 0]
                elif arr.ndim == 2:
                    vals = arr[0]
                else:
                    vals = arr.flatten()

                feature_names = vectorizer.get_feature_names_out()
                shap_df = pd.DataFrame({"word": feature_names, "shap": vals}) \
                    .sort_values("shap", key=abs, ascending=False).head(10)

                fig5, ax5 = plt.subplots(figsize=(8, 4))
                colors = ["green" if v > 0 else "red" for v in shap_df["shap"]]
                ax5.barh(shap_df["word"], shap_df["shap"], color=colors)
                ax5.set_title("Top Words Influencing Prediction")
                ax5.set_xlabel("SHAP Value")
                ax5.invert_yaxis()
                st.pyplot(fig5)
            except Exception as e:
                st.error(f"Explainability failed: {e}")

# --- Tab 7: RAG Groundedness ---
with tab7:
    st.subheader("Is the claim grounded in retrieved evidence?")
    st.caption(
        "Retrieves real Wikipedia evidence for the claim (RAG), then compares "
        "a closed-book verdict (no evidence) against a RAG-grounded verdict "
        "(answer using only the retrieved evidence) — testing whether "
        "retrieval actually reduces hallucination, rather than assuming it does."
    )
    rag_claim = st.text_input("Claim to check against Wikipedia", key="rag_input")
    rag_run = st.button("Retrieve evidence & check")

    if rag_claim and rag_run:
        with st.spinner("Retrieving evidence from Wikipedia..."):
            evidence = ru.fetch_wikipedia_evidence(rag_claim, max_results=3)

        if not evidence:
            st.warning(
                "No evidence could be retrieved (no internet access, rate limit, "
                "or Wikipedia unreachable from this environment). Grounding checks "
                "need live internet access to Wikipedia's API."
            )
        else:
            for e in evidence:
                with st.expander(f"📄 {e['title']}"):
                    st.write(e["snippet"])
                    st.caption(e["url"])

            snippets = [e["snippet"] for e in evidence]
            grounding = ru.lexical_grounding_score(rag_claim, snippets)
            st.metric("Lexical Grounding Score", f"{grounding:.3f}",
                       help="0 = claim shares no vocabulary with retrieved evidence. "
                            "1 = high lexical overlap. A cheap proxy, not a full "
                            "entailment check.")

            if llm_client.is_configured():
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Closed-book (no evidence)**")
                    closed = llm_client.query_llm_verdict(rag_claim)
                    if closed and closed["self_confidence"] is not None:
                        st.write(f"Verdict: **{closed['verdict']}**")
                        st.write(f"Confidence: `{closed['self_confidence']:.2f}`")
                with col_b:
                    st.markdown("**RAG-grounded (with evidence)**")
                    grounded = llm_client.query_llm_with_evidence(rag_claim, snippets)
                    if grounded and grounded["self_confidence"] is not None:
                        st.write(f"Verdict: **{grounded['verdict']}**")
                        st.write(f"Confidence: `{grounded['self_confidence']:.2f}`")

                if closed and grounded and closed.get("verdict") != grounded.get("verdict"):
                    st.warning("Verdict changed when evidence was provided — retrieval altered the answer.")
            else:
                st.info("Add a GROQ_API_KEY to compare closed-book vs RAG-grounded LLM verdicts.")

# --- Tab 8: Self-Verification Cascade ---
with tab8:
    st.subheader("Does repeated self-verification compound the drift?")
    st.caption(
        "Extends the single-round self-verification finding: wraps the claim in "
        "'Is this true?' repeatedly and tracks confidence across rounds."
    )
    cascade_claim = st.text_input("Claim", key="cascade_input",
                                    placeholder="e.g. The Eiffel Tower is in Paris.")
    rounds = st.slider("Verification rounds", 1, 8, 5)

    if cascade_claim:
        scores = mu.self_verification_cascade(model, vectorizer, cascade_claim, rounds=rounds)
        fig6, ax6 = plt.subplots(figsize=(7, 4))
        ax6.plot(range(len(scores)), scores, "o-", color="purple")
        ax6.set_xlabel("Verification Round (0 = original claim)")
        ax6.set_ylabel("Confidence Score")
        ax6.set_ylim(0, 1)
        ax6.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        st.pyplot(fig6)
        drift = scores[-1] - scores[0]
        st.write(f"Total drift after {rounds} rounds: `{drift:+.4f}`")

# --- Tab 9: Selective Prediction ---
with tab9:
    st.subheader("Risk-Coverage Curve: is confidence useful for deciding when to trust the model?")
    st.caption(
        "Even if raw confidence is miscalibrated in absolute terms, it may still "
        "be useful for deciding WHEN to trust a prediction — this checks whether "
        "restricting to higher-confidence claims (abstaining on the rest) improves accuracy."
    )
    curve = mu.selective_prediction_curve(test_probs, y_test)
    thresholds = [c[0] for c in curve]
    coverages = [c[1] for c in curve]
    accs_at_t = [c[2] if c[2] is not None else np.nan for c in curve]

    fig7, ax7 = plt.subplots(figsize=(7, 4))
    ax7.plot(coverages, accs_at_t, "o-", color="teal")
    ax7.set_xlabel("Coverage (fraction of claims answered)")
    ax7.set_ylabel("Accuracy on answered claims")
    ax7.set_title("Risk-Coverage Curve")
    st.pyplot(fig7)
    st.caption(
        "If accuracy rises as coverage drops, confidence is at least useful for "
        "ranking which claims to trust — even if it isn't perfectly calibrated."
    )

# --- Tab 10: Robustness (paraphrase consistency + error taxonomy) ---
with tab10:
    st.subheader("Paraphrase Consistency")
    st.caption(
        "Same fact, three different phrasings. Checks whether the model tracks "
        "MEANING or is sensitive to surface wording."
    )
    p1 = st.text_input("Phrasing 1", value="The Eiffel Tower is in Paris.")
    p2 = st.text_input("Phrasing 2", value="Paris has the Eiffel Tower.")
    p3 = st.text_input("Phrasing 3", value="The Eiffel Tower stands in the French capital.")

    if st.button("Check consistency"):
        pc = mu.paraphrase_consistency(model, vectorizer, [p1, p2, p3])
        fig8, ax8 = plt.subplots(figsize=(6, 3))
        ax8.bar(["Phrasing 1", "Phrasing 2", "Phrasing 3"], pc["scores"], color="orange")
        ax8.set_ylim(0, 1)
        ax8.axhline(0.5, color="gray", linestyle="--")
        st.pyplot(fig8)
        st.write(f"Std deviation across phrasings: `{pc['std']:.4f}`")
        st.write(f"Range: `{pc['range']:.4f}`")
        if pc["label_flip"]:
            st.warning("The predicted VERDICT itself changed across phrasings of the same fact.")
        else:
            st.success("Verdict stayed consistent across phrasings.")

    st.divider()
    st.subheader("Hallucination Error Taxonomy")
    st.caption(
        "Categorizes misclassified claims by simple surface patterns (negation, "
        "numeric/date content) — descriptive, not a causal explanation."
    )
    taxonomy = mu.error_taxonomy(df, model, vectorizer)
    if not taxonomy.empty:
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        ax9.bar(taxonomy["category"], taxonomy["count"], color="crimson")
        ax9.set_ylabel("Number of misclassified claims")
        plt.setp(ax9.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig9)
    else:
        st.info("No misclassified claims in the current test set to categorize.")

PYEOF_FILE_MARKER

cat > "model_utils.py" << 'PYEOF_FILE_MARKER'
"""
model_utils.py
--------------
Pure logic for the LLM Hallucination & Confidence Scorer project.
Deliberately has NO Streamlit import, so it can be unit-tested and
reused outside the web app (e.g. from a notebook or CLI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.optimize import minimize_scalar

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Offline fallback data — used only if the FEVER dataset can't be downloaded
# (no internet, dataset renamed, HF outage). Keeps the app runnable anywhere.
# ---------------------------------------------------------------------------
FALLBACK_CLAIMS = [
    ("The Eiffel Tower is located in Paris.", 1),
    ("The Eiffel Tower is located in Berlin.", 0),
    ("Water boils at 100 degrees Celsius at sea level.", 1),
    ("Water boils at 50 degrees Celsius at sea level.", 0),
    ("The Great Wall of China is visible from the Moon with the naked eye.", 0),
    ("Mount Everest is the tallest mountain above sea level on Earth.", 1),
    ("The Pacific Ocean is the largest ocean on Earth.", 1),
    ("The Atlantic Ocean is the largest ocean on Earth.", 0),
    ("Albert Einstein developed the theory of general relativity.", 1),
    ("Albert Einstein invented the light bulb.", 0),
    ("The chemical symbol for gold is Au.", 1),
    ("The chemical symbol for gold is Ag.", 0),
    ("The human body has 206 bones in adulthood.", 1),
    ("The human body has 300 bones in adulthood.", 0),
    ("World War II ended in 1945.", 1),
    ("World War II ended in 1939.", 0),
    ("The French Revolution began in 1789.", 1),
    ("The French Revolution began in 1850.", 0),
    ("Napoleon Bonaparte was exiled to the island of Saint Helena.", 1),
    ("Napoleon Bonaparte was exiled to Iceland.", 0),
    ("The Berlin Wall fell in 1989.", 1),
    ("The Berlin Wall fell in 1999.", 0),
    ("Canada shares a land border with the United States.", 1),
    ("Canada shares a land border with Brazil.", 0),
    ("The Nile is a river in Africa.", 1),
    ("The Nile is a river in South America.", 0),
    ("Mount Kilimanjaro is located in Tanzania.", 1),
    ("Mount Kilimanjaro is located in India.", 0),
    ("Australia is both a country and a continent.", 1),
    ("Australia is the largest country in Europe.", 0),
    ("The 2016 Olympics were held in Rio de Janeiro.", 1),
    ("The 2016 Olympics were held in Tokyo.", 0),
    ("Usain Bolt is a former Olympic sprinter from Jamaica.", 1),
    ("Usain Bolt is a former Olympic swimmer from Kenya.", 0),
    ("FIFA World Cup is held every four years.", 1),
    ("FIFA World Cup is held every year.", 0),
    ("Lionel Messi has won multiple Ballon d'Or awards.", 1),
    ("Lionel Messi has never played professional football.", 0),
    ("The United Nations was founded in 1945.", 1),
    ("The United Nations was founded in 1900.", 0),
    ("Barack Obama served as the 44th President of the United States.", 1),
    ("Barack Obama served as the 10th President of the United States.", 0),
    ("India gained independence from British rule in 1947.", 1),
    ("India gained independence from British rule in 1900.", 0),
    ("The Indian Parliament has two houses: Lok Sabha and Rajya Sabha.", 1),
    ("The Indian Parliament has only one house.", 0),
    ("Isaac Newton formulated the laws of motion.", 1),
    ("Isaac Newton formulated the theory of evolution.", 0),
    ("Charles Darwin proposed the theory of evolution by natural selection.", 1),
    ("Charles Darwin discovered penicillin.", 0),
    ("DNA carries genetic information in living organisms.", 1),
    ("DNA is a type of carbohydrate found only in plants.", 0),
    ("The speed of light is faster than the speed of sound.", 1),
    ("The speed of sound is faster than the speed of light.", 0),
    ("Jupiter is the largest planet in the solar system.", 1),
    ("Mercury is the largest planet in the solar system.", 0),
    ("The Amazon rainforest is primarily located in South America.", 1),
    ("The Amazon rainforest is primarily located in Australia.", 0),
    ("The Great Barrier Reef is located off the coast of Australia.", 1),
    ("The Great Barrier Reef is located off the coast of Norway.", 0),
    ("Shakespeare wrote the play Hamlet.", 1),
    ("Shakespeare wrote the Declaration of Independence.", 0),
] * 3

DOMAIN_KEYWORDS = {
    "Science": ["scientist", "discovery", "planet", "biology", "chemistry",
                "dna", "gold", "gravity", "newton", "einstein", "darwin",
                "evolution", "chemical", "light", "sound", "carbon"],
    "History": ["war", "king", "century", "ancient", "battle", "revolution",
                "independence", "empire", "napoleon", "wall fell", "1945",
                "1789", "1939"],
    "Geography": ["country", "capital", "river", "mountain", "continent",
                  "ocean", "border", "rainforest", "reef", "island"],
    "Sports": ["champion", "tournament", "player", "medal", "team",
               "olympic", "world cup", "football", "sprinter", "ballon"],
    "Politics": ["president", "election", "government", "parliament",
                 "minister", "united nations", "independence"],
}


def assign_domain(claim: str) -> str:
    claim_l = claim.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in claim_l for k in keywords):
            return domain
    return "Other"


def load_data() -> tuple[pd.DataFrame, str]:
    """Loads FEVER claims; falls back to a small offline set on any failure."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("copenlu/fever_gold_evidence", split="train[:500]")
        df = pd.DataFrame(dataset)
        df = df[df["label"].isin(["SUPPORTS", "REFUTES"])]
        df["label"] = df["label"].map({"SUPPORTS": 1, "REFUTES": 0})
        df = df[["claim", "label"]].dropna().reset_index(drop=True)
        if len(df) < 20:
            raise ValueError("Downloaded dataset too small, using fallback.")
        source = "FEVER (copenlu/fever_gold_evidence via HuggingFace)"
    except Exception as e:
        claims, labels = zip(*FALLBACK_CLAIMS)
        df = pd.DataFrame({"claim": claims, "label": labels}).reset_index(drop=True)
        source = f"Offline fallback set ({len(df)} claims) — FEVER unavailable: {type(e).__name__}"

    df["domain"] = df["claim"].apply(assign_domain)
    return df, source


# ---------------------------------------------------------------------------
# Neural classifier (the "in-house" confidence model)
# ---------------------------------------------------------------------------
class ConfidenceScorer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # logits — needed for temperature scaling


def train_model(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df["claim"]).toarray()
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y if len(set(y)) > 1 else None
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    model = ConfidenceScorer(X_train_t.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(150):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t).squeeze(-1), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t).squeeze(-1)
        test_probs = torch.sigmoid(test_logits).numpy()
        labels_pred = (test_probs > 0.5).astype(float)
        acc = (labels_pred == y_test).mean()

    return {
        "model": model,
        "vectorizer": vectorizer,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "test_probs": test_probs,
        "test_logits": test_logits.numpy(),
        "accuracy": acc,
    }


# ---------------------------------------------------------------------------
# Baseline: plain Logistic Regression, to justify (or challenge) the NN
# ---------------------------------------------------------------------------
def train_baseline(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = (preds == y_test).mean()
    return {"model": clf, "probs": probs, "accuracy": acc}


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Calibration metrics + temperature-scaling mitigation
# ---------------------------------------------------------------------------
def expected_calibration_error(probs, labels, n_bins: int = 10):
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs > 0.5).astype(int)
    correct = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        weight = mask.sum() / len(confidences)
        ece += weight * abs(bin_acc - bin_conf)
        bin_stats.append((bin_conf, bin_acc, int(mask.sum())))
    return ece, bin_stats


def fit_temperature(logits, labels) -> float:
    def nll(T):
        T = max(T, 1e-3)
        p = 1 / (1 + np.exp(-logits / T))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.05, 10), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Bias across domains
# ---------------------------------------------------------------------------
def bias_analysis(df: pd.DataFrame, model, vectorizer) -> dict:
    results = {}
    for domain in df["domain"].unique():
        subset = df[df["domain"] == domain]
        if len(subset) < 3:
            continue
        X = vectorizer.transform(subset["claim"]).toarray()
        X_t = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_t).squeeze(-1)).numpy()
        preds = (probs > 0.5).astype(int)
        true = subset["label"].values
        acc = (preds == true).mean()
        results[domain] = (round(acc * 100, 2), len(subset))
    return results


# ---------------------------------------------------------------------------
# Self-verification cascade — extends the paper's single-round finding to
# ask: does repeated self-verification cause confidence to drift further,
# or does it stabilize after one round?
# ---------------------------------------------------------------------------
def self_verification_cascade(model, vectorizer, claim: str, rounds: int = 5) -> list[float]:
    def score(text: str) -> float:
        feats = vectorizer.transform([text]).toarray()
        t = torch.tensor(feats, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(t)).item()

    scores = [score(claim)]
    current = claim
    for _ in range(rounds):
        current = f"Is this true? {current}"
        scores.append(score(current))
    return scores


# ---------------------------------------------------------------------------
# Selective prediction / abstention (risk-coverage curve) — tests whether
# confidence, even if miscalibrated in absolute terms, is still useful for
# deciding WHEN to trust the model (abstain below a threshold).
# ---------------------------------------------------------------------------
def selective_prediction_curve(probs: np.ndarray, labels: np.ndarray,
                                thresholds: np.ndarray | None = None) -> list[tuple]:
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.99, 15)

    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs > 0.5).astype(int)
    correct = (predictions == labels).astype(int)

    curve = []
    n = len(labels)
    for t in thresholds:
        mask = confidences >= t
        coverage = mask.sum() / n
        if mask.sum() == 0:
            curve.append((float(t), 0.0, None))
            continue
        acc_at_t = correct[mask].mean()
        curve.append((float(t), float(coverage), float(acc_at_t)))
    return curve


# ---------------------------------------------------------------------------
# Paraphrase consistency — same fact, different wording. Checks whether the
# model tracks MEANING or is sensitive to surface wording.
# ---------------------------------------------------------------------------
def paraphrase_consistency(model, vectorizer, paraphrases: list[str]) -> dict:
    def score(text: str) -> float:
        feats = vectorizer.transform([text]).toarray()
        t = torch.tensor(feats, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(t)).item()

    scores = [score(p) for p in paraphrases]
    labels = [s > 0.5 for s in scores]
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "range": float(max(scores) - min(scores)),
        "label_flip": len(set(labels)) > 1,  # did the verdict itself change across phrasings?
    }


# ---------------------------------------------------------------------------
# Hallucination error taxonomy — categorizes misclassified claims by a
# simple, transparent set of surface heuristics (not a claim of causal
# explanation, just a descriptive breakdown of what kinds of claims trip
# the model up).
# ---------------------------------------------------------------------------
NEGATION_WORDS = {"not", "never", "no", "none", "isn't", "wasn't", "doesn't", "didn't"}


def _has_negation(text: str) -> bool:
    tokens = set(text.lower().replace(".", "").split())
    return bool(tokens & NEGATION_WORDS)


def _has_number(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def error_taxonomy(df: pd.DataFrame, model, vectorizer) -> pd.DataFrame:
    X = vectorizer.transform(df["claim"]).toarray()
    X_t = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t).squeeze(-1)).numpy()
    preds = (probs > 0.5).astype(int)
    df = df.copy()
    df["predicted"] = preds
    df["correct"] = df["predicted"] == df["label"]

    errors = df[~df["correct"]].copy()
    if errors.empty:
        return pd.DataFrame(columns=["category", "count"])

    def categorize(claim: str) -> str:
        has_neg = _has_negation(claim)
        has_num = _has_number(claim)
        if has_neg and has_num:
            return "Negation + Numeric"
        if has_neg:
            return "Negation"
        if has_num:
            return "Numeric/Date"
        return "Other surface pattern"

    errors["category"] = errors["claim"].apply(categorize)
    counts = errors["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    return counts


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for accuracy — a single train/test split
# gives a noisy point estimate; this resamples the test set to report a
# range instead of a single misleadingly-precise number.
# ---------------------------------------------------------------------------
def bootstrap_accuracy_ci(y_true: np.ndarray, y_pred: np.ndarray,
                           n_bootstrap: int = 1000, ci: float = 0.95,
                           seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        accs.append((y_true[idx] == y_pred[idx]).mean())
    accs = np.array(accs)
    lower = np.percentile(accs, (1 - ci) / 2 * 100)
    upper = np.percentile(accs, (1 + ci) / 2 * 100)
    return float(accs.mean()), float(lower), float(upper)

PYEOF_FILE_MARKER

cat > "retrieval_utils.py" << 'PYEOF_FILE_MARKER'
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

PYEOF_FILE_MARKER

cat > "llm_client.py" << 'PYEOF_FILE_MARKER'
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

PYEOF_FILE_MARKER

cat > "requirements.txt" << 'PYEOF_FILE_MARKER'
streamlit>=1.30
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
scipy>=1.10
shap>=0.44
datasets>=2.14
requests>=2.31
pytest>=7.4

PYEOF_FILE_MARKER

cat > "README.md" << 'PYEOF_FILE_MARKER'
# 🔍 LLM Hallucination & Confidence Scorer

A diagnostic tool for studying confidence calibration and hallucination
detection, extending the empirical evaluation in *Empirical Multi-Phase
Evaluation of Hallucination and Confidence Consistency in LLMs* (ICIDSSD '26,
Scopus-indexed).

## What this actually is (scope, stated honestly)

There are two components, and the app is upfront about which is which:

- An **in-house TF-IDF + neural network classifier**, trained on FEVER
  claims, used as a controlled testbed for studying calibration and bias.
- An **optional live query to a real LLM** (via the free Groq API) so the
  app can compare a genuine LLM's self-reported confidence against the
  in-house model, rather than only measuring a small custom classifier and
  calling it "LLM hallucination detection."

If no API key is configured, the app runs in classifier-only mode and says
so in the UI — it never silently pretends to be testing an LLM when it isn't.

## Features

| Tab | What it shows |
|---|---|
| Claim Checker | In-house classifier score + self-verification shift; optional real-LLM verdict and self-reported confidence side by side |
| Metrics | Precision / Recall / F1 / Accuracy / Confusion Matrix, **Neural Net vs. a plain Logistic Regression baseline**, and a **bootstrap 95% confidence interval** for accuracy |
| Calibration | Expected Calibration Error (ECE) and a reliability diagram |
| Calibration Fix | Post-hoc **temperature scaling** (Guo et al., 2017), reporting honestly whether it helped or didn't |
| Bias Detection | Accuracy broken down by topical domain |
| Explainability | SHAP KernelExplainer — top tokens driving a given prediction |
| RAG Groundedness | Retrieves real Wikipedia evidence for a claim and compares a **closed-book LLM verdict vs. a RAG-grounded verdict** (answer using only the retrieved evidence) — tests whether retrieval actually changes/reduces hallucinated answers |
| Self-Verification Cascade | Extends the single-round self-verification finding: repeatedly re-verifies the claim and tracks how far confidence drifts over multiple rounds |
| Selective Prediction | Risk-coverage curve — checks whether confidence, even if miscalibrated in absolute terms, is still useful for deciding *when* to trust the model (abstaining below a threshold) |
| Robustness | Paraphrase consistency (same fact, different wording — does the verdict flip?) and a hallucination error taxonomy (negation / numeric-date / other surface patterns among misclassified claims) |

## Architecture

```
model_utils.py       # core logic: data, model, metrics, calibration, cascade,
                      # selective prediction, paraphrase check, error taxonomy — no Streamlit import
retrieval_utils.py    # Wikipedia RAG retrieval + lexical grounding score — no Streamlit import
llm_client.py         # optional Groq API wrapper (closed-book + RAG-grounded verdicts),
                       # degrades gracefully with no key
app.py                # Streamlit UI, imports the three modules above
tests/                 # pytest unit tests against model_utils and retrieval_utils
.github/workflows/tests.yml   # CI: runs the test suite on every push
```

Logic is deliberately separated from the UI so it can be unit-tested and
reused outside Streamlit (e.g. a notebook or CLI).

## Results (typical run — numbers vary slightly by data source used)

- Neural net test accuracy vs. Logistic Regression baseline are reported
  side-by-side in the Metrics tab, not just as a single accuracy number.
- ECE is reported before and after temperature scaling — the app shows
  whichever result actually occurs, including cases where scaling doesn't help.

## Limitations

- The in-house classifier is a small TF-IDF model, not a large language
  model — it's a controlled proxy for studying calibration behavior, not a
  claim to detect hallucination in GPT-scale models by itself.
- Domain bias analysis uses keyword-based domain tagging, which is noisy,
  especially for small subsets — treat domain breakdowns as directional.
- FEVER claims are derived from Wikipedia sentence rewrites; they don't
  fully represent open-domain claims an LLM might be asked about.
- The RAG groundedness check uses TF-IDF lexical overlap as a fast proxy for
  "is this grounded in the evidence" — it is not a full entailment/NLI model,
  so a high score means vocabulary overlap, not verified logical entailment.
- The self-verification cascade re-wraps the same claim through the in-house
  classifier only; it does not call a real LLM repeatedly (that would need
  an API budget), so it shows the in-house model's behavior, not necessarily
  a production LLM's.

## Setup

```bash
git clone <your-repo-url>
cd llm-confidence-scorer
pip install -r requirements.txt
```

### Optional: enable the real-LLM comparison
1. Get a free key at https://console.groq.com/keys
2. Set it as an environment variable before running:
   - macOS/Linux: `export GROQ_API_KEY=your_key_here`
   - Windows (PowerShell): `$env:GROQ_API_KEY="your_key_here"`
   - Or create a `.env` file locally (not committed — see `.gitignore`)

### Run locally
```bash
streamlit run app.py
```

### Run tests
```bash
pytest tests/ -v
```

## Deployment (real URL, not localhost)

Recommended: **Streamlit Community Cloud** (free, connects directly to GitHub).
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → "New app" → select this repo, branch
   `main`, file `app.py`.
3. If using the LLM comparison, add `GROQ_API_KEY` under
   **App settings → Secrets** (not in the repo).
4. Deploy — you'll get a permanent `https://<name>.streamlit.app` URL.

## Tech Stack
Python, PyTorch, Scikit-learn, SHAP, SciPy, Streamlit, Groq API, FEVER Dataset

## License
MIT — see `LICENSE`.

PYEOF_FILE_MARKER

cat > ".gitignore" << 'PYEOF_FILE_MARKER'
__pycache__/
*.pyc
.venv/
venv/
env/
.env
*.pt
.streamlit/secrets.toml
.pytest_cache/
.DS_Store
*.egg-info/

PYEOF_FILE_MARKER

cat > "LICENSE" << 'PYEOF_FILE_MARKER'
MIT License

Copyright (c) 2026 Shabnoor Fatima

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

PYEOF_FILE_MARKER

mkdir -p "tests"
cat > "tests/test_model_utils.py" << 'PYEOF_FILE_MARKER'
"""
tests/test_model_utils.py — run with: pytest
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model_utils as mu


def test_assign_domain_science():
    assert mu.assign_domain("Einstein developed the theory of relativity.") == "Science"


def test_assign_domain_geography():
    assert mu.assign_domain("The Nile is a river in Africa.") == "Geography"


def test_assign_domain_fallback_other():
    assert mu.assign_domain("Blorp is a fictional word with no keywords.") == "Other"


def test_load_data_returns_expected_columns():
    df, source = mu.load_data()
    assert {"claim", "label", "domain"}.issubset(df.columns)
    assert len(df) > 0
    assert set(df["label"].unique()).issubset({0, 1})
    assert isinstance(source, str)


def test_expected_calibration_error_perfect_calibration():
    # If confidence exactly matches accuracy in every bin, ECE should be ~0
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    ece, bins = mu.expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.15  # not exactly zero due to binning, but should be small


def test_expected_calibration_error_bounds():
    probs = np.random.uniform(0, 1, 50)
    labels = np.random.randint(0, 2, 50)
    ece, _ = mu.expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0


def test_fit_temperature_returns_positive_scalar():
    logits = np.random.randn(50) * 3
    labels = np.random.randint(0, 2, 50)
    T = mu.fit_temperature(logits, labels)
    assert T > 0


def test_train_model_and_baseline_run_end_to_end():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["test_probs"]) == len(result["y_test"])

    baseline = mu.train_baseline(
        result["X_train"], result["y_train"], result["X_test"], result["y_test"]
    )
    assert 0.0 <= baseline["accuracy"] <= 1.0


def test_classification_metrics_shape():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    metrics = mu.classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"precision", "recall", "f1", "confusion_matrix"}
    assert len(metrics["confusion_matrix"]) == 2


def test_bias_analysis_runs():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    bias = mu.bias_analysis(df, result["model"], result["vectorizer"])
    assert isinstance(bias, dict)
    for domain, (acc, count) in bias.items():
        assert 0.0 <= acc <= 100.0
        assert count > 0


def test_self_verification_cascade_length():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    scores = mu.self_verification_cascade(
        result["model"], result["vectorizer"], "The Eiffel Tower is in Paris.", rounds=4
    )
    assert len(scores) == 5  # original + 4 rounds
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_selective_prediction_curve_monotonic_coverage():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    curve = mu.selective_prediction_curve(result["test_probs"], result["y_test"])
    coverages = [c[1] for c in curve]
    # coverage should be non-increasing as threshold rises
    assert all(coverages[i] >= coverages[i + 1] - 1e-9 for i in range(len(coverages) - 1))


def test_paraphrase_consistency_keys():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    pc = mu.paraphrase_consistency(
        result["model"], result["vectorizer"],
        ["The Eiffel Tower is in Paris.", "Paris has the Eiffel Tower."]
    )
    assert set(pc.keys()) == {"scores", "mean", "std", "range", "label_flip"}
    assert len(pc["scores"]) == 2


def test_error_taxonomy_returns_dataframe():
    df, _ = mu.load_data()
    result = mu.train_model(df)
    taxonomy = mu.error_taxonomy(df, result["model"], result["vectorizer"])
    assert "category" in taxonomy.columns or taxonomy.empty
    assert "count" in taxonomy.columns or taxonomy.empty


def test_bootstrap_accuracy_ci_bounds():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])
    mean, lo, hi = mu.bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=200)
    assert 0.0 <= lo <= mean <= hi <= 1.0

PYEOF_FILE_MARKER

mkdir -p "tests"
cat > "tests/test_retrieval_utils.py" << 'PYEOF_FILE_MARKER'
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

PYEOF_FILE_MARKER

mkdir -p ".github/workflows"
cat > ".github/workflows/tests.yml" << 'PYEOF_FILE_MARKER'
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v

PYEOF_FILE_MARKER

echo "All files created successfully."
echo "Next: pip install -r requirements.txt"