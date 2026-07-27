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

