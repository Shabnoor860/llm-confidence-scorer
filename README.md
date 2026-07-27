# 🔍 LLM Hallucination & Confidence Scorer

A diagnostic tool for studying confidence calibration and hallucination detection in text classifiers and LLMs — built to answer a specific question: **does a model's confidence actually track whether it's right?**

## What this actually is (scope, stated honestly)

There are two components, and the app is upfront about which is which:

- An **in-house TF-IDF + neural network classifier**, trained on FEVER claims, used as a controlled testbed for studying calibration and bias.
- An **optional live query to a real LLM** (via the free Groq API) so the app can compare a genuine LLM's self-reported confidence against the in-house model, rather than only measuring a small custom classifier and calling it "LLM hallucination detection."

If no API key is configured, the app runs in classifier-only mode and says so in the UI — it never silently pretends to be testing an LLM when it isn't.

## Motivation

Confidence scores are everywhere in ML systems, but they're rarely checked against reality. A model can be 95% "confident" and still wrong just as often as a model that says 60%. This project builds a small, fully-inspectable pipeline to actually measure that gap — and then tests whether a standard fix (temperature scaling) closes it, rather than just reporting the problem and stopping there.

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
| Self-Verification Cascade | Repeatedly re-verifies a claim through the classifier and tracks how far confidence drifts over multiple rounds |
| Selective Prediction | Risk-coverage curve — checks whether confidence, even if miscalibrated in absolute terms, is still useful for deciding *when* to trust the model (abstaining below a threshold) |
| Robustness | Paraphrase consistency (same fact, different wording — does the verdict flip?) and a hallucination error taxonomy (negation / numeric-date / other surface patterns among misclassified claims) |

## Architecture
model_utils.py # core logic: data, model, metrics, calibration, cascade,
# selective prediction, paraphrase check, error taxonomy — no Streamlit import
retrieval_utils.py # Wikipedia RAG retrieval + lexical grounding score — no Streamlit import
llm_client.py # optional Groq API wrapper (closed-book + RAG-grounded verdicts),
# degrades gracefully with no key
app.py # Streamlit UI, imports the three modules above
tests/ # pytest unit tests against model_utils and retrieval_utils
.github/workflows/tests.yml # CI: runs the test suite on every push
Logic is deliberately separated from the UI so it can be unit-tested and reused outside Streamlit (e.g. a notebook or CLI).

## Results (typical run — numbers vary slightly by data source used)

- Neural net test accuracy vs. Logistic Regression baseline are reported side-by-side in the Metrics tab, not just as a single accuracy number.
- ECE is reported before and after temperature scaling — the app shows whichever result actually occurs, including cases where scaling doesn't help.
- Confidence often shifts after a self-verification prompt, but not always in the direction that improves accuracy — the app measures this directly instead of assuming it.

## Limitations

- The in-house classifier is a small TF-IDF model, not a large language model — it's a controlled proxy for studying calibration behavior, not a claim to detect hallucination in GPT-scale models by itself.
- Domain bias analysis uses keyword-based domain tagging, which is noisy, especially for small subsets — treat domain breakdowns as directional.
- FEVER claims are derived from Wikipedia sentence rewrites; they don't fully represent open-domain claims an LLM might be asked about.
- The RAG groundedness check uses TF-IDF lexical overlap as a fast proxy for "is this grounded in the evidence" — it is not a full entailment/NLI model, so a high score means vocabulary overlap, not verified logical entailment.
- The self-verification cascade re-wraps the same claim through the in-house classifier only; it does not call a real LLM repeatedly (that would need an API budget), so it shows the in-house model's behavior, not necessarily a production LLM's.

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
2. Go to https://share.streamlit.io → "New app" → select this repo, branch `main`, file `app.py`.
3. If using the LLM comparison, add `GROQ_API_KEY` under **App settings → Secrets** (not in the repo).
4. Deploy — you'll get a permanent `https://<name>.streamlit.app` URL.

## Tech Stack
Python, PyTorch, Scikit-learn, SHAP, SciPy, Streamlit, Groq API, FEVER Dataset

## License
MIT — see `LICENSE`.
