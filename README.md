LLM Hallucination & Confidence Scorer

A diagnostic tool for studying confidence calibration and hallucination detection
in text classifiers

## What it does
- **Claim Checker** — scores a claim as reliable / likely hallucinated, and runs
  a self-verification test to reproduce the paper's finding that self-verification
  prompting shifts (and can degrade) confidence.
- **Calibration** — measures Expected Calibration Error (ECE) and plots a
  reliability diagram, quantifying *how* miscalibrated the confidence scores are,
  not just that they're unreliable.
- **Calibration Fix** — applies post-hoc temperature scaling (Guo et al., 2017)
  and reports whether it actually reduces ECE, testing a standard mitigation
  rather than only diagnosing the problem.
- **Bias Detection** — compares accuracy across topical domains (Science,
  History, Geography, Sports, Politics).
- **Explainability** — SHAP KernelExplainer highlights which tokens drove a
  given prediction.

## Robustness
- Falls back to a small offline-labeled claim set if the FEVER dataset can't be
  reached, so the app never crashes on a network hiccup — it just reports which
  data source it used.
- SHAP computation runs on demand (button-triggered) instead of on every
  keystroke, and its output shape is handled defensively across SHAP versions.
- `requirements.txt` is a minimal, deploy-safe dependency list rather than a
  full environment dump.

## Tech Stack
Python, PyTorch, Streamlit, SHAP, Scikit-learn, SciPy, FEVER Dataset

## Key findings
- Self-verification prompting shifts confidence scores rather than reliably
  correcting them.
- Raw model confidence does not track actual correctness (non-zero ECE).
- Whether temperature scaling closes that gap is tested empirically in-app,
  not assumed.
