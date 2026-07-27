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