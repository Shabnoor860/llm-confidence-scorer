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

