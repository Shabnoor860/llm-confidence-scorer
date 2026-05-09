import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import shap

# --- Load FEVER Dataset ---
@st.cache_data
def load_data():
    dataset = load_dataset("copenlu/fever_gold_evidence", split="train[:500]")
    df = pd.DataFrame(dataset)
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])]
    df["label"] = df["label"].map({"SUPPORTS": 1, "REFUTES": 0})
    df = df[["claim", "label"]].dropna()
    return df

# --- Feature Extraction ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df["claim"]).toarray()
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    # --- Model ---
    class ConfidenceScorer(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1     = nn.Linear(input_dim, 128)
            self.relu    = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2     = nn.Linear(128, 64)
            self.fc3     = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            return self.sigmoid(self.fc3(x))

    model = ConfidenceScorer(X_train_t.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t).squeeze(), y_train_t)
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(model.state_dict(), "confidence_model.pt")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).squeeze().detach()
        labels = (preds > 0.5).float()
        acc = (labels == y_test_t).float().mean().item()

    return model, vectorizer, X_test, y_test, preds.numpy().flatten(), acc

# --- Bias Detection Across Domains ---
def bias_analysis(df, model, vectorizer):
    domains = {
        "Science": ["scientist", "discovery", "planet", "biology", "chemistry"],
        "History": ["war", "king", "century", "ancient", "battle"],
        "Geography": ["country", "capital", "river", "mountain", "continent"],
        "Sports": ["champion", "tournament", "player", "medal", "team"],
        "Politics": ["president", "election", "government", "parliament", "minister"]
    }

    results = {}
    for domain, keywords in domains.items():
        mask = df["claim"].str.contains("|".join(keywords), case=False)
        subset = df[mask]
        if len(subset) == 0:
            continue
        X = vectorizer.transform(subset["claim"]).toarray()
        X_t = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            scores = model(X_t).squeeze().numpy()
        preds = (scores > 0.5).astype(int)
        true = subset["label"].values
        if len(true) > 0:
            acc = (preds == true).mean()
            results[domain] = round(acc * 100, 2)

    return results

# --- Streamlit UI ---
st.set_page_config(page_title="LLM Confidence Scorer", page_icon="🔍", layout="wide")
st.title("🔍 LLM Hallucination & Confidence Scorer")
st.caption("tool")

with st.spinner("Loading FEVER dataset and training model..."):
    df = load_data()
    model, vectorizer, X_test, y_test, preds, acc = train_model(df)

st.success(f"Model trained on {len(df)} claims | Test Accuracy: {acc*100:.2f}%")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Claim Checker", "Confidence Analysis", "Bias Detection", "Explainability"
])

# Tab 1 - Claim Checker
with tab1:
    st.subheader("Enter a claim to check")
    claim = st.text_input("Claim", placeholder="e.g. The Eiffel Tower is in Paris.")
    
    if claim:
        features = vectorizer.transform([claim]).toarray()
        tensor = torch.tensor(features, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            score = model(tensor).item()
        label = "Reliable" if score > 0.5 else "Likely Hallucinated"
        color = "green" if score > 0.5 else "red"
        st.markdown(f"**Confidence Score:** `{score:.4f}`")
        st.markdown(f"**Prediction:** :{color}[{label}]")

        # Self-verification test
        st.markdown("---")
        st.markdown("**Self-Verification Test** (Paper Finding)")
        verify_prompt = f"Is this true? {claim}"
        features2 = vectorizer.transform([verify_prompt]).toarray()
        tensor2 = torch.tensor(features2, dtype=torch.float32)
        with torch.no_grad():
            score2 = model(tensor2).item()
        label2 = "Reliable" if score2 > 0.5 else "Likely Hallucinated"
        st.markdown(f"After self-verification: `{score2:.4f}` → **{label2}**")
        if abs(score - score2) > 0.05:
            st.warning("Confidence shifted after self-verification")

# Tab 2 - Confidence Histogram
with tab2:
    st.subheader("Confidence Score Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(preds[y_test == 1], bins=20, alpha=0.6, label="Reliable (SUPPORTS)", color="green")
    ax.hist(preds[y_test == 0], bins=20, alpha=0.6, label="Hallucinated (REFUTES)", color="red")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Confidence Score Distribution — Reliable vs Hallucinated")
    ax.legend()
    st.pyplot(fig)

    # Calibration
    st.subheader("Confidence Calibration")
    bins = np.linspace(0, 1, 11)
    bin_acc = []
    bin_conf = []
    for i in range(len(bins)-1):
        mask = (preds >= bins[i]) & (preds < bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(y_test[mask].mean())
            bin_conf.append(preds[mask].mean())
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([0,1], [0,1], "k--", label="Perfect calibration")
    ax2.plot(bin_conf, bin_acc, "bo-", label="Model calibration")
    ax2.set_xlabel("Mean Confidence")
    ax2.set_ylabel("Actual Accuracy")
    ax2.set_title("Calibration Curve")
    ax2.legend()
    st.pyplot(fig2)

# Tab 3 - Bias Detection
with tab3:
    st.subheader("Accuracy Across Domains (Bias Analysis)")
    bias = bias_analysis(df, model, vectorizer)
    if bias:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.bar(bias.keys(), bias.values(), color="steelblue")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Model Accuracy by Domain — Ethical AI Analysis")
        ax3.set_ylim(0, 100)
        st.pyplot(fig3)
        st.caption("Unequal accuracy across domains indicates potential bias in model reliability.")

# Tab 4 - Explainability
with tab4:
    st.subheader("Why did the model make this prediction?")
    st.caption("SHAP explainability — which words influenced the score most")
    sample_claim = st.text_input("Enter claim for explanation", 
                                  placeholder="e.g. Einstein won the Nobel Prize.")
    if sample_claim:
        X_bg = vectorizer.transform(df["claim"][:50]).toarray()
        X_sample = vectorizer.transform([sample_claim]).toarray()

        def model_predict(x):
            t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return model(t).numpy()

        explainer = shap.KernelExplainer(model_predict, X_bg)
        shap_vals = explainer.shap_values(X_sample, nsamples=50)

        feature_names = vectorizer.get_feature_names_out()
        shap_df = pd.DataFrame({
            "word": feature_names,
            "shap": np.array(shap_vals[0]).flatten()
        }).sort_values("shap", key=abs, ascending=False).head(10)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        colors = ["green" if v > 0 else "red" for v in shap_df["shap"]]
        ax4.barh(shap_df["word"], shap_df["shap"], color=colors)
        ax4.set_title("Top Words Influencing Prediction")
        ax4.set_xlabel("SHAP Value")
        st.pyplot(fig4)