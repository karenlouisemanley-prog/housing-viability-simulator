import streamlit as st
import pandas as pd
import numpy as np
import hashlib

st.set_page_config(page_title="CHP Housing Viability Simulator", layout="wide")

st.title("🏡 CHP Housing Viability Simulator")
st.write("Upload housing portfolio data or use the sample dataset to explore long-term viability under stress tests.")

# --- Example dataset ---
def load_sample_data():
    data = {
        "PropertyID": [f"CHP-{i}" for i in range(1, 11)],
        "EPC": np.random.choice(["A","B","C","D","E"], 10),
        "CurrentRent": np.random.randint(80, 150, 10),
        "ArrearsPct": np.random.uniform(0.5, 7.0, 10).round(2),
        "CapexNeeded": np.random.randint(5000, 25000, 10),
        "EnergyCosts": np.random.randint(600, 2000, 10),
    }
    return pd.DataFrame(data)

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your CSV portfolio (PropertyID, EPC, CurrentRent, ArrearsPct, CapexNeeded, EnergyCosts)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
else:
    df = load_sample_data()
    st.info("Using synthetic CHP sample dataset")

st.subheader("📊 Portfolio Snapshot")
st.dataframe(df, use_container_width=True)

# --- Viability scoring ---
def score_property(row):
    score = 100
    if row["EPC"] in ["D", "E"]:
        score -= 20
    if row["ArrearsPct"] > 5:
        score -= 15
    if row["CapexNeeded"] > 20000:
        score -= 20
    if row["EnergyCosts"] > 1500:
        score -= 10
    return max(score, 0)

df["ViabilityScore"] = df.apply(score_property, axis=1)

# --- Monte Carlo Stress Test ---
def monte_carlo(df, n=500):
    outcomes = []
    for _ in range(n):
        noise = np.random.normal(1, 0.05, size=len(df))
        scenario = df["ViabilityScore"] * noise
        outcomes.append(scenario)
    return np.array(outcomes)

simulations = monte_carlo(df, n=300)

p10 = np.percentile(simulations, 10, axis=0)
p50 = np.percentile(simulations, 50, axis=0)
p90 = np.percentile(simulations, 90, axis=0)

df["p10"] = p10.round(1)
df["p50"] = p50.round(1)
df["p90"] = p90.round(1)

# --- RAG classification ---
def rag_class(score):
    if score < 40:
        return "🔴 High Risk"
    elif score < 70:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

df["RAG"] = df["p50"].apply(rag_class)

# --- Display results ---
st.subheader("🏠 Stress-Test Results")
st.dataframe(df, use_container_width=True)

st.subheader("🔎 Portfolio Risk Summary")
st.bar_chart(df["RAG"].value_counts())

# --- Top 10 at-risk ---
st.subheader("⚠️ Top 10 At-Risk Properties")
st.table(df.sort_values("p10").head(10))

# --- Dataset integrity ---
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
st.caption(f"Dataset hash: {data_hash[:16]}...")

# --- Download ---
st.download_button(
    label="📥 Download Results CSV",
    data=df.to_csv(index=False),
    file_name="viability_results.csv",
    mime="text/csv"
)
* **Downloadable results**.

👉 Next step: copy this into `app.py`, push to GitHub with the `requirements.txt`, then deploy to Streamlit Cloud. Would you like me to also give you the exact `requirements.txt` content here so you can paste it straight in?
