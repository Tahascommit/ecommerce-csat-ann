# Local deployment of the CSAT models: a Streamlit web app.
# Run from the repo folder:  streamlit run app/app.py
import json
import os
import sys
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                     # csat_features.py lives in the repo folder
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras                                                 # noqa: E402
from csat_features import make_features, model_inputs        # noqa: E402

st.set_page_config(page_title="Shopzilla CSAT predictor", layout="wide")
SCORES = [1, 2, 3, 4, 5]
COLORS = ["#C44E52", "#DD8452", "#CCB974", "#8FBF6F", "#55A868"]


@st.cache_resource
def load_models():
    """Load both saved models and their settings once, when the app starts."""
    with open(os.path.join(ROOT, "models", "csat_metadata.json")) as fh:
        meta = json.load(fh)
    models = {k: keras.models.load_model(os.path.join(ROOT, "models", v["file"])) for k, v in meta["models"].items()}
    return meta, models


def predict(rows, which):
    """Raw Shopzilla rows -> probability of each score, using the same feature code as the notebook."""
    settings = META["models"][which]
    feat = make_features(rows)
    return MODELS[which].predict(model_inputs(feat, settings["num_cols"], settings["with_text"]), batch_size=4096, verbose=0)


def summarise(p, which):
    """Predicted score, expected score, chance of a low score and whether the alert fires."""
    low = p[:, 0] + p[:, 1]
    return pd.DataFrame({"predicted score": p.argmax(1) + 1, "expected score": (p * np.array(SCORES)).sum(1).round(2),
                         "P(low CSAT)": low.round(3), "alert": low >= META["models"][which]["alert_threshold"]})


def fmt(dt):
    return dt.strftime("%d/%m/%Y %H:%M")


try:
    META, MODELS = load_models()
except FileNotFoundError:
    st.error("No saved models found. Run the notebook first: it writes the models to the models/ folder.")
    st.stop()

OPT = META["options"]

# ---------- sidebar: what the models are ----------
with st.sidebar:
    st.header("Models")
    st.write("**Real time**: uses only what is known while the contact is handled (channel, issue, agent, "
             "response time, order and product). Works before the customer answers the survey.")
    st.write("**With remarks**: also reads the customer's written remark, so it can only run after the survey.")
    scores = pd.DataFrame(META["test_scores"]).T
    keep = [m for m in ["ANN real time tuned", "ANN with remarks tuned"] if m in scores.index]
    st.caption("Scores on the test set (rows never used for training)")
    st.dataframe(scores.loc[keep, ["accuracy", "macro F1", "low-CSAT ROC-AUC", "low-CSAT PR-AUC"]].round(3).T)
    st.caption(f"Alert thresholds: real time {META['models']['realtime']['alert_threshold']}, "
               f"with remarks {META['models']['with_remarks']['alert_threshold']}")

st.title("Shopzilla customer satisfaction (CSAT) predictor")
tab_one, tab_file, tab_insights = st.tabs(["Predict one contact", "Score a file", "Insights"])

# ---------- tab 1: one contact ----------
with tab_one:
    # the category sits outside the form, so the sub-category list follows it straight away
    category = st.selectbox("Issue category", sorted(META["subcategories"]),
                            index=sorted(META["subcategories"]).index("Returns"))
    subs = META["subcategories"][category]
    with st.form("contact"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("The issue")
            channel = st.selectbox("Channel", OPT["channel_name"], index=OPT["channel_name"].index("Inbound"))
            sub_category = st.selectbox("Sub-category", subs,
                                        index=subs.index("Reverse Pickup Enquiry") if "Reverse Pickup Enquiry" in subs else 0)
            reported_day = st.date_input("Issue reported on", date(2023, 8, 16))
            reported_time = st.time_input("at", time(11, 0))
            wait = st.number_input("Minutes until the first response", min_value=0, max_value=10000, value=10)
        with c2:
            st.subheader("The agent")
            agents = sorted(META["agents"])
            agent = st.selectbox("Agent (type to search)", agents)
            st.text(f"Supervisor: {META['agents'][agent]['Supervisor']}\nManager: {META['agents'][agent]['Manager']}")
            tenure = st.selectbox("Tenure bucket", OPT["Tenure Bucket"])
            shift = st.selectbox("Shift", OPT["Agent Shift"], index=OPT["Agent Shift"].index("Morning"))
        with c3:
            st.subheader("Order and product")
            has_order = st.checkbox("Linked to an order", value=True)
            product = st.selectbox("Product category", OPT["Product_category"],
                                   index=OPT["Product_category"].index("Unknown"))
            city = st.selectbox("Customer city", ["UNKNOWN"] + [c for c in META["cities"] if c != "UNKNOWN"])
            price = st.number_input("Item price (0 = not known)", min_value=0.0, value=0.0, step=100.0)
            order_days = st.number_input("Days since the order (0 = not known)", min_value=0, max_value=600, value=0)
        remark = st.text_area("Customer remark from the survey (optional: leave empty for a real-time prediction)")
        submitted = st.form_submit_button("Predict CSAT")

    if submitted:
        reported = datetime.combine(reported_day, reported_time)
        row = {"Unique id": "app", "channel_name": channel, "category": category, "Sub-category": sub_category,
               "Customer Remarks": remark.strip() or np.nan, "Order_id": "app-order" if has_order else np.nan,
               "order_date_time": fmt(reported - timedelta(days=order_days)) if order_days > 0 else np.nan,
               "Issue_reported at": fmt(reported), "issue_responded": fmt(reported + timedelta(minutes=int(wait))),
               "Survey_response_Date": reported.strftime("%d-%b-%y"),
               "Customer_City": np.nan if city == "UNKNOWN" else city,
               "Product_category": np.nan if product == "Unknown" else product,
               "Item_price": price if price > 0 else np.nan, "connected_handling_time": np.nan,
               "Agent_name": agent, "Supervisor": META["agents"][agent]["Supervisor"],
               "Manager": META["agents"][agent]["Manager"], "Tenure Bucket": tenure, "Agent Shift": shift}
        rows = pd.DataFrame([row])
        which = "with_remarks" if remark.strip() else "realtime"
        p = predict(rows, which)
        res = summarise(p, which).iloc[0]

        st.markdown(f"**Model used:** {'with remarks' if which == 'with_remarks' else 'real time (no remark given)'}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Most likely score", int(res["predicted score"]))
        m2.metric("Expected score", f"{res['expected score']:.2f}")
        m3.metric("Chance of a low score (1-2)", f"{res['P(low CSAT)']:.0%}")
        if res["alert"]:
            st.error("Alert: this customer is likely to be unhappy. Suggest a supervisor call-back.")
        else:
            st.success("No alert: this customer is likely to be satisfied.")
        st.bar_chart(pd.DataFrame({"probability": p[0]}, index=[f"CSAT {s}" for s in SCORES]))

        # what-if: the same contact answered within 5 minutes
        if wait > 5:
            fast = rows.copy()
            fast["issue_responded"] = fmt(reported + timedelta(minutes=5))
            p_fast = predict(fast, which)
            st.info(f"What-if: answered within 5 minutes, the chance of a low score would be "
                    f"{p_fast[0, :2].sum():.0%} instead of {res['P(low CSAT)']:.0%}.")

# ---------- tab 2: a file of contacts ----------
with tab_file:
    st.write("Upload contacts in the same format as the Shopzilla export (the CSAT Score column is optional). "
             "Every row gets the real-time prediction, and rows with a remark also get the with-remarks prediction.")
    upload = st.file_uploader("CSV file", type="csv")
    sample_path = os.path.join(ROOT, "outputs", "sample_new_contacts.csv")
    use_sample = st.button("Use the sample file (200 contacts the models never saw)")
    rows = None
    if upload is not None:
        rows = pd.read_csv(upload)
    elif use_sample and os.path.exists(sample_path):
        rows = pd.read_csv(sample_path)
    if rows is not None:
        missing_cols = [c for c in ["channel_name", "category", "Sub-category", "Issue_reported at", "issue_responded",
                                    "Agent_name", "Supervisor", "Manager", "Tenure Bucket", "Agent Shift"] if c not in rows]
        if missing_cols:
            st.error(f"These columns are missing: {missing_cols}")
            st.stop()
        for c in ["Customer Remarks", "Order_id", "order_date_time", "Survey_response_Date", "Customer_City",
                  "Product_category", "Item_price", "connected_handling_time", "Unique id"]:
            if c not in rows:
                rows[c] = np.nan
        rows["Survey_response_Date"] = rows["Survey_response_Date"].fillna(
            pd.to_datetime(rows["Issue_reported at"], format="%d/%m/%Y %H:%M", errors="coerce").dt.strftime("%d-%b-%y"))
        rt = summarise(predict(rows, "realtime"), "realtime").add_prefix("real time: ")
        out = pd.concat([rows[["Unique id", "channel_name", "category", "Sub-category", "Agent_name"]].reset_index(drop=True), rt], axis=1)
        has_remark = rows["Customer Remarks"].notna().to_numpy()
        if has_remark.any():
            rm = summarise(predict(rows[has_remark], "with_remarks"), "with_remarks").add_prefix("with remarks: ")
            rm.index = np.where(has_remark)[0]
            out = out.join(rm)
        c1, c2, c3 = st.columns(3)
        c1.metric("Contacts", f"{len(out):,}")
        c2.metric("Real-time alerts", f"{out['real time: alert'].mean():.1%}")
        c3.metric("Average chance of a low score", f"{out['real time: P(low CSAT)'].mean():.1%}")
        if "CSAT Score" in rows:
            real_low = (rows["CSAT Score"] <= 2).to_numpy()
            caught = out.loc[real_low, "real time: alert"].mean() if real_low.any() else float("nan")
            st.caption(f"The file has real scores: {real_low.mean():.1%} are low, and the real-time alert catches {caught:.0%} of them.")
        st.dataframe(out.sort_values("real time: P(low CSAT)", ascending=False), width="stretch")
        st.download_button("Download the predictions", out.to_csv(index=False), "csat_predictions.csv", "text/csv")

# ---------- tab 3: insights ----------
with tab_insights:
    st.write("What the analysis and the models say about customer satisfaction at Shopzilla.")
    fig_dir = os.path.join(ROOT, "outputs", "figures")
    for fname, caption in [("chart06_low_csat_by_response_time.png", "Unhappy customers by response time"),
                           ("what_if_response_time.png", "What-if: predicted share of unhappy customers by response time"),
                           ("insight_subcategory_risk.png", "Sub-categories: predicted vs real share of low CSAT"),
                           ("permutation_importance.png", "Which inputs the models rely on")]:
        path = os.path.join(fig_dir, fname)
        if os.path.exists(path):
            st.image(path, caption=caption)
    for fname, title in [("risk_by_subcategory.csv", "Riskiest sub-categories"), ("risk_by_supervisor.csv", "Supervisors by predicted risk")]:
        path = os.path.join(ROOT, "outputs", fname)
        if os.path.exists(path):
            st.subheader(title)
            st.dataframe(pd.read_csv(path).head(10).round(3), width="stretch")
