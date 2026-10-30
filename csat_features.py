# Feature code shared by the notebook and the Streamlit app.
# The notebook writes this file, so training and deployment prepare the data in exactly the same way.
import re

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

# model inputs
CAT_COLS = ["channel_name", "category", "Sub-category", "Customer_City", "Product_category", "Agent_name",
            "Supervisor", "Manager", "Tenure Bucket", "Agent Shift", "Reported_Weekday"]
NUM_REALTIME = ["Response_Log_Min", "Response_Unknown", "Hour_Sin", "Hour_Cos", "Has_Order_Id",
                "Order_Age_Log_Days", "Has_Price", "Price_Log"]
NUM_SURVEY = NUM_REALTIME + ["Survey_Lag_Days", "Remark_Log_Words"]   # only known after the survey
TEXT_COL = "Remark_Clean"
# Keras input names can't have spaces, so every column gets a simple name: "Tenure Bucket" -> "tenure_bucket"
INPUT_NAMES = {c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in CAT_COLS}

# ---------- text cleaning, one function per step ----------
CONTRACTIONS = {r"won't": "will not", r"can't": "cannot", r"n't": " not", r"'re": " are", r"'s": " is",
                r"'d": " would", r"'ll": " will", r"'ve": " have", r"'m": " am"}
# a short stop list of function words only: the standard lists also drop "not", "never", "keep", "well" and "done",
# which carry the customer's opinion in remarks like "not resolved" or "keep it up"
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am", "to", "of", "for", "in", "on",
              "at", "by", "with", "and", "or", "as", "from", "about", "into", "this", "that", "these", "those", "it",
              "its", "i", "me", "my", "we", "our", "you", "your", "u", "ur", "he", "him", "his", "she", "her", "they",
              "them", "their", "do", "does", "did", "have", "has", "had", "will", "would", "shall", "should", "can",
              "could", "may", "might", "so", "there", "here", "also", "just", "sir", "mam", "maam", "madam", "dear"}
SHORT_FORMS = {"pls": "please", "plz": "please", "thx": "thanks", "thnx": "thanks", "thanku": "thank you",
               "thankyou": "thank you", "tq": "thank you", "ty": "thank you", "gud": "good", "gd": "good",
               "nt": "not", "nhi": "not", "nahi": "not", "ok": "okay", "k": "okay", "nyc": "nice", "issu": "issue",
               "slove": "solve", "solv": "solve", "excelent": "excellent", "servic": "service", "custmer": "customer"}
STEMMER = PorterStemmer()


def expand_contractions(t):
    for short, full in CONTRACTIONS.items():
        t = re.sub(short, full, t, flags=re.IGNORECASE)
    return t


def lower_case(t):
    return t.lower()


def remove_punctuation(t):
    return re.sub(r"[^\w\s]|_", " ", t)                        # also removes the '??' left where emojis were


def remove_urls_and_digit_words(t):
    t = re.sub(r"\b(?:https?|www|com)\b", " ", t)               # pieces of a link left after removing punctuation
    return re.sub(r"\b\w*\d\w*\b", " ", t)


def remove_stopwords(t):
    return " ".join(w for w in t.split() if w not in STOP_WORDS)   # split/join also removes extra white space


def rephrase(t):
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)                       # goooood -> good, sooo -> soo
    return " ".join(SHORT_FORMS.get(w, w) for w in t.split())


def tokenize(t):
    return re.findall(r"[a-z]+", t)


def stem(tokens):
    return [STEMMER.stem(w) for w in tokens if len(w) > 1]


def clean_remark(text):
    """Customer remark -> cleaned, stemmed tokens joined by spaces ('' when there is no remark)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = remove_stopwords(remove_urls_and_digit_words(remove_punctuation(lower_case(expand_contractions(text)))))
    return " ".join(stem(tokenize(rephrase(t))))


# ---------- features ----------
def parse_dates(df):
    """Parse the four date columns of the raw file (day first)."""
    out = pd.DataFrame(index=df.index)
    out["Reported"] = pd.to_datetime(df["Issue_reported at"], format="%d/%m/%Y %H:%M", errors="coerce")
    out["Responded"] = pd.to_datetime(df["issue_responded"], format="%d/%m/%Y %H:%M", errors="coerce")
    out["Survey"] = pd.to_datetime(df["Survey_response_Date"], format="%d-%b-%y", errors="coerce")
    out["Ordered"] = pd.to_datetime(df["order_date_time"], format="%d/%m/%Y %H:%M", errors="coerce")
    return out


def make_features(df):
    """Raw Shopzilla rows -> model inputs (categorical strings, numbers and the cleaned remark)."""
    d = parse_dates(df)
    f = pd.concat([df[c].fillna("Unknown").astype(str).str.strip().rename(c) for c in CAT_COLS[:-1]], axis=1)
    f["Customer_City"] = f["Customer_City"].str.upper()
    f["Reported_Weekday"] = d["Reported"].dt.day_name().fillna("Unknown")
    f = pd.concat([f, d], axis=1)

    # response time: on 28 Aug 2023 the reply was saved at 00:00 without a time, so those are unknown
    minutes = (d["Responded"] - d["Reported"]).dt.total_seconds() / 60
    no_time = (d["Responded"].dt.hour == 0) & (d["Responded"].dt.minute == 0)
    unknown = no_time | minutes.isna() | (minutes < 0)
    f["Response_Min"] = minutes.where(~unknown)
    f["Response_Unknown"] = unknown.astype(int)
    f["Response_Log_Min"] = np.log1p(f["Response_Min"].fillna(0))

    hour = d["Reported"].dt.hour.fillna(12)
    f["Reported_Hour"] = hour.astype(int)
    f["Hour_Sin"] = np.sin(2 * np.pi * hour / 24)               # hour as a circle: 23:00 is next to 00:00
    f["Hour_Cos"] = np.cos(2 * np.pi * hour / 24)

    f["Has_Order_Id"] = df["Order_id"].notna().astype(int)
    f["Has_Order_Date"] = d["Ordered"].notna().astype(int)
    age = (d["Reported"] - d["Ordered"]).dt.total_seconds() / 86400
    f["Order_Age_Days"] = age.where(age >= 0)
    f["Order_Age_Log_Days"] = np.log1p(f["Order_Age_Days"].fillna(0))
    price = pd.to_numeric(df["Item_price"], errors="coerce")
    f["Item_Price"] = price
    f["Has_Price"] = price.notna().astype(int)
    f["Price_Log"] = np.log1p(price.clip(lower=0).fillna(0))

    # known only after the customer answers the survey
    f["Survey_Lag_Days"] = (d["Survey"] - d["Reported"].dt.normalize()).dt.days.clip(lower=0).fillna(0)
    remark = df["Customer Remarks"]
    f["Has_Remark"] = (remark.notna() & remark.astype(str).str.strip().ne("")).astype(int)
    f["Remark_Words"] = remark.fillna("").astype(str).str.split().str.len()
    f["Remark_Log_Words"] = np.log1p(f["Remark_Words"])
    f[TEXT_COL] = remark.map(clean_remark)
    return f


def to_tensors(x):
    """Keras takes text inputs only as TensorFlow string tensors, so the text arrays are converted."""
    import tensorflow as tf
    return {k: tf.constant(v.astype(str)) if v.dtype == object else v for k, v in x.items()}


def model_inputs(f, num_cols, with_text, tensors=True):
    """Feature table -> the dict of inputs the Keras model takes (one entry per input layer)."""
    x = {INPUT_NAMES[c]: f[c].astype(str).to_numpy(dtype=object).reshape(-1, 1) for c in CAT_COLS}
    x["numeric"] = f[num_cols].astype("float32").to_numpy()
    if with_text:
        x["remark"] = f[TEXT_COL].astype(str).to_numpy(dtype=object)          # one string per row
    return to_tensors(x) if tensors else x
