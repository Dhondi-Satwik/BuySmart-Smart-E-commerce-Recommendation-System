import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data.csv")

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df

def build_user_item_matrix(df):
    return df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

def recommend_for_user(user_id, user_item_matrix, top_n=5):
    similarity = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(similarity,
                          index=user_item_matrix.index,
                          columns=user_item_matrix.index)

    similar_users = sim_df[user_id].sort_values(ascending=False).iloc[1:6].index
    recs = user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)

    already_bought = user_item_matrix.loc[user_id]
    recs = recs[already_bought == 0]

    return recs.head(top_n)

from datetime import datetime

# -------- FESTIVAL DATA (PHASE 2) --------
festivals = [
    {"name": "Diwali", "date": "2025-10-20", "discount": "50–80%"},
    {"name": "Christmas", "date": "2025-12-25", "discount": "40–70%"},
    {"name": "New Year", "date": "2026-01-01", "discount": "30–60%"},
]

def get_next_festival():
    today = datetime.today().date()
    upcoming = []

    for f in festivals:
        f_date = datetime.strptime(f["date"], "%Y-%m-%d").date()
        if f_date >= today:
            upcoming.append((f, (f_date - today).days))

    if upcoming:
        return min(upcoming, key=lambda x: x[1])
    return None


# ---------------- UI ----------------
st.title("FestiveCart – Smart Product Recommender")

df = load_data()
user_item_matrix = build_user_item_matrix(df)

user_id = st.selectbox("Select User ID", user_item_matrix.index)

if st.button("Get Recommendations"):
    recs = recommend_for_user(user_id, user_item_matrix)
    st.subheader("Recommended Products")
    st.write(recs)
    
    # -------- FESTIVAL ADVICE (PHASE 2 UI) --------
festival_info = get_next_festival()

if festival_info:
    fest, days = festival_info

    if days <= 30:
        st.warning(
            f"🎯 Upcoming {fest['name']} Sale in {days} days "
            f"(Expected Discount: {fest['discount']}). Consider waiting!"
        )
    else:
        st.success("No major festival sale soon. Buying now is reasonable.")

