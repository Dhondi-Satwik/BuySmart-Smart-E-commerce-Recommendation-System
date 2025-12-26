import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PATH SETUP (ROBUST – NO PATH ERRORS)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data.csv")

# -------------------------------------------------
# DATA LOADING & CLEANING
# -------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df

# -------------------------------------------------
# POPULARITY-BASED RECOMMENDER
# -------------------------------------------------
def get_popular_products(df, top_n=10):
    popular = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return popular

# -------------------------------------------------
# COLLABORATIVE FILTERING (USER–USER)
# -------------------------------------------------
def build_user_item_matrix(df):
    user_item = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )
    return user_item

def recommend_for_user(user_id, user_item_matrix, top_n=5):
    if user_id not in user_item_matrix.index:
        return "User not found"

    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    similar_users = (
        similarity_df[user_id]
        .sort_values(ascending=False)
        .iloc[1:6]  # top 5 similar users
        .index
    )

    recommendations = (
        user_item_matrix.loc[similar_users]
        .sum()
        .sort_values(ascending=False)
    )

    already_bought = user_item_matrix.loc[user_id]
    recommendations = recommendations[already_bought == 0]

    return recommendations.head(top_n)

# -------------------------------------------------
# MAIN EXECUTION (TEST EVERYTHING)
# -------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print("Data loaded:", df.shape)

    print("\n🔥 TOP 10 TRENDING PRODUCTS 🔥")
    popular_products = get_popular_products(df, top_n=10)
    print(popular_products)

    print("\nBuilding user-item matrix...")
    user_item_matrix = build_user_item_matrix(df)

    sample_user = user_item_matrix.index[0]
    print(f"\n🎯 RECOMMENDATIONS FOR USER {sample_user}")
    print(recommend_for_user(sample_user, user_item_matrix, top_n=5))
