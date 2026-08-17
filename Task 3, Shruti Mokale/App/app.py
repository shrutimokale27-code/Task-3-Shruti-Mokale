import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Product Recommendation System",
    page_icon="AI",
    layout="centered"
)


# =========================================================
# LOAD DATASET
# =========================================================

file_path = "dataset/Dataset for Data Analytics.xlsx.xlsx"

df = pd.read_excel(file_path)


# =========================================================
# DATA CLEANING
# =========================================================

required_columns = [
    "Product",
    "PaymentMethod",
    "ReferralSource",
    "Quantity",
    "UnitPrice",
    "ItemsInCart"
]

df = df.dropna(subset=required_columns).copy()


# =========================================================
# FEATURE SELECTION
# =========================================================

categorical_features = [
    "Product",
    "PaymentMethod",
    "ReferralSource"
]

numeric_features = [
    "Quantity",
    "UnitPrice",
    "ItemsInCart"
]


# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(
    df[categorical_features]
)


# =========================================================
# SCALE NUMERICAL FEATURES
# =========================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    df[numeric_features]
)


# =========================================================
# CREATE FEATURE MATRIX
# =========================================================

encoded_df = pd.DataFrame(
    encoded_data,
    index=df.index
)

scaled_df = pd.DataFrame(
    scaled_data,
    index=df.index
)

feature_matrix = pd.concat(
    [encoded_df, scaled_df],
    axis=1
)


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("AI Product Recommendation System")

st.write(
    "Select your preferences and get personalized product "
    "recommendations using similarity-based recommendation logic."
)


# =========================================================
# DATASET INFORMATION
# =========================================================

st.info(
    f"Dataset contains {len(df)} records and "
    f"{len(df.columns)} columns."
)


# =========================================================
# USER PREFERENCES
# =========================================================

st.subheader("Select Your Preferences")


# Product selection
product_options = sorted(
    df["Product"].unique().tolist()
)

product = st.selectbox(
    "Preferred Product",
    product_options
)


# Payment method selection
payment_options = sorted(
    df["PaymentMethod"].unique().tolist()
)

payment = st.selectbox(
    "Preferred Payment Method",
    payment_options
)


# Referral source selection
referral_options = sorted(
    df["ReferralSource"].unique().tolist()
)

referral = st.selectbox(
    "Referral Source",
    referral_options
)


# =========================================================
# PRICE INPUT
# =========================================================

minimum_price = 0.0
maximum_price = float(df["UnitPrice"].max())

default_price = maximum_price

max_price = st.number_input(
    "Maximum Preferred Unit Price",
    min_value=minimum_price,
    max_value=maximum_price,
    value=default_price,
    step=10.0
)


# =========================================================
# RECOMMENDATION BUTTON
# =========================================================

if st.button(
    "Get Recommendations",
    type="primary"
):

    # -----------------------------------------------------
    # Create user categorical preferences
    # -----------------------------------------------------

    user_categorical = pd.DataFrame(
        [[
            product,
            payment,
            referral
        ]],
        columns=categorical_features
    )


    # -----------------------------------------------------
    # Create user numerical preferences
    # -----------------------------------------------------

    user_numeric = pd.DataFrame(
        [[
            1,
            max_price,
            3
        ]],
        columns=numeric_features
    )


    # -----------------------------------------------------
    # Encode user categorical preferences
    # -----------------------------------------------------

    user_encoded = encoder.transform(
        user_categorical
    )


    # -----------------------------------------------------
    # Scale user numerical preferences
    # -----------------------------------------------------

    user_scaled = scaler.transform(
        user_numeric
    )


    # -----------------------------------------------------
    # Create user feature vector
    # -----------------------------------------------------

    user_encoded_df = pd.DataFrame(
        user_encoded
    )

    user_scaled_df = pd.DataFrame(
        user_scaled
    )

    user_vector = pd.concat(
        [
            user_encoded_df,
            user_scaled_df
        ],
        axis=1
    )


    # -----------------------------------------------------
    # Calculate cosine similarity
    # -----------------------------------------------------

    similarity_scores = cosine_similarity(
        user_vector,
        feature_matrix
    ).flatten()


    # -----------------------------------------------------
    # Add similarity score
    # -----------------------------------------------------

    recommendation_df = df.copy()

    recommendation_df[
        "SimilarityScore"
    ] = similarity_scores


    # -----------------------------------------------------
    # Sort by similarity
    # -----------------------------------------------------

    recommendation_df = recommendation_df.sort_values(
        by="SimilarityScore",
        ascending=False
    )


    # -----------------------------------------------------
    # Remove duplicate products
    # -----------------------------------------------------

    recommendation_df = recommendation_df.drop_duplicates(
        subset=["Product"]
    )


    # -----------------------------------------------------
    # Select top 5 recommendations
    # -----------------------------------------------------

    top_recommendations = recommendation_df.head(5)


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.subheader("Recommended Products")

    st.write(
        "Recommendations are ranked according to similarity "
        "with your selected preferences."
    )


    for rank, (_, row) in enumerate(
        top_recommendations.iterrows(),
        start=1
    ):

        st.markdown(
            f"### {rank}. {row['Product']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                f"**Unit Price:** "
                f"Rs. {row['UnitPrice']:.2f}"
            )

        with col2:
            st.write(
                f"**Similarity Score:** "
                f"{row['SimilarityScore']:.2f}"
            )

        score = float(
            row["SimilarityScore"]
        )

        score = max(
            0.0,
            min(score, 1.0)
        )

        st.progress(score)

        st.divider()


# =========================================================
# PROJECT INFORMATION
# =========================================================

with st.expander("About This Project"):

    st.write(
        "This project implements a simple AI recommendation "
        "system using user preferences and similarity logic."
    )

    st.write(
        "The system uses product, payment method, referral "
        "source, quantity, unit price, and items in cart "
        "as recommendation features."
    )

    st.write(
        "Cosine similarity is used to compare the user "
        "preference vector with records in the dataset."
    )