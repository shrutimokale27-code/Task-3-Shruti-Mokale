import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = "dataset/Dataset for Data Analytics.xlsx.xlsx"
df = pd.read_excel(file_path)

print("AI PRODUCT RECOMMENDATION SYSTEM")
print("---------------------------------")

# Clean data
df = df.dropna(
    subset=[
        "Product",
        "PaymentMethod",
        "ReferralSource",
        "Quantity",
        "UnitPrice",
        "ItemsInCart"
    ]
).copy()

# Features used for recommendation
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

# Encode categorical features
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(
    df[categorical_features]
)

# Scale numerical features
scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    df[numeric_features]
)

# Combine features
feature_matrix = pd.concat(
    [
        pd.DataFrame(encoded_data),
        pd.DataFrame(scaled_data)
    ],
    axis=1
)

# User input
print("\nAvailable Products:")
print(df["Product"].unique())

product = input("\nEnter preferred product: ").strip().title()

print("\nAvailable Payment Methods:")
print(df["PaymentMethod"].unique())

payment = input("Enter preferred payment method: ").strip().title()

print("\nAvailable Referral Sources:")
print(df["ReferralSource"].unique())

referral = input("Enter preferred referral source: ").strip().title()

# Ask for price preference
max_price = float(
    input("\nEnter your maximum preferred unit price: ")
)

# Create user profile
user_categorical = pd.DataFrame(
    [[product, payment, referral]],
    columns=categorical_features
)

user_numeric = pd.DataFrame(
    [[1, max_price, 3]],
    columns=numeric_features
)

# Encode user profile
user_encoded = encoder.transform(
    user_categorical
)

user_scaled = scaler.transform(
    user_numeric
)

user_vector = pd.concat(
    [
        pd.DataFrame(user_encoded),
        pd.DataFrame(user_scaled)
    ],
    axis=1
)

# Calculate cosine similarity
similarity_scores = cosine_similarity(
    user_vector,
    feature_matrix
).flatten()

# Add similarity score
df["SimilarityScore"] = similarity_scores

# Sort by similarity
recommendations = df.sort_values(
    by="SimilarityScore",
    ascending=False
)

# Remove duplicate products
recommendations = recommendations.drop_duplicates(
    subset=["Product"]
)

# Display recommendations
print("\nRecommended Products")
print("---------------------")

top_recommendations = recommendations.head(5)

for _, row in top_recommendations.iterrows():
    print(
        f"{row['Product']} | "
        f"Unit Price: ₹{row['UnitPrice']:.2f} | "
        f"Similarity Score: {row['SimilarityScore']:.2f}"
    )