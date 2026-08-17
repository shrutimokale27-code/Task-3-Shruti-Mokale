# AI Product Recommendation System

## Project Overview

This project implements a simple AI Product Recommendation System using user preferences and similarity-based recommendation logic.

The system takes user preferences such as preferred product, payment method, referral source, and maximum preferred unit price. It then compares the user preference profile with records from the dataset using Cosine Similarity and displays the top recommended products.

## Objective

The main objective of this project is to build a simple recommendation system that can:

- Take user preferences as input
- Match user preferences with available product data
- Calculate similarity between user preferences and dataset records
- Rank products based on similarity
- Display the top 5 recommended products

## Dataset

The project uses an Excel dataset named:

`Dataset for Data Analytics.xlsx.xlsx`

Dataset details:

- Records: 1200
- Columns: 14

Important columns used in the recommendation system:

- Product
- Quantity
- UnitPrice
- PaymentMethod
- ItemsInCart
- ReferralSource

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- OpenPyXL
- Cosine Similarity

## Recommendation Method

The system uses:

1. Categorical feature encoding using One-Hot Encoding
2. Numerical feature scaling using StandardScaler
3. Feature combination
4. Cosine Similarity
5. Similarity-based ranking
6. Top 5 product recommendations

## Application Workflow

User Preferences
        |
        v
Feature Processing
        |
        v
One-Hot Encoding + Scaling
        |
        v
Cosine Similarity
        |
        v
Similarity Ranking
        |
        v
Top 5 Recommended Products

## Features

The Streamlit application provides:

- Preferred Product selection
- Preferred Payment Method selection
- Referral Source selection
- Maximum Preferred Unit Price
- Recommendation button
- Top 5 recommended products
- Unit price of recommended products
- Similarity score

## How to Run

### 1. Install required libraries

```bash
pip install pandas openpyxl scikit-learn streamlit