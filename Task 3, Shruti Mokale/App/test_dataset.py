import pandas as pd

file_path = "dataset/Dataset for Data Analytics.xlsx.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Rows and Columns:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())