import pandas as pd

df = pd.read_csv("data/pizzaplace.csv")

print(df.head())

print(df.info())

print(df.describe())

print(df.isna().sum()) # no null values

print(df['size'].value_counts())

print(df['name'].value_counts())