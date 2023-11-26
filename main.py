import pandas as pd

df = pd.read_csv("data/pizzaplace.csv")

print(df.head()) # overview of data

print(df.info()) # getting types of data

print(df.isna().sum()) # no null values

print(df['size'].value_counts())

print(df['name'].value_counts())

print(df["type"].value_counts())

print(df["time"].str.match(r'\d{2}:\d{2}:\d{2}').all())