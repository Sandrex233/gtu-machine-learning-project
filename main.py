import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/pizzaplace.csv")

print(df.head()) # overview of data

print(df.info()) # getting types of data

print(df.isna().sum()) # no null values

print(df['size'].value_counts())

print(df['name'].value_counts())

print(df["type"].value_counts())

print(df["time"].str.match(r'\d{2}:\d{2}:\d{2}').all())

print(df["date"].str.match(r'\d{4}-\d{2}-\d{2}').all())

#concat date and time column to transform it in datetime type column
df["datetime"] = pd.to_datetime(df["date"] + ' ' + df["time"])

#assign hour of datetime column to hour column
df["hour"] = df["datetime"].dt.hour

#assign week day number of datetime column to week_day column
df["week_day"] = df["datetime"].dt.day_of_week

print(df)

#little exploration if hour or week_day columns contain any ilegal values
print(df["hour"].value_counts())

print(df["week_day"].value_counts())

#drop all unecessery columns
df.drop(["Unnamed: 0", "id", "date", "time", "datetime"], axis=1, inplace=True)

print(df)

#create labelencoder instance and transform all textual column values to numerical values
le = LabelEncoder()

df["name"] = le.fit_transform(df["name"])
df["size"] = le.fit_transform(df["size"])
df["type"] = le.fit_transform(df["type"])

print(df)