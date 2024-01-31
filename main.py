import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
df["datetime"]

#assign hour of datetime column to hour column
df["hour"] = df["datetime"].dt.hour
df["hour"]

#assign week day number of datetime column to week_day column
df["week_day"] = df["datetime"].dt.day_of_week

print(df)

print(df["hour"].value_counts())

print(df["week_day"].value_counts())

#drop all unecessery columns
df.drop(["Unnamed: 0", "id", "date", "time", "datetime"], axis=1, inplace=True)

print(df)

#Visualizing the distribution of pizza types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='type')
plt.title('Pizza Type Distribution')
plt.xlabel('Pizza Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
#we see 4 types of pizza with amount of distribution

# Visualize the distribution of pizza sizes
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='size')
plt.title('Pizza Size Distribution')
plt.xlabel('Pizza Size')
plt.ylabel('Count')
plt.show()
#we see most common size bought by users is L size

df = df.query("size not in ['XXL', 'XL']")

daily_orders = df['week_day'].value_counts().sort_index().reindex(range(7), fill_value=0)
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10, 6))
sns.barplot(x=days_of_week, y=daily_orders.values)
plt.title('Number of Orders by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.show()

#Visualizing the distribution of pizza hours
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='hour')
plt.title('Pizza hour Distribution')
plt.xlabel('Pizza hour')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

df = df.query("hour not in [9, 10, 23]")

le = LabelEncoder()

df["name"] = le.fit_transform(df["name"])
df["size"] = le.fit_transform(df["size"])
df["type"] = le.fit_transform(df["type"])

df

X= df.drop(['price', "hour", 'week_day'],axis=1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

lr = LinearRegression()
lr.fit(X_train,y_train)

rf  = RandomForestRegressor()
rf.fit(X_train,y_train)

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

xg = XGBRegressor()
xg.fit(X_train,y_train)

tree_reg = DecisionTreeRegressor()

# Fit the model to the training data
tree_reg.fit(X_train, y_train)

y_pred1 = lr.predict(X_test)
y_pred2 = rf.predict(X_test)
y_pred3 = gbr.predict(X_test)
y_pred4 = xg.predict(X_test)
y_pred5= tree_reg.predict(X_test)

from sklearn import metrics
score1 = metrics.mean_squared_error(y_test,y_pred1) #linear regression
score2 = metrics.mean_squared_error(y_test,y_pred2) #random forest
score3 = metrics.mean_squared_error(y_test,y_pred3) #gbr
score4 = metrics.mean_squared_error(y_test,y_pred4) #xg
score5 = metrics.mean_squared_error(y_test,y_pred5) #dt

print(score1,score2,score3,score4,score5)

#GradientBoostingRegressor regression is best