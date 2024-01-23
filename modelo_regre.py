## Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

## Dataset Analysis

# Load the dataset
df = sns.load_dataset("tips")

# Display the first 5 rows of the dataset
print(df.head())
print()

# Basic statistics
print(df.describe())

# Plots

fig, axs = plt.subplots(2, 3, figsize=(10,7))

sns.histplot(data=df, x="day", hue="sex", multiple="dodge", ax=axs[0,0])
axs[0,0].set_title("Distribution of days")

sns.histplot(data=df, x="tip", hue="sex", multiple="dodge", ax=axs[0,1])
axs[0,1].set_title("Tip Distribution")

sns.histplot(data=df, x="total_bill", hue="sex", multiple="dodge", ax=axs[1,0])
axs[1,0].set_title("Total Bill Distribution")

sns.countplot(data=df, x="time", ax=axs[1,1])
axs[1,1].set_title("Time Distribution")

sns.histplot(data=df, x="day", hue="smoker", multiple="dodge", ax=axs[0,2])
axs[0,2].set_title("Smoker Distribution by Day")

sns.histplot(data=df, x="total_bill", hue="smoker", multiple="dodge", ax=axs[1,2])
axs[1,2].set_title("Smoker Distribution by Total Bill")

fig.tight_layout()
plt.show()

numerical_df = df.drop(["sex", "time", "smoker", "day"], axis=1)
cor = numerical_df.corr()
sns.heatmap(cor, annot=True)
sns.pairplot(df, hue="sex")
plt.show()

## Preprocessing

x = df["total_bill"].values.reshape(-1,1)
y = df["tip"].values.reshape(-1,1)

scalerx = StandardScaler()
scalery = StandardScaler()

scl_x = scalerx.fit_transform(x)
scl_y = scalery.fit_transform(y)

## Model Training

model = LinearRegression()
model.fit(scl_x, scl_y)

## Model Visualization

plt.scatter(x=scl_x, y=scl_y)
plt.plot(scl_x, model.predict(scl_x), c="red")
plt.ylabel("Tips")
plt.xlabel("Total_bill")
plt.show()