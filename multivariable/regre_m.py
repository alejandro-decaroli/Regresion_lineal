## Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from regressors import stats

## Dataset Analysis

# Load the dataset
df = pd.read_csv("/home/alejandro/proyectos/youtube/regre_l/multivariable/insurance.csv")

# Display the first 5 rows of the dataset
print(df.head())
print()

# Basic statistics
print(df.describe())

# Plots
numerical_df = df.drop(["sex","smoker","region"], axis=1)
sns.heatmap(data=numerical_df.corr(), annot=True)
sns.pairplot(data=df, hue="sex")
plt.show()

## Preprocessing

encode_df = pd.get_dummies(df, columns=["sex","smoker","region"], drop_first=True, dtype=int)

x = encode_df.drop(["charges", "sex_male", "region_northwest", "region_southeast", "region_southwest"], axis=1).values
y = (encode_df["charges"].values).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_std = StandardScaler().fit(x)
y_std = StandardScaler().fit(y)

x_train = x_std.transform(x_train)
x_test = x_std.transform(x_test)
y_train = y_std.transform(y_train)
y_test = y_std.transform(y_test)

## Model Training

model = LinearRegression(fit_intercept=False)
model.fit(x_train, y_train)

## Metrics

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print()
print(f"mean square error: {mse}")
print(f"r2 score: {r2}")
print()

print("===================Sumary===================")
x_col = encode_df.drop(["charges", "sex_male", "region_northwest", "region_southeast", "region_southwest"], axis=1).columns
#model.intercept_ = model.intercept_[0]
model.coef_ = model.coef_.reshape(-1)
y_test = y_test.reshape(-1)
print()
print(stats.summary(model, x_test, y_test, x_col))

residual = np.subtract(y_test,y_pred.reshape(-1))
plt.scatter(y_pred, residual)
plt.show()


