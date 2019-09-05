import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sb
df=pd.read_csv("D://analytics//Analytics_Base//base-analytics-master//data//house_prices.csv")
df.head()
df.info()
df.describe()
sb.pairplot(df)
sb.distplot(df['Price'])
sb.heatmap(df.corr(),annot=True)
df.columns
x=df[['Home', 'SqFt', 'Bedrooms', 'Bathrooms', 'Offers']]
y=df[['Price']]
from sklearn.cross_validation import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
 from sklearn.linear_model import LinearRegression
 lm=LinearRegression()
 print(lm.intercept_)
lm.coef_
pred=lm.predict(X_train)
mp.scatter(y_train,pred)
sb.distplot(y_train-pred)
