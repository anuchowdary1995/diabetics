import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import datasets
diabetes=datasets.load_diabetes()
#print (diabetes.data.shape)
#print (diabetes.target.shape)
#print (diabetes.feature_names)
feature_cols=diabetes.feature_names
#diabetes=pd.DataFrame(diabetes)
#print (diabetes.iloc[0:100,:])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(diabetes.data,diabetes.target,test_size=0.2,random_state=0)
#print(X_train)
model=LinearRegression()
model.fit(X_train,y_train)
print ("Model_Score or R-Squared",model.score(X_test,y_test))
print ("Model_Quotients",model.coef_)
print ("Model Intercept",model.intercept_)
#To View featurewise model co-efficients
A=list(zip(feature_cols,model.coef_))
print("Feature wise co-efficients",A)
#print(model.predict(X_test))
y_pred=model.predict(X_test)
plt.plot(y_test,y_pred,".")
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
#plt.show()
from sklearn import metrics
#Absolute Mean Squared Error
print("Absolute_Mean_Squared_Error",metrics.mean_absolute_error(y_test,y_pred))
#Mean Squared Error
print("Mean_Squared_Error",metrics.mean_squared_error(y_test,y_pred))
#Root Mean squared error
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#To get r_squared and adjusted_r_squared
SS_Residual = sum((y_test-y_pred)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
print (r_squared,"r_squared")
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print(adjusted_r_squared,"adjusted_r_squared")

#shapes
print(X_train.shape[0])#rows
print(X_train.shape[1])#columns

#To check the significant variables
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

#Calculate ViF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_train=pd.DataFrame(X_train)
vif=pd.DataFrame()
vif["Vif Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["Features"]=X_train.columns
print (vif.round(1))
