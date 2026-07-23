import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns  
import pickle

print("hello ")   
df = pd.read_csv("mobile_package_price.csv")
print(df)   
df.drop(columns=['Customer_Age'], inplace=True)
print(df.head() )
lr = linear_model.LinearRegression() 
lr.fit(df[["Data_MB","Call_Minutes","SMS_Count","Validity_Days","Internet_Speed_Mbps"]],df.Package_Price)   
print(lr.predict([[7000,700,3000,30,70]]))
print(lr.coef_ ,"cofficients")
print(lr.intercept_,"intercepts")     
package_price =9.67747728e-02* 10000 +   9.67747728e-03*700 -3.60745987e-03*5000 +   2.30887178e+00*45 + 5.51079310e+00*100 -43.138529599886624
print(package_price,"package price")
print(lr.score(df[["Data_MB","Call_Minutes","SMS_Count","Validity_Days","Internet_Speed_Mbps"]],df.Package_Price))

with open("model.pkl", "wb") as f:
    pickle.dump(lr, f)
 
print("Model saved to model.pkl")