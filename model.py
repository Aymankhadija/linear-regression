import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
  
df = pd.read_csv("mobile_package_price.csv")
print(df)   
df.drop(columns=['Customer_Age'], inplace=True)
print(df.head() )
print(df.isnull().sum())
lr = linear_model.LinearRegression() 
lr.fit(df[["Data_MB","Call_Minutes","SMS_Count","Validity_Days","Internet_Speed_Mbps"]],df.Package_Price)   
print(lr.predict([[7000,700,3000,30,70]]),"prediction ")
print(lr.coef_ ,"cofficients")
print(lr.intercept_,"intercepts")     
package_price =9.67747728e-02* 10000 +   9.67747728e-03*700 -3.60745987e-03*5000 +   2.30887178e+00*45 + 5.51079310e+00*100 -43.138529599886624
print(package_price,"package price")

X = df[["Data_MB","Call_Minutes","SMS_Count","Validity_Days","Internet_Speed_Mbps"]]
y = df["Package_Price"]

score = lr.score(X, y)
model_data = {
    "model": lr,
    "score": score
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model Score:", score)
print("Model saved successfully!")