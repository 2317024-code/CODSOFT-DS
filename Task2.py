import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\aabij\OneDrive\Desktop\IMDb Movies India.csv", encoding="latin1")
df = df.dropna(subset=['Rating'])
df['Year'] = df['Year'].astype(str)
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)
df['Votes'] = df['Votes'].astype(str).str.replace(',','').astype(float)

df[['Duration','Votes']] = df[['Duration','Votes']].fillna(0)
df[['Genre','Director','Actor 1','Actor 2','Actor 3']] = df[['Genre','Director','Actor 1','Actor 2','Actor 3']].fillna("Unknown")

X = df[['Year','Duration','Genre','Votes','Director','Actor 1','Actor 2','Actor 3']]
y = df['Rating']

ct = ColumnTransformer([('enc', OneHotEncoder(handle_unknown='ignore'), ['Year','Genre','Director','Actor 1','Actor 2','Actor 3'])], remainder='passthrough')
X = ct.fit_transform(X)

Xtrain, Xtest, ytrain, ytest, dfTrain, dfTest = train_test_split(X, y, df, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("MSE:", mean_squared_error(ytest, ypred))

dfOut = dfTest.copy()
dfOut['Actual'] = ytest
dfOut['Predicted'] = ypred
print(dfOut[['Name','Year','Genre','Director','Actor 1','Actual','Predicted']].head())
