import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('C://Users//aabij//OneDrive//Desktop//advertising.csv')
print(df.head())
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
Comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted Sales:\n", Comparison)
New_Budget = pd.DataFrame([[150, 30, 20]], columns=['TV', 'Radio', 'Newspaper'])
Predicted = model.predict(New_Budget)
print("\nPredicted Sales for new advertising budget:", Predicted[0])