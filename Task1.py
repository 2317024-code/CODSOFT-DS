import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("C:\\Users\\aabij\\OneDrive\\Desktop\\Titanic-Dataset (2).csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Pclass'], drop_first=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42) 
model.fit(X_train, Y_train)
ypred = model.predict(X_test)

results_df = pd.DataFrame({'Actual': Y_test, 'Predicted': ypred})
print("First 5 rows of Prediction by Abijin:")
print(results_df.head())

print(f"\nModel Accuracy: {accuracy_score(Y_test, ypred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(Y_test, ypred))
print("Classification Report:")
print(classification_report(Y_test, ypred))