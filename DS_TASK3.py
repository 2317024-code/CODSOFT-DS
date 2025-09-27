import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('C://Users//aabij//OneDrive//Desktop//IRIS.csv')
print(df.head())
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])  
X = df.drop('species', axis=1)  
y = df['species']              
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

sample_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                             columns=X.columns)
predicted_class = le.inverse_transform(clf.predict(sample_flower))
print("\nPredicted species for sample flower:", predicted_class[0])