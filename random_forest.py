import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# load dataset
df = pd.read_csv("secondary_data.csv", sep=';')

# encode features
label_encoders = {}
for col in df.columns:
    if col == 'class':
        continue
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# class and features
X = df.drop('class', axis=1)
y = df['class']

# split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

# train
clf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42) # best when 50 estimators and 20 max_depth
clf.fit(X_train, y_train)

# dev set
y_dev_pred = clf.predict(X_dev)
print("Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("\nDev Classification Report:\n", classification_report(y_dev, y_dev_pred))

# print incorrect predictions
incorrect_indices = y_dev != y_dev_pred
incorrect_predictions = X_dev[incorrect_indices].copy()
incorrect_predictions['Actual'] = y_dev[incorrect_indices]
incorrect_predictions['Predicted'] = y_dev_pred[incorrect_indices]

print("\nIncorrect Predictions:")
print(incorrect_predictions)

# test set (uncomment for test)
y_test_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# print incorrect predictions
incorrect_indices = y_test != y_test_pred
incorrect_predictions = X_test[incorrect_indices].copy()
incorrect_predictions['Actual'] = y_test[incorrect_indices]
incorrect_predictions['Predicted'] = y_test_pred[incorrect_indices]

print("\nIncorrect Predictions:")
print(incorrect_predictions)