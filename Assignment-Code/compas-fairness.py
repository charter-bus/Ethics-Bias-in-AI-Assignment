import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load COMPAS dataset from ProPublica (assumed local file)
df = pd.read_csv("compas-scores-two-years.csv")

# Filter according to ProPublica's preprocessing
# Reference: https://github.com/propublica/compas-analysis
df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)]
df = df[df.is_recid != -1]
df = df[df.c_charge_degree != 'O']
df = df[df.score_text != 'N/A']
df = df[df.race.isin(['African-American', 'Caucasian'])]

# Select features and target
features = ['age', 'sex', 'race', 'priors_count', 'c_charge_degree']
target = 'two_year_recid'

# Encode categorical features
df_encoded = pd.get_dummies(df[features], drop_first=True)
y = df[target]

# Scale numeric features
scaler = StandardScaler()
X = scaler.fit_transform(df_encoded)

# Split dataset as specified(60% train, 20% val, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_val)

print("\nLogistic Regression Performance:")
print(classification_report(y_val, y_pred_logreg))

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_val)

print("\nDecision Tree Performance:")
print(classification_report(y_val, y_pred_tree))

# Optional: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_val, y_pred_logreg), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Logistic Regression Confusion Matrix')
sns.heatmap(confusion_matrix(y_val, y_pred_tree), annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title('Decision Tree Confusion Matrix')
plt.tight_layout()
plt.show()
