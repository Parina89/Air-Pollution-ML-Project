import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("air_pollution_dataset_75000.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# EDA
# -----------------------------

print(df.head())
print(df.describe())

# Class distribution
plt.figure()
sns.countplot(x="AirQualityLevel", data=df)
plt.title("Class Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Distribution plot
plt.figure()
sns.histplot(df["PM2_5"], kde=True)
plt.title("PM2.5 Distribution")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x="AirQualityLevel", y="PM2_5", data=df)
plt.title("PM2.5 vs Air Quality")
plt.show()

# -----------------------------
# Encode Target
# -----------------------------

le = LabelEncoder()
df["AirQualityLevel"] = le.fit_transform(df["AirQualityLevel"])

X = df.drop("AirQualityLevel", axis=1)
y = df["AirQualityLevel"]

print("Original Distribution:", Counter(y))

# -----------------------------
# Balance Dataset using SMOTE
# -----------------------------

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("Balanced Distribution:", Counter(y_balanced))

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------

models = {
    "RandomForest": RandomForestClassifier(n_estimators=150),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier()
}

results = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    results[name] = model.score(X_test, y_test)

# -----------------------------
# Model Accuracy Comparison
# -----------------------------

print("\nModel Comparison")

for k,v in results.items():
    print(k, ":", v)

plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# -----------------------------
# Feature Importance
# -----------------------------

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_

plt.figure()
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance")
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)

fpr = {}
tpr = {}

for i in range(len(le.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])

plt.figure()

for i in range(len(le.classes_)):
    plt.plot(fpr[i], tpr[i], label=f"Class {i}")

plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
