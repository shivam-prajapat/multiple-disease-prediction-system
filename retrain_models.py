"""
Retrain all disease prediction models with the current scikit-learn version.
This fixes the InconsistentVersionWarning and wrong predictions.
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# ============================================================
# 1. DIABETES MODEL (SVM) - Pima Indians Diabetes Dataset
# ============================================================
print("=" * 60)
print("Training Diabetes Model (SVM)...")
print("=" * 60)

url_diabetes = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
cols_diabetes = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df_diab = pd.read_csv(url_diabetes, names=cols_diabetes)
print(f"Loaded diabetes dataset: {df_diab.shape}")

X_diab = df_diab.drop('Outcome', axis=1)
y_diab = df_diab['Outcome']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diab, y_diab, test_size=0.2, stratify=y_diab, random_state=2
)

scaler_diab = StandardScaler()
X_train_d_scaled = scaler_diab.fit_transform(X_train_d)
X_test_d_scaled = scaler_diab.transform(X_test_d)

diab_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
diab_model.fit(X_train_d_scaled, y_train_d)

train_acc = accuracy_score(y_train_d, diab_model.predict(X_train_d_scaled))
test_acc = accuracy_score(y_test_d, diab_model.predict(X_test_d_scaled))
print(f"Diabetes Model -> Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

pickle.dump({'model': diab_model, 'scaler': scaler_diab}, open('diabetes_model.sav', 'wb'))
print("Saved diabetes_model.sav")

# Verify
test_pos = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
test_neg = [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]
print(f"  Verify Diabetic (expect 1):     {diab_model.predict(scaler_diab.transform(test_pos))[0]}")
print(f"  Verify Non-Diabetic (expect 0): {diab_model.predict(scaler_diab.transform(test_neg))[0]}")


# ============================================================
# 2. HEART DISEASE MODEL (Logistic Regression) - Cleveland
# ============================================================
print()
print("=" * 60)
print("Training Heart Disease Model (Logistic Regression)...")
print("=" * 60)

url_heart = 'https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv'
df_heart = pd.read_csv(url_heart)
print(f"Loaded heart dataset: {df_heart.shape}")
print(f"Columns: {list(df_heart.columns)}")

X_heart = df_heart.drop('target', axis=1)
y_heart = df_heart['target']
print(f"Target distribution: {y_heart.value_counts().to_dict()}")

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart, y_heart, test_size=0.2, stratify=y_heart, random_state=2
)

scaler_heart = StandardScaler()
X_train_h_scaled = scaler_heart.fit_transform(X_train_h)
X_test_h_scaled = scaler_heart.transform(X_test_h)

heart_model = LogisticRegression(max_iter=1000, random_state=42)
heart_model.fit(X_train_h_scaled, y_train_h)

train_acc = accuracy_score(y_train_h, heart_model.predict(X_train_h_scaled))
test_acc = accuracy_score(y_test_h, heart_model.predict(X_test_h_scaled))
print(f"Heart Model -> Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

pickle.dump({'model': heart_model, 'scaler': scaler_heart}, open('trained_model.sav', 'wb'))
print("Saved trained_model.sav")


# ============================================================
# 3. PARKINSONS MODEL (SVM) - UCI Parkinsons Dataset
# ============================================================
print()
print("=" * 60)
print("Training Parkinsons Model (SVM)...")
print("=" * 60)

url_park = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
df_park = pd.read_csv(url_park)
print(f"Loaded parkinsons dataset: {df_park.shape}")

# Drop 'name' column
if 'name' in df_park.columns:
    df_park = df_park.drop('name', axis=1)

X_park = df_park.drop('status', axis=1)
y_park = df_park['status']
park_feature_cols = list(X_park.columns)
print(f"Features ({len(park_feature_cols)}): {park_feature_cols}")
print(f"Target distribution: {y_park.value_counts().to_dict()}")

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_park, y_park, test_size=0.2, stratify=y_park, random_state=2
)

scaler_park = StandardScaler()
X_train_p_scaled = scaler_park.fit_transform(X_train_p)
X_test_p_scaled = scaler_park.transform(X_test_p)

park_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
park_model.fit(X_train_p_scaled, y_train_p)

train_acc = accuracy_score(y_train_p, park_model.predict(X_train_p_scaled))
test_acc = accuracy_score(y_test_p, park_model.predict(X_test_p_scaled))
print(f"Parkinsons Model -> Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

pickle.dump({'model': park_model, 'scaler': scaler_park, 'feature_cols': park_feature_cols},
            open('parkinsons_model.sav', 'wb'))
print("Saved parkinsons_model.sav")
print(f"Feature column order: {park_feature_cols}")

print()
print("=" * 60)
print("ALL MODELS RETRAINED AND SAVED SUCCESSFULLY!")
print("=" * 60)
