# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import joblib

# Load the dataset
df = pd.read_csv('framingham.csv')

# Basic EDA
print("First five rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['cigsPerDay'].fillna(df['cigsPerDay'].mean(), inplace=True)
df['BPMeds'].fillna(df['BPMeds'].mode()[0], inplace=True)
df['totChol'].fillna(df['totChol'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df['heartRate'].fillna(df['heartRate'].mean(), inplace=True)
df['glucose'].fillna(df['glucose'].mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Target variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='TenYearCHD', data=df)
plt.title("Class Distribution of CHD")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Define features and target
X = df.drop(columns='TenYearCHD')
y = df['TenYearCHD']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with Cross-Validation
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("Logistic Regression Cross-Validation Scores:", lr_cv_scores)
print("Logistic Regression Mean CV Accuracy:", lr_cv_scores.mean())

# Decision Tree with Hyperparameter Tuning
dt_params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train_scaled, y_train)
best_dt_model = dt_grid_search.best_estimator_
dt_predictions = best_dt_model.predict(X_test_scaled)
print("\nDecision Tree Best Parameters:", dt_grid_search.best_params_)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))

# Support Vector Machine
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
print("\nSVM Accuracy:", accuracy_score(y_test, svm_predictions))

# XGBoost with Hyperparameter Tuning
xgb_params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=5, scoring='accuracy')
xgb_grid_search.fit(X_train_scaled, y_train)
best_xgb_model = xgb_grid_search.best_estimator_
xgb_predictions = best_xgb_model.predict(X_test_scaled)
print("\nXGBoost Best Parameters:", xgb_grid_search.best_params_)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_predictions))

# Ensemble Model (Voting Classifier)
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('dt', best_dt_model),
        ('svm', svm_model),
        ('xgb', best_xgb_model)
    ],
    voting='soft'  # Use 'soft' for probability-based voting
)
ensemble_model.fit(X_train_scaled, y_train)
ensemble_predictions = ensemble_model.predict(X_test_scaled)
print("\nEnsemble Model Accuracy:", accuracy_score(y_test, ensemble_predictions))

# ROC-AUC Scores
print("\nROC-AUC Scores:")
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1]))
print("SVM ROC-AUC:", roc_auc_score(y_test, svm_model.predict_proba(X_test_scaled)[:, 1]))
print("XGBoost ROC-AUC:", roc_auc_score(y_test, best_xgb_model.predict_proba(X_test_scaled)[:, 1]))
print("Ensemble Model ROC-AUC:", roc_auc_score(y_test, ensemble_model.predict_proba(X_test_scaled)[:, 1]))

# Visualize Model Comparison
model_accuracies = {
    "Logistic Regression": accuracy_score(y_test, lr_predictions),
    "Decision Tree": accuracy_score(y_test, dt_predictions),
    "SVM": accuracy_score(y_test, svm_predictions),
    "XGBoost": accuracy_score(y_test, xgb_predictions),
    "Ensemble": accuracy_score(y_test, ensemble_predictions)
}

plt.bar(model_accuracies.keys(), model_accuracies.values(), color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.show()

# Save Ensemble Model
joblib.dump(ensemble_model, 'ensemble_model.pkl')
print("\nEnsemble Model Saved as 'ensemble_model.pkl'.")

# Testing with new data
new_data_values = [[1, 70, 2, 1, 30, 1, 0, 1, 1, 300, 180, 110, 35, 100, 200]]  # Example input
new_data = pd.DataFrame(new_data_values, columns=X.columns)
loaded_model = joblib.load('ensemble_model.pkl')
prediction = loaded_model.predict(new_data)
print("\nPrediction for New Data:", "High Risk" if prediction[0] == 1 else "Low Risk")
