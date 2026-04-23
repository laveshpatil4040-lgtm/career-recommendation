import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint

X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").values.ravel()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

base_model = RandomForestClassifier(random_state=42)

cv_scores = cross_val_score(base_model, X, y, cv=5)
print("CV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("Std CV accuracy:", cv_scores.std())

param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": randint(2, 12),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X, y)

print("Best Parameters:", search.best_params_)
print("Best CV Score:", search.best_score_)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(best_model, "career_model.pkl")
print("Final tuned model saved successfully.")