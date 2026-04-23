import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("career_recommendation_expanded.csv")

print("First 5 rows:")
print(df.head())
print("Dataset shape:", df.shape)

print("\nDataset info:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

df = df.drop(columns=["student_id"])
print("\nColumns after dropping student_id:")
print(df.columns)

X = df.drop("career_label", axis=1)
y = df["career_label"]

categorical_cols = X.select_dtypes(include=["object", "str"]).columns
numerical_cols = X.select_dtypes(exclude=["object", "str"]).columns

print("\nCategorical columns:", list(categorical_cols))
print("Numerical columns:", list(numerical_cols))

X = pd.get_dummies(X, columns=categorical_cols)
print("\nEncoded feature sample:")
print(X.head())
print("Encoded feature shape:", X.shape)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nEncoded target sample:", y[:10])
print("Career classes:", label_encoder.classes_)

joblib.dump(label_encoder, "label_encoder.pkl")
print("Label encoder saved as label_encoder.pkl")

print("\nStarting train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-test split completed successfully")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_train, columns=["career_label"]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=["career_label"]).to_csv("y_test.csv", index=False)

print("\nPreprocessed files saved successfully.")
print("Preprocessing completed successfully")