import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Fetch the Census Income dataset from UCI Repository."""
    dataset = fetch_ucirepo(id=2)  # ID=2 corresponds to the Adult dataset

    # Convert dataset to DataFrame
    df = dataset.data.original
    return df.copy()  # Ensure we are working with a copy of the dataset

def preprocess_data(df):
    """Preprocess dataset: clean missing values, encode categorical variables, and split features/target."""
    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical variables (excluding target column)
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = categorical_cols.drop("income")  # Exclude target from encoding

    label_encoders = {col: LabelEncoder() for col in categorical_cols}

    for col in categorical_cols:
        df.loc[:, col] = label_encoders[col].fit_transform(df[col])  # Use .loc[] to avoid warning

    # Ensure correct target encoding
    df.loc[:, "income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    # Check for NaN values after mapping
    if df["income"].isnull().sum() > 0:
        print("⚠️ Warning: NaN values found in 'income' column! Dropping affected rows.")
        df = df.dropna(subset=["income"])  # Remove rows with NaN in 'income'

    # Debugging prints
    print(f"🔍 Number of samples after preprocessing: {df.shape[0]}")
    print(f"🔍 Unique values in 'income': {df['income'].unique()}")

    # Convert target to integers
    y = df["income"].astype(int)
    X = df.drop(columns=["income"])  # Features

    return X, y

def get_train_test_data():
    """Returns train-test split of dataset."""
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Debugging prints
    print("Unique values in y_train:", set(y_train))
    print("Data type of y_train:", y_train.dtype)

    return X_train, X_test, y_train, y_test
