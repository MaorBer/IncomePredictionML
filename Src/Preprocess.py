import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data():
    """Fetch the Census Income dataset from UCI Repository."""
    dataset = fetch_ucirepo(id=2)  # ID=2 corresponds to the Adult dataset

    # Convert dataset to DataFrame
    df = dataset.data.original
    return df


def preprocess_data(df):
    """Preprocess dataset: clean missing values, encode categorical variables, and split features/target."""
    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_cols}

    for col in categorical_cols:
        df[col] = label_encoders[col].fit_transform(df[col])

    # Split into X (features) and y (target)
    X = df.drop(columns=["income"])  # Features
    y = df["income"]  # Target

    return X, y


def get_train_test_data():
    """Returns train-test split of dataset."""
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
