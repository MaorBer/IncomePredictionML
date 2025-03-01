from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"KNN Accuracy: {accuracy:.4f}")
    return model


if __name__ == "__main__":
    # Optional: test this script in isolation
    from src.preprocess import get_train_test_data
    X_train, X_test, y_train, y_test = get_train_test_data()
    train_knn(X_train, X_test, y_train, y_test)
