from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from preprocess import get_train_test_data


def train_svm():
    X_train, X_test, y_train, y_test = get_train_test_data()

    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"SVM Accuracy: {accuracy:.4f}")
    return model


if __name__ == "__main__":
    train_svm()
