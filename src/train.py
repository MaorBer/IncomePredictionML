from src.adaboost import train_adaboost
from src.svm import train_svm
from src.decision_tree import train_decision_tree
from src.knn import train_knn
from src.preprocess import get_train_test_data

if __name__ == "__main__":
    print("Training Models...")

    # Load and split data ONCE
    X_train, X_test, y_train, y_test = get_train_test_data()

    models = {
        "AdaBoost": train_adaboost(X_train, X_test, y_train, y_test),
        "SVM": train_svm(X_train, X_test, y_train, y_test),
        "Decision Tree": train_decision_tree(X_train, X_test, y_train, y_test),
        "KNN": train_knn(X_train, X_test, y_train, y_test),
    }

    print("Training Complete.")
