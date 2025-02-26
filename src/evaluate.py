import pandas as pd
from src.preprocess import get_train_test_data
from src.adaboost import train_adaboost
from src.svm import train_svm
from src.decision_tree import train_decision_tree
from src.knn import train_knn

def evaluate_models():
    # Load and split data ONCE here as well
    X_train, X_test, y_train, y_test = get_train_test_data()

    # Train each model
    results = {}
    results["Adaboost"] = train_adaboost(X_train, X_test, y_train, y_test)
    results["SVM"] = train_svm(X_train, X_test, y_train, y_test)
    results["Decision Tree"] = train_decision_tree(X_train, X_test, y_train, y_test)
    results["KNN"] = train_knn(X_train, X_test, y_train, y_test)

    # Evaluate each model on the same test data
    accuracies = {name: model.score(X_test, y_test) for name, model in results.items()}

    # Save and display results
    df_results = pd.DataFrame(accuracies.items(), columns=["Model", "Accuracy"])
    df_results.to_csv("results/model_comparison.csv", index=False)
    print(df_results)

if __name__ == "__main__":
    evaluate_models()
