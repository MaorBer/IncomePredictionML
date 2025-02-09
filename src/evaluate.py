import pandas as pd
from adaboost import train_adaboost
from svm import train_svm
from decision_tree import train_decision_tree
from knn import train_knn


def evaluate_models():
    results = {}

    results["Adaboost"] = train_adaboost()
    results["SVM"] = train_svm()
    results["Decision Tree"] = train_decision_tree()
    results["KNN"] = train_knn()

    accuracies = {name: model.score(X_test, y_test) for name, model in results.items()}

    df_results = pd.DataFrame(accuracies.items(), columns=["Model", "Accuracy"])
    df_results.to_csv("results/model_comparison.csv", index=False)
    print(df_results)


if __name__ == "__main__":
    evaluate_models()
