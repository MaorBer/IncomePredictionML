from adaboost import train_adaboost
from svm import train_svm
from decision_tree import train_decision_tree
from knn import train_knn


if __name__ == "__main__":

    print("Training Models...")
    models = {
        "Adaboost": train_adaboost(),
        "SVM": train_svm(),
        "Decision Tree": train_decision_tree(),
        "KNN": train_knn(),
    }
    print("Training Complete.")
