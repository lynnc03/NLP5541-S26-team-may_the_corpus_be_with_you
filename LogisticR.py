import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def run_experiment(name, path):
    print(f"\n===== {name} =====")

    # Load data
    X_train = sparse.load_npz(f"{path}/X_train_tfidf.npz")
    X_test = sparse.load_npz(f"{path}/X_test_tfidf.npz")
    y_train = np.load(f"{path}/y_train.npy")
    y_test = np.load(f"{path}/y_test.npy")

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Results
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))


#  3 experiments
run_experiment("TF-IDF (clean)", "/data/features/clean")
run_experiment("TF-IDF (disfluency)", "/data/features/disfluency")
run_experiment("TF-IDF (special tokens)", "/data/features/special_tokens")
