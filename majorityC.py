from src.models.majority_classifier import train_majority_classifier
from sklearn.metrics import classification_report
from scipy.sparse import load_npz
import numpy as np


def run_majority_experiment(name, path):
    print(f"\n===== {name} =====")

    X_train = load_npz(f"{path}/X_train_tfidf.npz")
    X_test = load_npz(f"{path}/X_test_tfidf.npz")
    y_train = np.load(f"{path}/y_train.npy")
    y_test = np.load(f"{path}/y_test.npy")

    model = train_majority_classifier(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))


# utterance-level features — split controlled by PID, matching the transformer
run_majority_experiment("Majority (clean)",          "data/features/utterance/clean")
run_majority_experiment("Majority (disfluency)",     "data/features/utterance/disfluency")
run_majority_experiment("Majority (special tokens)", "data/features/utterance/special_tokens")
