from src.models.majority_classifier import train_majority_classifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import load_npz
import numpy as np

# Load X and y data
X_train = load_npz("data/features/X_train_tfidf.npz")
X_test = load_npz("data/features/X_test_tfidf.npz")
y_train = np.load("data/features/y_train.npy")
y_test = np.load("data/features/y_test.npy")

# "Train"
model = train_majority_classifier(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions, zero_division=0))
