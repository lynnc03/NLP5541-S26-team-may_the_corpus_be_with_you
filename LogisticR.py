import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

X_train = sparse.load_npz("data/features/X_train_tfidf.npz")
X_test = sparse.load_npz("data/features/X_test_tfidf.npz")
y_train = np.load("data/features/y_train.npy")
y_test = np.load("data/features/y_test.npy")

LR_baseline = LogisticRegression(max_iter=1000, random_state=42)
LR_baseline.fit(X_train, y_train)

y_pred = LR_baseline.predict(X_test)
y_prob = LR_baseline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))