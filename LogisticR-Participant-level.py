import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt


# UTTERANCE-LEVEL 
def compute_threshold_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )

    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision_control": precision[0],
        "recall_control": recall[0],
        "f1_control": f1[0],
        "precision_sli": precision[1],
        "recall_sli": recall[1],
        "f1_sli": f1[1],
        "f1_macro": f1_macro,
    }


def threshold_sweep(labels, probs, thresholds=np.arange(0.05, 0.96, 0.01)):
    results = []

    for t in thresholds:
        results.append(compute_threshold_metrics(labels, probs, t))

    best = max(results, key=lambda x: x["f1_macro"])
    return best, results


# PID-LEVEL
def agg_participant_preds(df, prob_thresh=0.5, vote_thresh=0.5):

    grouped = df.groupby("pid")

    out = grouped.agg(
        true_label=("true_label", "first"),
        mean_prob=("prob_sli", "mean"),
        vote_rate=("pred_sli", "mean"),
    ).reset_index()

    out["pred_mean"] = (out["mean_prob"] >= prob_thresh).astype(int)
    out["pred_vote"] = (out["vote_rate"] >= vote_thresh).astype(int)

    return out


def compute_participant_metrics(df, pred_col):
    y_true = df["true_label"].values
    y_pred = df[pred_col].values

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_control": precision[0],
        "recall_control": recall[0],
        "f1_control": f1[0],
        "precision_sli": precision[1],
        "recall_sli": recall[1],
        "f1_sli": f1[1],
        "f1_macro": f1_macro,
    }


def participant_threshold_sweep(df, thresholds=np.arange(0.05, 0.96, 0.01)):
    results = []

    for t in thresholds:
        temp = agg_participant_preds(df, prob_thresh=t)
        metrics = compute_participant_metrics(temp, "pred_mean")
        metrics["threshold"] = t
        results.append(metrics)

    best = max(results, key=lambda x: x["f1_macro"])
    return best, results



# MAIN EXPERIMENT
def run_logreg_experiment(name, path):

    print(f"\n===== {name} =====")

    # LOAD
    X_train = sparse.load_npz(f"{path}/X_train_tfidf.npz")
    X_test = sparse.load_npz(f"{path}/X_test_tfidf.npz")

    y_train = np.load(f"{path}/y_train.npy")
    y_test = np.load(f"{path}/y_test.npy")

    meta_test = pd.read_csv(f"{path}/test_metadata.csv")

    # CHECK
    assert len(meta_test) == len(y_test), "Metadata mismatch!"

    # TRAIN
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # PREDICT
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\n--- Utterance-level (0.5 threshold) ---")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # BUILD DF
    df = pd.DataFrame({
        "pid": meta_test["pid"].values,
        "file_id": meta_test["file_id"].values,
        "true_label": y_test,
        "prob_sli": y_prob
    })

    # UTTERANCE THRESHOLD
    best_uttr, _ = threshold_sweep(y_test, y_prob)
    best_thresh = best_uttr["threshold"]

    print(f"\nBest utterance threshold: {best_thresh:.2f}")

    df["pred_sli"] = (df["prob_sli"] >= best_thresh).astype(int)

    # PID (default)
    part_df = agg_participant_preds(df)
    
    print("\n--- Participant Distribution (True Labels) ---")
    print(part_df["true_label"].value_counts())

    n_control = (part_df["true_label"] == 0).sum()
    n_sli = (part_df["true_label"] == 1).sum()

    print(f"Control (0): {n_control}")
    print(f"SLI (1): {n_sli}")
    print(f"Total participants: {len(part_df)}")
    

    print("\n--- Participant (default 0.5) ---")
    print(compute_participant_metrics(part_df, "pred_mean"))

    # PID TUNING
    best_part, _ = participant_threshold_sweep(df)
    best_part_thresh = best_part["threshold"]

    print(f"\nBest participant threshold: {best_part_thresh:.2f}")

    final_df = agg_participant_preds(df, prob_thresh=best_part_thresh)

    final_metrics = compute_participant_metrics(final_df, "pred_mean")

    print("\n--- FINAL PARTICIPANT RESULTS ---")
    print(final_metrics)

    # ROC CURVE
    y_true = final_df["true_label"].values
    y_scores = final_df["mean_prob"].values

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    print("Participant AUC:", auc_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title(name)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.show()

    return final_metrics

if __name__ == "__main__":

    run_logreg_experiment(
        "TFIDF_clean",
        "data/features/clean"
    )

