#transformerT.py

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
import re
from scipy.special import softmax

#my scripts
from load_data import LoadData
from split_by_pid import SplitByPID
from transformerB import TransformerBuilder

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



def compute_scores(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  accuracy = accuracy_score(labels, predictions)

  #changed to per class, not just SLI
  precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(labels, predictions, labels =[0,1], average=None, zero_division=0)

  _, _, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
  return {"accuracy": accuracy, 
          "precision_control_class": precision_per_class[0], 
          "recall_control_class": recall_per_class[0], 
          "f1_control_class": f1_per_class[0], 
          "support_control_class": int(support_per_class[0]),
          "precision_sli_class": precision_per_class[1],
          "recall_sli_class": recall_per_class[1],
          "f1_sli_class": f1_per_class[1],
          "support_sli_class": int(support_per_class[1]),
          "f1_macro": f1_macro}

##now add metrics
def compute_threshold_metrics(labels, probs, threshold):
  predictions = (probs >= threshold).astype(int)
  accuracy = accuracy_score(labels, predictions)
  precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(labels, predictions, labels =[0,1], average=None, zero_division=0)
  _, _, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
   
  return{"threshold": float(threshold),
    "accuracy": float(accuracy),
    "precision_control_class": float(precision_per_class[0]),
    "recall_control_class": float(recall_per_class[0]),
    "f1_control_class": float(f1_per_class[0]),
    "support_control_class": int(support_per_class[0]),
    "precision_sli_class": float(precision_per_class[1]),
    "recall_sli_class": float(recall_per_class[1]),
    "f1_sli_class": float(f1_per_class[1]),
    "support_sli_class": int(support_per_class[1]),
    "f1_macro": float(f1_macro)}

def threshold_sweep(labels, probs, thresholds = np.arange(0.05, 0.96, 0.01), metric_optimized = "f1_macro"):
  results = []
  for threshold in thresholds:
    metrics = compute_threshold_metrics(labels, probs, threshold)
    results.append(metrics)
  
  best_result = max(results, key=lambda x: x[metric_optimized])
  return best_result, results

def make_prediction_df(df, labels, probs, threshold, pid_column = "pid", label_column = "label_binary"):
   """
   attaches predictions to utterance DF
   """
   pred_df = df[[pid_column, "file_id", label_column]].copy()
   pred_df["true_label"] = labels
   pred_df["prob_sli"] = probs
   pred_df["pred_sli"] = (pred_df["prob_sli"] >= threshold).astype(int)

   return pred_df

def agg_participant_preds(utterance_pred_df, participant_prob_threshold=0.5,
                          participant_vote_threshold = 0.5, pid_column = "pid", label_column = "true_label"):
   """
   aggregates predictions so one per participant
   predicting by either average probability, or % of utterances with SLI
   """
   summary_stats = {"true_label": (label_column, "first"),
                    "n_utterances": ("prob_sli", "size"),
                    "mean_prob_sli": ("prob_sli", "mean"),
                    "median_prob_sli": ("prob_sli", "median"),
                    "max_prob_sli": ("prob_sli", "max"),
                    "n_utterances_pred_sli":("pred_sli", "sum"),
                    "prob_utterances_pred_sli": ("pred_sli", "mean")}
   participant_df = (utterance_pred_df.groupby(pid_column).agg(**summary_stats).reset_index())

   participant_df["participant_pred_mean_prob"] = (participant_df["mean_prob_sli"] >= participant_prob_threshold).astype(int)

   participant_df["participant_pred_vote"] = (participant_df["prob_utterances_pred_sli"] >= participant_vote_threshold).astype(int)

   return participant_df

def compute_participant_metrics(participant_df, pred_column, label_column = "true_label"):
   """
   participant level classification
   """
   labels = participant_df[label_column].values
   predictions = participant_df[pred_column].values

   accuracy = accuracy_score(labels, predictions)

   precision_per_class, recall_per_class, f1_per_class, support_per_class = (precision_recall_fscore_support(labels, predictions,
                                                                                                             labels = [0,1],
                                                                                                             average=None,
                                                                                                             zero_division=0))
   
   _, _, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division = 0)

   return {"prediction_column": pred_column,
           "accuracy": float(accuracy),
           "precision_control_class": float(precision_per_class[0]),
           "recall_control_class": float(recall_per_class[0]),
           "f1_control_class": float(f1_per_class[0]),
           "support_control_class": int(support_per_class[0]),
           "precision_sli_class": float(precision_per_class[1]),
           "recall_sli_class": float(recall_per_class[1]),
           "f1_sli_class": float(f1_per_class[1]),
           "support_sli_class": int(support_per_class[1]),
           "f1_macro": float(f1_macro)}

def participant_threshold_sweep(utterance_pred_df, thresholds=np.arange(0.05, 0.96,0.01),
                                metric_optimized="f1_macro", method = "mean_prob"):
  results = []
  for thresh in thresholds:
      if method == "mean_prob":
         participant_df = agg_participant_preds(utterance_pred_df, participant_prob_threshold=thresh,
                                                participant_vote_threshold=0.5)
         pred_column = "participant_pred_mean_prob"
      elif method == "vote":
         participant_df = agg_participant_preds(utterance_pred_df,
                                                participant_prob_threshold = 0.5,
                                                participant_vote_threshold = thresh)
         pred_column = "participant_pred_vote"
      else:
         raise ValueError("method not correctly chosen")
      
      metrics = compute_participant_metrics(participant_df, pred_column = pred_column)

      metrics["participant_threshold"] = thresh
      metrics["method"] = method

      results.append(metrics)
    
  best_result = max(results, key=lambda x: x[metric_optimized])
  return best_result, results
   


def get_pos_class_probs(trainer, dataset):
   predictions_output = trainer.predict(dataset)
   logits = predictions_output.predictions
   labels = predictions_output.label_ids
   probs = softmax(logits, axis=-1)[:, 1]
   return labels, probs



#teensy bit messy to have this here, but it helps with age as a year;month format. later can put into load or split files.
def parse_age(age_val):
    age_str = str(age_val).strip()
    match = re.match(r"^\s*(\d+)\s*;\s*(\d+)\.?\s*$", age_str)
    if match: 
        years = int(match.group(1))
        months = int(match.group(2))
        age = years+months/12
        return age
    try:
        return float(age_str)
    except ValueError:
        return np.nan

def main():

  feature_cols = [
    "pause_short",
    "pause_medium",
    "pause_long",
    "pause_timed_count",
    "pause_timed_total_sec",
    "pause_total",
    "filled_pause_count",
    "repetition_count",
    "revision_count",
    "reformulation_count",
    "trailing_off_count",
    "interrupted_count",
    "disfluency_total",
    "uncertain_count",
    "unintelligible_xxx",
    "unintelligible_yyy",
    "unintelligible_total",
    "paralinguistic_count",
    "age"]

  required_cols = ["pid", "file_id", "utterance_clean", "label_binary"] + feature_cols

  data_loader = LoadData(data_path="/content/drive/MyDrive/NLP_Project_Processed_data/child_utterances.csv", text_col="utterance_clean", label_col="label_binary", required_cols=required_cols)
  main_df = data_loader.load_data()
  main_df["age"] = main_df["age"].apply(parse_age)
  #deal with missingness in age, just create indicator var
  main_df["age_missing"] = main_df["age"].isna().astype(int)
  main_df["age"] = main_df["age"].fillna(main_df["age"].median())


  splitter_obj = SplitByPID(data=main_df, pid_col="pid", text_col="utterance_clean", label_col="label_binary", random_state=55)

  #does everything look good? no issues with duplicate pids now?
  pid_label_counts = splitter_obj.check_pid_label_counts()
  print(pid_label_counts.value_counts())

  #do we want file_id in final transformer? or pid?
  keep_cols = ["pid", "file_id", "utterance_clean", "label_binary"] + feature_cols

  df_with_splits = splitter_obj.split_data(test_size=0.15, val_size=0.15)
  working_df = df_with_splits[keep_cols + ["split"]].copy()
  
  train_df = working_df[working_df["split"] == "train"].drop(columns=["split"]).copy()
  val_df = working_df[working_df["split"] == "val"].drop(columns=["split"]).copy()
  test_df = working_df[working_df["split"] == "test"].drop(columns=["split"]).copy()

  #duplicates just dropped for manifest
  manifest = (df_with_splits[["pid", "label_binary", "file_id", "split"]].drop_duplicates(subset="pid").copy())
  manifest.to_csv("split_manifest_by_pid.csv", index=False)

  manifest = pd.read_csv("split_manifest_by_pid.csv")
  print(manifest["split"].value_counts())
  print(manifest.groupby(["split", "label_binary"]).size())
  print(f"Train: {len(train_df)} rows, {train_df['label_binary'].value_counts().to_dict()}")
  print(f"Val:   {len(val_df)} rows,  {val_df['label_binary'].value_counts().to_dict()}")
  print(f"Test:  {len(test_df)} rows, {test_df['label_binary'].value_counts().to_dict()}")
  print(f"Total: {len(working_df)} rows")

  builder = TransformerBuilder(model_name='distilbert-base-uncased', text_col = "utterance_clean", label_col="label_binary", feature_cols=feature_cols,
                               train_df = train_df, val_df = val_df, test_df = test_df, max_len=128, num_labels=2)
  
  train_dataset, val_dataset, test_dataset = builder.get_datasets()
  model = builder.model


  ##things to change and iterate on
  training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/NLP_Project_Transformer_Tuned/checkpoints",
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=2000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay = 0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    ##added for run 3
    warmup_steps=2700,
    report_to="none")

  trainer=Trainer(model=model, args = training_args, train_dataset = train_dataset, eval_dataset=val_dataset, compute_metrics = compute_scores)
  trainer.train() 
  #then final evaluate

  val_metrics = trainer.evaluate(eval_dataset=val_dataset)
  print(f'Validation results at default threshold 0.5 {val_metrics}')

  val_labels, val_probs = get_pos_class_probs(trainer, val_dataset)
  best_val_result, val_thresh_results = threshold_sweep(labels = val_labels, probs = val_probs, 
                                                             thresholds=np.arange(0.05, 0.96, 0.01), metric_optimized="f1_macro")
  best_threshold = best_val_result['threshold']
  print(f"Best validation threshold: {best_threshold:.2f}")
  print(f"Best threshold on validation set: {best_threshold}, with metrics: {best_val_result}")
  val_threshold_df = pd.DataFrame(val_thresh_results)
  val_threshold_df.to_csv("validation_threshold_sweep.csv", index=False)

  #utterance level test performance
  test_labels, test_probs = get_pos_class_probs(trainer, test_dataset)
  test_metrics_thresholded = compute_threshold_metrics(test_labels, test_probs, best_threshold)

  print(f'Test results at best validation utterance threshold {best_threshold:.2f}: {test_metrics_thresholded}')

  test_metrics = trainer.evaluate(eval_dataset=test_dataset)
  print(f'Test results at default utterance threshold 0.5: {test_metrics}')

  val_utterance_pred_df = make_prediction_df(df=val_df, labels=val_labels, probs=val_probs, threshold=best_threshold)

  val_utterance_pred_df.to_csv("validation_utterance_level_predictions.csv",index=False)

  test_utterance_pred_df = make_prediction_df(df=test_df, labels=test_labels, probs=test_probs, threshold=best_threshold)

  test_utterance_pred_df.to_csv("test_utterance_level_predictions.csv", index=False)

  print("Saved utterance-level prediction files.")

  test_participant_pred_df_default = agg_participant_preds(test_utterance_pred_df, participant_prob_threshold=0.5, participant_vote_threshold=0.5)

  test_participant_pred_df_default.to_csv("test_participant_level_predictions_default_thresholds.csv", index=False)

  participant_mean_metrics_default = compute_participant_metrics(test_participant_pred_df_default, pred_column="participant_pred_mean_prob")

  participant_vote_metrics_default = compute_participant_metrics(test_participant_pred_df_default, pred_column="participant_pred_vote")

  print(f"Participant-level TEST results using mean probability threshold 0.5: {participant_mean_metrics_default}")
  print(f"Participant-level TEST results using vote threshold 0.5: {participant_vote_metrics_default}")


  best_participant_mean_result, participant_mean_results = participant_threshold_sweep(utterance_pred_df = val_utterance_pred_df, thresholds=np.arange(0.05, 0.96, 0.01), metric_optimized="f1_macro", method="mean_prob")

  best_participant_vote_result, participant_vote_results = participant_threshold_sweep(utterance_pred_df=val_utterance_pred_df, thresholds=np.arange(0.05, 0.96, 0.01), metric_optimized="f1_macro",
      method="vote")

  participant_mean_threshold = best_participant_mean_result["participant_threshold"]
  participant_vote_threshold = best_participant_vote_result["participant_threshold"]

  print(f"Best validation participant threshold using mean probability: {participant_mean_threshold:.2f}")
  print(f"Best validation participant mean-prob metrics: {best_participant_mean_result}")

  print(f"Best validation participant threshold using vote: {participant_vote_threshold:.2f}")
  print(f"Best validation participant vote metrics: {best_participant_vote_result}")

  pd.DataFrame(participant_mean_results).to_csv("validation_participant_mean_prob_threshold_sweep.csv", index=False)

  pd.DataFrame(participant_vote_results).to_csv("validation_participant_vote_threshold_sweep.csv", index=False)

  test_participant_pred_df_tuned = agg_participant_preds(test_utterance_pred_df, participant_prob_threshold=participant_mean_threshold,
      participant_vote_threshold=participant_vote_threshold)

  test_participant_pred_df_tuned.to_csv("test_participant_level_predictions_tuned_thresholds.csv", index=False)

  participant_mean_metrics_tuned = compute_participant_metrics(test_participant_pred_df_tuned, pred_column="participant_pred_mean_prob")

  participant_vote_metrics_tuned = compute_participant_metrics(test_participant_pred_df_tuned, pred_column="participant_pred_vote")

  print(f"Participant-level TEST results using tuned mean-prob threshold " f"{participant_mean_threshold:.2f}: {participant_mean_metrics_tuned}")

  print(f"Participant-level TEST results using tuned vote threshold " f"{participant_vote_threshold:.2f}: {participant_vote_metrics_tuned}")


  #ROC/AUC curve
  y_true = test_participant_pred_df_tuned["true_label"].values
  #using probability for scores
  y_scores_transformer = test_participant_pred_df_tuned["mean_prob_sli"].values
  fpr_t, tpr_t, thresholds_t = roc_curve(y_true, y_scores_transformer)
  auc_t = auc(fpr_t, tpr_t)

  plt.figure(figsize=(6,6))
  plt.plot(fpr_t, tpr_t, label=f"Transformer (AUC = {auc_t:.3f})", linewidth=2)
  plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate (SLI Recall)")
  plt.title("ROC Curve (Participant-Level)")
  plt.legend()
  plt.grid(True)

  plt.savefig("roc_curve_participant_level.png", dpi=300, bbox_inches="tight")
  plt.show()

  roc_df = pd.DataFrame({
    "fpr_transformer": fpr_t,
    "tpr_transformer": tpr_t,
  })


  roc_df.to_csv("roc_transformer.csv", index=False)

  auc_df = pd.DataFrame({
      "model": ["Transformer"],
      "auc": [auc_t]
  })

  auc_df.to_csv("auc_summary.csv", index=False)

  threshold_df = pd.DataFrame({
    "threshold": thresholds_t,  # from roc_curve
    "fpr": fpr_t,
    "tpr": tpr_t
  })

  threshold_df.to_csv("roc_thresholds_transformer.csv", index=False)

if __name__ == "__main__":
  main()

