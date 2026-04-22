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

  test_labels, test_probs = get_pos_class_probs(trainer, test_dataset)
  test_metrics_thresholded = compute_threshold_metrics(test_labels, test_probs, best_threshold)
  print(f'Test results at best validation threshold {best_threshold:.2f}: {test_metrics_thresholded}')

  test_metrics = trainer.evaluate(eval_dataset=test_dataset)
  print(f'Test results at default threshold 0.5: {test_metrics}')


if __name__ == "__main__":
  main()

