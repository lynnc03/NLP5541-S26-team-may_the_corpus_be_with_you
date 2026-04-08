#transformerT.py

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
import re

#my scripts
from load_data import LoadData
from split_by_pid import SplitByPID
from transformerB import TransformerBuilder


def compute_scores(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  accuracy = accuracy_score(labels, predictions)
  precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

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

  builder = TransformerBuilder(model_name='distilbert-base-uncased', text_col = "utterance_clean", label_col="label_binary", feature_cols=feature_cols,
                               train_df = train_df, val_df = val_df, test_df = test_df, max_len=128, num_labels=2)
  
  train_dataset, val_dataset, test_dataset = builder.get_datasets()
  model = builder.model


  ##things to change and iterate on
  training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/NLP_Project_Transformer_Tuned/checkpoints",
    eval_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_steps = 5000,
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    learning_rate=1e-5,
    weight_decay = 0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    ##added for run 3
    warmup_steps=2700,
    report_to="none")

  trainer=Trainer(model=model, args = training_args, train_dataset = train_dataset, eval_dataset=val_dataset, compute_metrics = compute_scores)
  trainer.train()

  #then final evaluate

  val_metrics = trainer.evaluate(eval_dataset=val_dataset)
  print(f'Validation results: {val_metrics}')

  test_metrics = trainer.evaluate(eval_dataset=test_dataset)
  print(f'Test results: {test_metrics}')



if __name__ == "__main__":
  main()

