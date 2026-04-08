#transformerB.py
import pandas as pd
import torch.nn as nn
import torch
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class DistilBertVersion(nn.Module):
  def __init__(self, model_name: str, num_features: int, num_classes: int, dropout: float) -> None:
    super().__init__()
  
    self.bert = AutoModel.from_pretrained(model_name)
    hidden_size = self.bert.config.hidden_size
    self.dropout = nn.Dropout(dropout)
    #linear to classifier head
    self.classifier = nn.Sequential(
      nn.Linear(hidden_size + num_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
    self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

  ##forward pass
  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: torch.Tensor, labels: torch.Tensor | None = None ) -> SequenceClassifierOutput:
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

    embed = outputs.last_hidden_state[:, 0, :]
    features = features.float()
    combined = torch.cat([embed, features], dim=1)
    #apply dropout
    combined = self.dropout(combined)
    #apply classifier
    logits = self.classifier(combined)
 
    loss = None
    if labels is not None:
      loss = self.loss_fn(logits, labels.long())

    return SequenceClassifierOutput(loss=loss, logits=logits)

class TransformerBuilder:
  def __init__(self, model_name: str, text_col: str, label_col: str, feature_cols: list[str], train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, max_len: int, num_labels: int,
               device: str | torch.device | None = None) -> None:
    self.model_name = model_name
    self.text_col = text_col
    self.label_col = label_col
    self.feature_cols = feature_cols
    self.train_df = train_df.copy()
    self.val_df = val_df.copy()
    self.test_df = test_df.copy()
    self.max_len = max_len
    self.num_labels = num_labels

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = torch.device(device)

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.scaler = StandardScaler()
    self._scaler_fitted = False

    self.model = DistilBertVersion(self.model_name, len(self.feature_cols), self.num_labels, dropout=0.2)
    self.model.to(self.device)

  def fit_scaler(self) -> None:
    train_features = self.train_df[self.feature_cols].astype(float)
    self.scaler.fit(train_features)
    self._scaler_fitted = True

  def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
    if not self._scaler_fitted:
        raise ValueError("Fit scaler before transforming df")

    transformed_df = df.copy()

    transformed_df[self.text_col] = transformed_df[self.text_col].astype(str)
    transformed_df[self.label_col] = transformed_df[self.label_col].astype(int)

    scaled_features = self.scaler.transform(transformed_df[self.feature_cols].astype(float))
    df_with_features = pd.DataFrame(
        scaled_features,
        columns=self.feature_cols,
        index=transformed_df.index,
    )

    transformed_df[self.feature_cols] = df_with_features
    return transformed_df

  def df_to_dataset(self, df: pd.DataFrame) -> Dataset:
    df = self.transform_df(df)

    df_for_huggingface = pd.DataFrame({"text": df[self.text_col].tolist(), "labels": df[self.label_col].tolist(), "features": df[self.feature_cols].values.tolist(),})

    dataset = Dataset.from_pandas(df_for_huggingface, preserve_index=False)

    def tokenize_batch(batch: dict) -> dict:
      #note: do we want to truncate?
        tokenized = self.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=self.max_len)
        tokenized["features"] = batch["features"]
        tokenized["labels"] = batch["labels"]
        return tokenized

    #yes for batching?
    dataset = dataset.map(tokenize_batch, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "features", "labels"])
    return dataset
  
  def get_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
    self.fit_scaler()
    train_dataset = self.df_to_dataset(self.train_df)
    val_dataset = self.df_to_dataset(self.val_df)
    test_dataset = self.df_to_dataset(self.test_df)
    return train_dataset, val_dataset, test_dataset
