##with import statements, call this split_by_pid.py

from sklearn.model_selection import train_test_split
import pandas as pd

class SplitByPID:
  def __init__(self, data: pd.DataFrame, pid_col: str, text_col: str, label_col: str, random_state: int = 55) -> None:
    self.data = data.copy()
    self.pid_col = pid_col
    self.text_col = text_col
    self.label_col = label_col
    self.random_state = random_state


  def check_pid_label_counts(self) -> pd.Series:
    """
    checks if each pid has exactly 1 label
    """
    pid_label_counts = self.data.groupby(self.pid_col)[self.label_col].nunique()
    return pid_label_counts

  def build_df_by_pid(self) -> pd.DataFrame:
    """
    one row per pid with its label and text.
    """
    pid_label_counts = self.check_pid_label_counts()
    inconsistent = pid_label_counts[pid_label_counts > 1]
    #are there multiple labels? is bad, needs to be either disordered or not disordered
    if len(inconsistent) > 0:
      raise ValueError("some PID have multiple labels")
  
    pid_df = self.data[[self.pid_col, self.label_col, self.text_col]].copy()
    return pid_df
  
  def split_mapping(self, test_size:float, val_size:float) -> pd.DataFrame:
    """
    splits DF by pid, to avoid leakage. stratifies by label."""
    if test_size + val_size >= 1:
      raise ValueError("test_size + val_size doesn't work. needs to be less than 1")
    
    df = self.build_df_by_pid()

    #old version of split mapping didn't have this line. doesn't permanently drop, just dorps duplicates for mapping
    pid_df = df[[self.pid_col, self.label_col]].drop_duplicates(subset=self.pid_col)

    #make sure we are stratifiying by label
    train_df, temp_df = train_test_split(pid_df, test_size = (test_size + val_size),
      stratify = pid_df[self.label_col], random_state = self.random_state)
    
    val_df, test_df = train_test_split(temp_df, test_size = test_size/(test_size + val_size),
      stratify = temp_df[self.label_col], random_state = self.random_state)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return split_df[[self.pid_col, "split"]]


  def split_data(self, test_size:float, val_size:float) -> pd.DataFrame:
    """
    split column added to full df, merges pid splits back in.
    """
    split_map = self.split_mapping(test_size=test_size, val_size=val_size)

    df_with_splits = self.data.merge(
        split_map[[self.pid_col, "split"]], on=self.pid_col, how="left")

    return df_with_splits

  def get_dfs(self, test_size: float, val_size: float, keep_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    once pid-level splitting, gets train, val, test dfs.
    """
    df = self.split_data(test_size=test_size, val_size=val_size)

    if keep_cols is not None:
      df = df[keep_cols + ["split"]].copy()

    train_df = df[df["split"] == "train"].drop(columns=["split"]).copy()
    val_df = df[df["split"] == "val"].drop(columns=["split"]).copy()
    test_df = df[df["split"] == "test"].drop(columns=["split"]).copy()

    return train_df, val_df, test_df


  def split_manifest(self, test_size: float, val_size: float, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """
    creates a pid-level split manifest with pid, label, split, and optional extra columns.
    """
    df = self.split_data(test_size=test_size, val_size=val_size)

    manifest_cols = [self.pid_col, self.label_col, "split"]

    if extra_cols is not None:
      manifest_cols.extend(extra_cols)

    manifest = df[manifest_cols].copy()
    return manifest