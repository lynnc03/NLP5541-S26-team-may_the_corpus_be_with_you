##load_data.py
import pandas as pd

class LoadData:
  def __init__(self, data_path: str, text_col: str, label_col: str, required_cols: list[str] | None = None) -> None:
    self.data_path = data_path
    self.text_col = text_col
    self.label_col = label_col
    self.required_cols = required_cols or [text_col, label_col]

  def load_data(self) -> pd.DataFrame:
    #or other format if in other format
    df = pd.read_csv(self.data_path)

    #do required columns match what is in DF?
    missing = [col for col in self.required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing these columns {missing}")

    df[self.text_col] = df[self.text_col].astype(str)
    df[self.label_col] = pd.to_numeric(df[self.label_col], errors="raise").astype(int)

    return df





