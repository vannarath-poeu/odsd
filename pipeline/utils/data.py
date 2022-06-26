import pandas as pd

from utils.text import (
  clean_text,
  remove_stopwords,
  stemm_text,
)

def load_data(file_path: str):
  df = pd.read_csv(file_path, encoding="latin-1")
  df = df.dropna(how="any", axis=1)
  df = df.rename(columns={ "v1": "target", "v2": "message" })

def process_df(df):
  df['processed'] = df['message'].apply(clean_text)
  df['processed'] = df['processed'].apply(remove_stopwords)
  df['processed'] = df['processed'].apply(stemm_text)
  return df

def encode_target(df):
  df['target_encoded'] = df['target'].apply(lambda x: 1 if x == "spam" else 0)