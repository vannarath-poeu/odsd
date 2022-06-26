from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component

import google.cloud.aiplatform as aip

@component(
    output_component_file="train_model.yaml",
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage", "pandas", "sklearn", "fsspec", "gcsfs", "nltk", "tensorflow", "tensorflowjs"]
)
def train_model(
  data_path: str,
  bucket: str,
  save_path: str,
  model_name: str,
  glove_embedding_file: str,
  output_bucket: str,
) -> str:
  from google.cloud import storage

  import pandas as pd
  import numpy as np
  import re
  import string
  import nltk
  from sklearn.model_selection import train_test_split
  import tensorflow as tf

  import subprocess
  import os

  from nltk.corpus import stopwords

  def nltk_init():
    nltk.download('stopwords')
    nltk.download('punkt')

  def remove_stopwords(text):
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'im', 'c']
    stop_words = stop_words + more_stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

  def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

  def stemm_text(text):
    stemmer = nltk.SnowballStemmer("english")
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

  def upload_file_to_bucket(bucket_name: str, source: str, destination: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_filename(source)
  
  def read_file_from_bucket(bucket_name: str, file_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.download_as_string()

  def process_df(df):
    df['processed'] = df['message'].apply(clean_text)
    df['processed'] = df['processed'].apply(remove_stopwords)
    df['processed'] = df['processed'].apply(stemm_text)
    return df
  
  def load_data(file_path: str):
    df = pd.read_csv(file_path, encoding="latin-1")
    df = df.dropna(how="any", axis=1)
    df = df.rename(columns={ "v1": "target", "v2": "message" })
    return df
  
  def encode_target(df):
    df['target_encoded'] = df['target'].apply(lambda x: 1 if x == "spam" else 0)
    return df
  
  bucket_name = bucket.split('//')[-1] # Only use bucket name, without gs://
  output_bucket_name = output_bucket.split('//')[-1] # Only use bucket name, without gs://
  gcs_path = f"{bucket}/{save_path}/{model_name}"

  nltk_init()

  df = load_data(data_path)
  df = process_df(df)
  df = encode_target(df)

  texts = df['processed']
  target = df['target_encoded']

  X_train, X_test, y_train, y_test = train_test_split(
    texts, 
    target, 
    test_size=0.1,
    random_state=42,
  )

  word_tokenizer = tf.keras.preprocessing.text.Tokenizer()
  word_tokenizer.fit_on_texts(X_train)

  vocab_length = len(word_tokenizer.word_index) + 1
  max_length = 80
  metadata = {}
  metadata["vocabulary_size"] = vocab_length
  metadata["max_length"] = max_length

  embeddings_dictionary = dict()
  embedding_dim = 100

  # Load GloVe 100D embeddings
  embedding_txt = read_file_from_bucket(bucket_name, glove_embedding_file)
  for line in embedding_txt.splitlines():
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions

  embedding_matrix = np.zeros((vocab_length, embedding_dim))

  for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
      embedding_matrix[index] = embedding_vector
  
  def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

  train_padded_sentences = tf.keras.preprocessing.sequence.pad_sequences(
    embed(X_train), 
    max_length, 
    padding='post'
  )

  import json
  word_index = {}
  for word, index in word_tokenizer.word_index.items():
    word_index[word] = index

  metadata["word_index"] = word_index

  with open("metadata.json", "w") as f:
    json.dump(metadata, f)
  
  def glove_lstm(max_length):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((max_length,)))
    model.add(tf.keras.layers.Embedding(
      input_dim=vocab_length,
      output_dim=embedding_dim,
      input_length=max_length,
      weights = [embedding_matrix],
      trainable=False
    ))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
      max_length, 
      return_sequences = True, 
      recurrent_dropout=0.2
    )))
    model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(max_length, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(max_length, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

  model = glove_lstm(max_length)
  test_padded_sentences = tf.keras.preprocessing.sequence.pad_sequences(
    embed(X_test), 
    max_length, 
    padding='post'
  )
  model.fit(
    train_padded_sentences, 
    y_train, 
    epochs = 7,
    batch_size = 32,
    validation_data = (test_padded_sentences, y_test),
    verbose = 1,
  )

  if not os.path.exists('lstm_glove'):
    os.mkdir('lstm_glove')

  if not os.path.exists('tfjs_lstm_glove'):
    os.mkdir('tfjs_lstm_glove')

  model.save('lstm_glove')
  os.listdir('lstm_glove')

  subprocess.run(["ls", "-l"])

  command = "tensorflowjs_converter --input_format=tf_saved_model --output_format tfjs_graph_model --control_flow_v2=true ./lstm_glove ./tfjs_lstm_glove".split(" ")
  subprocess.run(command)

  upload_file_to_bucket(output_bucket_name, "metadata.json", f"{save_path}/{model_name}/metadata.json")
  upload_file_to_bucket(output_bucket_name, "tfjs_lstm_glove/model.json", f"{save_path}/{model_name}/model.json")
  upload_file_to_bucket(output_bucket_name, "tfjs_lstm_glove/group1-shard1of1.bin", f"{save_path}/{model_name}/group1-shard1of1.bin")
  return gcs_path

def main():
  Project_ID = "odsd-354513"
  GCS_Bucket = "gs://odsd"
  Output_GCS_Bucket = "gs://odsd-public"
  Region = "asia-southeast1"

  PIPELINE_ROOT = "{}/pipeline_root/odsd".format(GCS_Bucket)

  @dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="odsd-pipeline",
  )
  def odsd_pipeline():
    train_model_task = train_model(
      data_path=f"{GCS_Bucket}/data/spam.csv",
      glove_embedding_file="data/glove.6B.100d.txt",
      bucket=GCS_Bucket,
      output_bucket=Output_GCS_Bucket,
      save_path="model",
      model_name="odsd",
    )

  compiler.Compiler().compile(
    pipeline_func=odsd_pipeline,
    package_path="odsd_pipeline.json",
  )

  aip.init(project=Project_ID, location=Region)

  job = aip.PipelineJob(
    enable_caching=False,
    display_name="odsd_pipeline",
    template_path="odsd_pipeline.json",
    location=Region,
  )

  job.submit()

if __name__ == "__main__":
  main()