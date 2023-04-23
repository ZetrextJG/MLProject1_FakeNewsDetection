import logging
import re
import string
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import nltk
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from textstat import textstat as ts

nltk.download("stopwords")
nltk.download("wordnet")

punc_regex = re.compile("[%s]" % re.escape(string.punctuation))
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def remove_links(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s\n]+", " ", text)
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def normalize_text(text: str) -> Iterable[str]:
    text = text.lower()
    # Get rid of urls
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Get rid of non words and extra spaces
    text = re.sub(r"\W", " ", text)
    # Remove punctuation
    text = re.sub(punc_regex, "", text)
    # Contract addtional spaces
    text = re.sub(r"[\s\n]+", " ", text)
    splited = text.strip().split()
    # Remove stop words
    splited = [w for w in splited if w not in stop_words]
    # Lemmatization
    splited = [lemmatizer.lemmatize(w) for w in splited]

    return splited


def create_df(texts: Iterable[str]) -> pd.DataFrame:
    text_norms = [normalize_text(text) for text in texts]
    text_cleans = [remove_links(text) for text in texts]

    df = pd.DataFrame(
        {
            "text_norm": text_norms,
            "text": text_cleans,
        }
    )

    df["mean_word_len"] = df["text_norm"].apply(
        lambda x: sum([len(w) for w in x]) / len(x)
    )

    df["gulpease_index"] = df["text"].apply(ts.gulpease_index)
    df["smog_index"] = df["text"].apply(ts.smog_index)

    return df


MAX_VOCAB = 20000
MAX_LEN = 256


class ModelInputCreator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_vocab: int,
        max_len: int,
        texts_column: str,
        numerical_columns: List[str],
    ):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.texts_column = texts_column
        self.numerical_columns = numerical_columns
        self.scaler = StandardScaler()
        self.tokenizer = Tokenizer(num_words=MAX_VOCAB)

    def fit(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        self.tokenizer.fit_on_texts(df[self.texts_column])
        self.scaler.fit(df[self.numerical_columns])
        return self

    def load(self):
        path = Path(__file__).parent / "model"
        tokenizer_path = path / "tokenizer.save"
        scaler_path = path / "scaler.save"
        self.tokenizer = joblib.load(tokenizer_path)
        self.scaler = joblib.load(scaler_path)
        print(self.scaler.var_)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple:
        df = df.reset_index(drop=True)
        text_sequences = self.tokenizer.texts_to_sequences(df[self.texts_column])
        text_sequences = pad_sequences(text_sequences, padding="post", maxlen=MAX_LEN)
        numerical = self.scaler.transform(df[self.numerical_columns])
        return (text_sequences, numerical)

    def fit_transform(self, df: pd.DataFrame) -> Tuple:
        df = df.reset_index(drop=True)
        self.fit(df)
        return self.transform(df)


def prepare_input(input_creator: ModelInputCreator, texts: Iterable[str]) -> Tuple:
    input_df = create_df(texts)
    transformed = input_creator.transform(input_df)
    return transformed


def prepare_input_from_df(input_creator: ModelInputCreator, df: pd.DataFrame) -> Tuple:
    transformed = input_creator.transform(df)
    return transformed


print("Processing input")
df_test = pd.read_csv("data/train-checkpoint3.csv")
df_test = df_test.iloc[:1000]
df_test.drop(columns=["Unnamed: 0"], inplace=True)
df_test.drop(columns=["text"], inplace=True)

df_test_X = df_test.drop(columns=["real"])
df_test_y = df_test.real

input_creator = ModelInputCreator(
    max_vocab=MAX_VOCAB,
    max_len=MAX_LEN,
    texts_column="text_norm",
    numerical_columns=["gulpease_index", "smog_index", "mean_word_len"],
)
input_creator.load()
to_pred = prepare_input_from_df(input_creator, df_test_X)
print(to_pred)

print("Loading from file")
model_path = Path(__file__).parent / "model" / "best_model.h5"
model = tf.keras.models.load_model(model_path)
if model is None:
    raise Exception("Model is None")

model.summary()

print("Predicting..")
pred = model.predict(to_pred)
roc05 = ((pred > 0.5) * 1).reshape(-1)
y_test = df_test_y.to_numpy()

acc = sum(roc05 == y_test) / len(y_test)
print(f"Accuracy on possibly seen data: {acc} ")
