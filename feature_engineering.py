# %% import
import re
from typing import Iterable

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns 
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat as ts

plt.style.use("ggplot")


# %% load
df = pd.read_csv("data/train-checkpoint2.csv", index_col=False)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.head()

# %% Text length

sns.boxplot(df["text"].apply(len))

sns.histplot(df[df["real"] == 0]["text"].apply(len), bins = range(0, 20000, 250)) # Distribution for fake news
sns.histplot(df[df["real"] == 1]["text"].apply(len), bins = range(0, 20000, 250)) # Distribution for real news

df["text_len"] = df["text"].apply(len)
df[df["text_len"] < 100]
np.sum(df[df["text_len"] < 100]["real"] == 1)

# Drop values that are too short for proper prediction
df = df.drop(df[df["text_len"] < 100].index)

df = df.astype({
    "text": str,
    "real": np.float32,
    "text_len": np.float32
})

per_corr = df[["text_len", "real"]].corr(method='pearson')
sns.heatmap(per_corr)
per_corr

spear_corr = df[["text_len", "real"]].corr(method='spearman')
sns.heatmap(spear_corr)
spear_corr

# Little to no correlation between them

# %% Words count (computations)

def text_splitter(text: str) -> Iterable[str]:
  text = text.lower()
  # Replace non words with space
  text = re.sub(r"\W", " ", text)
  # Replace additional spaces with single space
  text = re.sub(r"[\s\n]+", " ", text)
  return nltk.word_tokenize(text.strip())

df["split_text"] = df["text"].apply(text_splitter)
df["split_text_len"] = df["split_text"].apply(len)


# %% Words count (analysis)

df[["split_text_len", "real"]].corr(method="pearson")
df[["split_text_len", "real"]].corr(method="spearman")

# Little to now correlations sadly

# %% Remove unnecesarry columns

df = df[["text", "real"]]
df = df.reset_index()

# %% Normalize and tokenize text
punc_regex = re.compile('[%s]' % re.escape(string.punctuation))
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_links(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s\n]+", " ", text)
    return re.sub("https?://\S+|www\.\S+", "", text)

def normalize_text(text: str) -> Iterable[str]:
    text = text.lower()
    # Get rid of urls
    text = re.sub("https?://\S+|www\.\S+", "", text)
    # Get rid of non words and extra spaces
    text = re.sub(r"\W", " ", text)
    # Remove punctuation
    text = re.sub(punc_regex, "", text)
    # Contract addtional spaces
    text = re.sub(r"[\s\n]+", " ", text)
    splited = text.strip().split()
    # Remove stop words
    splited = [w for w in splited if not w in stop_words]
    # Lemmatization
    splited = [lemmatizer.lemmatize(w) for w in splited]

    return splited

df["text_norm"] = df["text"].apply(normalize_text)
df["text"] = df["text"].apply(remove_links)

# %% Remove super short sentances < 5 words
sum(df["text_norm"].apply(len) < 5)
super_short_text = df[df["text_norm"].apply(len) < 5]

# As we find out, there are all empty after removing links

df = df.drop(super_short_text.index)

# %% Mean word length

df["mean_word_len"] = df["text_norm"].apply(lambda x: sum([len(w) for w in x]) / len(x))

df[["mean_word_len", "real"]].corr(method="pearson")
df[["mean_word_len", "real"]].corr(method="spearman")

# Still not good enought, but better than last time. Might consider leaving

# %% Try with most frequent words
def most_frequent(values):
    occurence_count = Counter(values)
    return [t[0] for t in occurence_count.most_common(10)]


df["most_freq"] = df["text_norm"].apply(most_frequent)

df["mean_freq_len"] = df["most_freq"].apply(lambda x: sum([len(w) for w in x]) / len(x))

df[["mean_freq_len", "real"]].corr(method="pearson")
df[["mean_freq_len", "real"]].corr(method="spearman")

# It decreased the predictiblity

# %% Remove frequent counts columns
df = df[["text", "text_norm", "mean_word_len", "real"]]

# %% Count the number of number in the text

number_regex = re.compile("r\d+")

df["number_count"] = df["text_norm"].apply(lambda x: sum(len(number_regex.findall(s)) for s in x))

df[["number_count", "real"]].corr(method="pearson")
df[["number_count", "real"]].corr(method="spearman")

# Another failed pick

# %% Sentiment Analysis
sia = SentimentIntensityAnalyzer()

df["sentiment_score"] = df["text_norm"].apply(lambda x: sia.polarity_scores(" ".join(x))["compound"])

df[["sentiment_score", "real"]].corr(method="pearson")
df[["sentiment_score", "real"]].corr(method="spearman")

# As we find out not really good. Surprisingly

# %% Remove sentiment_score
df = df[["text", "text_norm", "mean_word_len", "real"]]

# %% Text statistics 

statistics = [
    ("flesch_reading_ease", ts.flesch_reading_ease),
    ("flesch_kincaid_grade", ts.flesch_kincaid_grade),
    ("smog_index", ts.smog_index),
    ("gulpease_index", ts.gulpease_index),
    ("text_standard", lambda x: ts.text_standard(x, float_output=True)),
    ("spache_readability", ts.spache_readability),
    ("reading_time", ts.reading_time)
]

for stat, func in statistics:
    print(stat)
    applied = df["text"].apply(func)
    per_corr = applied.corr(df["real"], method="pearson")
    print(f"    per_corr: {per_corr}")
    spear_corr = applied.corr(df["real"], method="spearman")
    print(f"    spear_corr: {spear_corr}")
    
    
# Outputs
"""
flesch_reading_ease
    per_corr: -0.19842374253451905
    spear_corr: -0.25833853418826486
flesch_kincaid_grade
    per_corr: 0.1588879332116099
    spear_corr: 0.22516284594092845
smog_index
    per_corr: 0.1633218740137106
    spear_corr: 0.24201689516640493
gulpease_index
    per_corr: -0.17426772320356124
    spear_corr: -0.2558078195105583
text_standard
    per_corr: 0.17970700592278932
    spear_corr: 0.2430292723702297
spache_readability
    per_corr: 0.16591633223029942
    spear_corr: 0.23676406465356153
"""

# %% Adding text statistics 

# Considering the results from the above I would add to my dataframe:
#    - flesch_reading_ease
#    - gulpease_index
#    - text_standard

df["flesch_reading_ease"] = df["text"].apply(ts.flesch_reading_ease)
df["gulpease_index"] = df["text"].apply(ts.gulpease_index)
df["text_standard"] = df["text"].apply(lambda x: ts.text_standard(x, float_output=True))
df["smog_index"] = df["text"].apply(ts.smog_index)


# Also I will be keeping the mean_word_len

# %% Check correlation between those statistics and pick good onces

df_stats = df[["mean_word_len", "gulpease_index", "smog_index", "real"]]

per_corr_matrix = df_stats.corr(method="pearson")

sns.heatmap(per_corr_matrix, 
            xticklabels=per_corr_matrix.columns,
            yticklabels=per_corr_matrix.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5
            )


spear_corr_matrix = df_stats.corr(method="spearman")

sns.heatmap(spear_corr_matrix,
            xticklabels=spear_corr_matrix.columns,
            yticklabels=spear_corr_matrix.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5
            )

# I will be keeping the gulpease_index, smog_index and the mean_word_len
# %% Pick columns for final df
df = df[["text", "text_norm", "gulpease_index", "smog_index", "mean_word_len", "real"]]

# %% Explore outliner in gulpease_index
sns.boxplot(df, y="gulpease_index", x="real")

fre_mean, fre_std = np.mean(df["gulpease_index"]), np.std(df["gulpease_index"])

print(fre_mean, fre_std)

weird = df[df["gulpease_index"] < -75]
weird

# Those records contain to much noise and some js code. I will remove them
df = df.drop(weird.index)

# %% Explore outliner in smog_index
sns.boxplot(df, y="smog_index", x="real")

fre_mean, fre_std = np.mean(df["smog_index"]), np.std(df["smog_index"])

print(fre_mean, fre_std)

weird = df[df["smog_index"] > 27]
weird
# Those records provide actual information so we should keep them

# %% Explore outliner in mean_word_len
sns.boxplot(df, y="mean_word_len", x="real")

fre_mean, fre_std = np.mean(df["mean_word_len"]), np.std(df["mean_word_len"])

print(fre_mean, fre_std)

weird = df[df["mean_word_len"] > 10]
weird

# Those records contain raw code or raw links with images. I will remove them
df = df.drop(weird.index)

# %% Last check for the correlation matrix

df_stats = df[["mean_word_len", "gulpease_index", "smog_index", "real"]]

per_corr_matrix = df_stats.corr(method="pearson")

sns.heatmap(per_corr_matrix, 
            xticklabels=per_corr_matrix.columns,
            yticklabels=per_corr_matrix.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5
            )


spear_corr_matrix = df_stats.corr(method="spearman")

sns.heatmap(spear_corr_matrix,
            xticklabels=spear_corr_matrix.columns,
            yticklabels=spear_corr_matrix.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5
            )

# %% Checkpoint 3

df.to_csv("train-checkpoint3.csv")
