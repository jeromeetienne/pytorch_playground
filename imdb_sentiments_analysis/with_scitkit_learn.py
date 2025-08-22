# from https://www.kaggle.com/code/avnika22/imdb-perform-sentiment-analysis-with-scikit-learn/notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()

# TODO use __dirname__
import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))


########################################################################
# Read the data
#

dataset_filename = os.path.join(
    __dirname__, "./input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv"
)
data = pd.read_csv(dataset_filename)
print(data.head())

########################################################################
# Downsample the dataset
#

data = data.sample(frac=0.2, random_state=42)

########################################################################
# Display the distribution of sentiments
#

display_enabled = False  # Set to True to display the distribution of sentiments
if display_enabled:
    plt.figure(figsize=(10, 5))
    sns.countplot(x="label", data=data, palette="Set2")
    plt.title("Distribution of Sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["Negative", "Positive"])
    plt.show()


#############################################################
# Bag of Words
#

if False:
    df = [
        "Hey Jude, refrain Dont carry the world upon your shoulders For well you know that its a fool Who plays it cool By making his world a little colder Na-na-na,a, na Na-na-na, na"
    ]
    bag = count.fit_transform(df)
    print(f"bag feature names: {count.get_feature_names_out()}")

    print(f"bag array: {bag.toarray()}")

##############################################################

import re


def preprocessor(text):
    """
    Text preprocessor for cleaning and normalizing text data. e.g. <a> and emojis
    """
    text = re.sub("<[^>]*>", "", text)
    emojis = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emojis).replace("-", "")
    return text


if False:
    print(f"======================================================")
    text_to_preprocess1 = data.loc[0, "text"][-50:]
    preprocessed_text2 = preprocessor(text_to_preprocess1)
    print(f"original text 1: {text_to_preprocess1}")
    print(f"preprocessed text 1: {preprocessed_text2}")

    test_to_preprocess2 = "<a> this is :(  aweomee wohhhh :)"
    preprocessed_text2 = preprocessor(test_to_preprocess2)
    print(f"original text 2: {test_to_preprocess2}")
    print(f"preprocessed text 2: {preprocessed_text2}")

# Apply preprocessor to the text column
data["text"] = data["text"].apply(preprocessor)

####################################################################

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


if True:
    print(f"======================================================")
    text_to_tokenize1 = "Haters love Hating as they Hate"
    print(f"original text 1: {text_to_tokenize1}")
    print(f"tokenized text 1: {tokenizer(text_to_tokenize1)}")
    print(f"stemmed text 1: {tokenizer_porter(text_to_tokenize1)}")

######################################################
# import nltk

# nltk.download("stopwords")

from nltk.corpus import stopwords

stop = stopwords.words("english")

#####################################################

from wordcloud import WordCloud

positive_data = data[data["label"] == 1]
positive_data = positive_data["text"]
negative_data = data[data["label"] == 0]
negative_data = negative_data["text"]


def wordcloud_draw(data, color="white"):
    words = " ".join(data)
    cleaned_word = " ".join(
        [word for word in words.split() if (word != "movie" and word != "film")]
    )
    wordcloud = WordCloud(
        stopwords=stop, background_color=color, width=2500, height=2000
    ).generate(cleaned_word)
    plt.figure(1, figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


if False:
    print("Positive words are as follows")
    wordcloud_draw(positive_data, "white")
    print("Negative words are as follows")
    wordcloud_draw(negative_data, "black")


################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    strip_accents=None,
    lowercase=False,
    preprocessor=None,
    tokenizer=tokenizer_porter,
    use_idf=True,
    norm="l2",
    smooth_idf=True,
)

y = data.label.values
x = tfidf.fit_transform(data.text)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size=0.5, shuffle=False
)


######################################################################
from sklearn.linear_model import LogisticRegressionCV

classifier_logistic_regression = LogisticRegressionCV(
    cv=6, scoring="accuracy", random_state=0, n_jobs=-1, verbose=3, max_iter=500
)
classifier_logistic_regression.fit(x_train, y_train)

y_pred = classifier_logistic_regression.predict(x_test)


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("LogisticRegressionCV Accuracy:", metrics.accuracy_score(y_test, y_pred))


#####################################################

from sklearn.linear_model import SGDClassifier

classifier_sgd = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=0.0001,
    l1_ratio=0.15,
    fit_intercept=True,
    max_iter=1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    n_jobs=None,
    random_state=None,
    learning_rate="optimal",
    eta0=0.0,
    power_t=0.5,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    class_weight=None,
    warm_start=False,
    average=False,
)
classifier_sgd.fit(x_train, y_train)

y_pred = classifier_sgd.predict(x_test)

print("SGDClassifier Accuracy:", metrics.accuracy_score(y_test, y_pred))
