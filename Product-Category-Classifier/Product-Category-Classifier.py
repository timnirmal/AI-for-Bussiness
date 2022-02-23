# https://prakhargurawa.medium.com/creating-an-e-commerce-product-category-classifier-using-deep-learning-part-1-36431a5fbc4e
#
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import joblib

filename = 'data/products.json'
file = open(filename, encoding="utf8")
data = json.load(file)
print(data[0])

############################################# 1. Data Preparation #############################################

# creating dataframe with three rows (name, description, categories)

column_names = ['name', 'description', 'categories']
df = pd.DataFrame(columns=column_names)

names, descriptions, categories = [], [], []
for product in data:
    names.append(product['name'])
    descriptions.append(product['description'])
    productCategories = []
    for cat in product['category']:
        productCategories.append(cat['name'])
    categories.append(productCategories)

df = pd.DataFrame(list(zip(names, descriptions, categories)), columns=column_names)
print(df)

############################################# 2. Data Transformation #############################################

cat = pd.DataFrame(df['categories'].to_list())  # listing the categories seperately
pd.set_option('display.max_columns', None)
# To Reset use
# pd.reset_option('display.max_columns')
# pd.reset_option("max_columns")
print(cat.head(10))

# CHeck null values in cat
print(cat.isnull().sum())
# Check not null values in cat
print(cat.notnull().sum())

####### finding total unique categories/classes from which our prediction will belong too

category_0 = cat[0].unique()
category_1 = cat[1].unique()
category_2 = cat[2].unique()
category_3 = cat[3].unique()
category_4 = cat[4].unique()
category_5 = cat[5].unique()
category_6 = cat[6].unique()
cates = np.concatenate([category_0, category_1, category_2, category_3, category_4, category_5, category_6])
cates = list(dict.fromkeys(cates))
cates = [x for x in cates if x is not None]  # remove None
print("\nTotal Unique Cat : ", len(cates))  # number of unique classes/categories

######## unfold the dataset to view all the prediction classes at once
cat = pd.concat([cat, pd.DataFrame(columns=list(cates))])  # concatnate categories/classes to original dataframe
cat.fillna(0, inplace=True)  # fill with zero
# print(cat.head())
# print columns
print(cat.columns)

####### filling attendence for all the categories/classes

for i in range(7):
    row = 0
    for category in cat[i]:
        if category != 0:
            cat.loc[
                row, category] = 1  # loc is label-based, which means that you have to specify rows and columns based on their row and column labels.
        row = row + 1
# iloc is integer position-based, so you have to specify rows and columns by their integer position values (0-based integer position)
print(cat.head())

df2 = pd.concat([df['name'], df['description'], cat.loc[:, "2-Channel Amps":]],
                axis=1)  # creating new dataframe which contains name of product,description and categories it belong to
print(df2.head())
print(df2.shape)

############################################# 3. Data Analysis #############################################

bar_plot = pd.DataFrame()
bar_plot['category'] = df2.columns[2:]  # column name, which are categories
bar_plot['count'] = df2.iloc[:, 2:].sum().values
bar_plot.sort_values(['count'], inplace=True, ascending=False)
bar_plot.reset_index(inplace=True, drop=True)
print(bar_plot.head())  # Top 5 most occuring categories

threshold = 500  # A threshold is a value below which all those classes whose count is less than threshold will be treated as 'other' category

plt.figure(figsize=(15, 8))
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

pal = sns.color_palette("Blues_r", len(bar_plot))
rank = bar_plot['count'].argsort().argsort()

sns.barplot(bar_plot['category'][:60], bar_plot['count'][:60],
            palette=np.array(pal[::-1])[rank])  # include first few values
plt.axhline(threshold, ls='--', c='red')
plt.title("Most Common Categories", fontsize=24)
plt.ylabel('Counts', fontsize=18)
plt.xlabel('Category', fontsize=14)
plt.xticks(rotation='vertical')
# plt.xticks(color='w') # comment this to view labels
# plt.show()

threshold = 100  # taking a lower threshold so can include higher number of classes/categories in consideration, can change this to even lower if want more classes

warnings.filterwarnings('ignore')

main_categories = pd.DataFrame()
main_categories = bar_plot[bar_plot['count'] > threshold]
categories = main_categories['category'].values
categories = np.append(categories, 'Others')
not_category = []
df2['Others'] = 0

for i in df2.columns[2:]:
    if i not in categories:
        df2['Others'][df2[i] == 1] = 1
        not_category.append(i)

df2.drop(not_category, axis=1, inplace=True)

print(df2.shape)
print(df2)

# Now our data frame is of dimension (51646, 271), so our number of categories has been reduced by a significant number.


most_common_cat = pd.DataFrame()
most_common_cat['category'] = df2.columns[2:]
most_common_cat['count'] = df2.iloc[:, 2:].sum().values
most_common_cat.sort_values(['count'], inplace=True, ascending=False)
most_common_cat.reset_index(inplace=True, drop=True)
print(most_common_cat.head())

plt.figure(figsize=(15, 8))
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

pal = sns.color_palette("Blues_r", len(most_common_cat))
rank = most_common_cat['count'].argsort().argsort()

sns.barplot(most_common_cat['category'][:50], most_common_cat['count'][:50], palette=np.array(pal[::-1])[rank])
plt.axhline(threshold, ls='--', c='red')
plt.title("Most common categories", fontsize=24)
plt.ylabel('Counts', fontsize=18)
plt.xlabel('Category', fontsize=14)
plt.xticks(rotation='vertical')
# plt.show()


rowSums = df2.iloc[:, 2:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()

sns.set(font_scale=1.5)
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Number of categories per product", fontsize=24)
plt.ylabel('Number of Products', fontsize=18)
plt.xlabel('Number of categories', fontsize=18)

# plt.show()


boxplot = df2.copy()
boxplot['len'] = df2.description.apply(lambda x: len(x))  # length of descriptions
sns.set(style="whitegrid", rc={"font.size": 13, "axes.labelsize": 13})
plt.figure(figsize=(9, 4))

ax = sns.boxplot(x='len', data=boxplot, orient="h", palette="Set2")
plt.ylabel('')
plt.xlabel('Words')
plt.title("Distribution of the word frequency", fontsize=13)
plt.tight_layout(h_pad=3)

# Visualizing the main context/keywords of the description of products, to better understand the nature of data
from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize=(25, 25))
text = df2.description.values
cloud = WordCloud(stopwords=STOPWORDS,
                  background_color='black',
                  collocations=False,
                  width=2500,
                  height=1800
                  ).generate(" ".join(text))
plt.axis('off')
plt.title("Common words on the description", fontsize=40)
plt.imshow(cloud)


# plt.show()


############################################# 4. Data Cleaning NLP #############################################

# Utility function for data cleaning, natural language processing concepts

def decontract(sentence):
    sentence = str(sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence


def cleanPunc(sentence):
    sentence = str(sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def keepAlpha(sentence):
    sentence = str(sentence)
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', '', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def removeStopWords(sentence):
    sentence = str(sentence)
    global re_stop_words
    return re_stop_words.sub("", sentence)


stopwords = {'br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
             "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
             'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
             'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
             'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
             'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
             'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
             "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
             'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
             'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
             "wouldn't"}

re_stop_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W", re.I)

stemmer = SnowballStemmer("english")


def stemming(sentence):
    sentence = str(sentence)
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


# Applying data cleaning on description to ignore irrelevant words
df2['description'] = df2['description'].str.lower()
df2['description'] = df2['description'].apply(decontract)
df2['description'] = df2['description'].apply(cleanPunc)
df2['description'] = df2['description'].apply(keepAlpha)
df2['description'] = df2['description'].apply(removeStopWords)
df2['description'] = df2['description'].apply(stemming)

# Applying data cleaning on product name to ignore irrelevant words
df2['name'] = df2['name'].str.lower()
df2['name'] = df2['name'].apply(decontract)
df2['name'] = df2['name'].apply(cleanPunc)
df2['name'] = df2['name'].apply(keepAlpha)
df2['name'] = df2['name'].apply(removeStopWords)
df2['name'] = df2['name'].apply(stemming)

# creating new column information which is concatenation of product name and description, which stores overall
# context about any product
df2["information"] = df2["name"] + df2["description"]
print(df2.head())

############################################# 5. Data Splitting #############################################

"""
X_train, X_test, y_train, y_test = train_test_split(df2['information'],
                                                    df2[df2.columns[2:-1]],
                                                    test_size=0.3,
                                                    random_state=0,
                                                    shuffle=True)

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 3),
                             norm='l2')  # Using a tf-idf weighting scheme rather than normal boolean weights for better performance
vectorizer.fit(
    X_train)  # Reference : https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print("X_train shape : ", X_train.shape)
print("X_test shape : ", X_test.shape)

# To solve multi-label problems, we mainly have approaches:
#
# Binary classification : This strategy divides the problem into several independent binary classification tasks. It
# resembles the one-vs-rest method, but each classifier deals with a single label, which means the algorithm assumes
# they are mutually exclusive.

# Multi-class classification : The labels are combined into one big binary classifier called powerset. For instance,
# having the targets A, B, and C, with 0 or 1 as outputs, we have A B C -> [0 1 0], while the binary classification
# transformation treats it as A B C -> [0] [1] [0].

# We will first use the Binary classification technique.
"""

############################################# 6. Binary Classification Technique #############################################

"""

# We will first use the Binary classification technique, which has been also explained above. In the below, you can see how we are creating a separate classifier for a separate product category, in machine learning this technique is called one-vs-all. We have used a simple linear regression model as a single product classification model. Other models worth trying are Naive Bayes, SVC, Random Forest.

LR_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1))])
# one-vs-all, this strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes. In addition to its computational efficiency (only n_classes classifiers are needed), one advantage of this approach is its interpretability. Since each class is represented by one and one classifier only, it is possible to gain knowledge about the class by inspecting its corresponding classifier. This is the most commonly used strategy for multiclass classification and is a fair default choice.
# Reference : https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

# sag : Algorithm to use in the optimization problem, ‘saga’ also supports ‘elasticnet’ penalty  Reference : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# n_jobs : The number of jobs to use for the computation: the n_classes one-vs-rest problems are computed in parallel. Reference : https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
# clf : name given to the Pipeline

accuracy = 0
auc = 0
modelNumber = 1
for category in df2.columns[2:-1]:
    LR_pipeline.fit(X_train, y_train[category]) # Training logistic regression model on train data
    prediction = LR_pipeline.predict(X_test)    # calculating predictions
    acc = accuracy_score(y_test[category], prediction)
    au = roc_auc_score(y_test[category],prediction)
    accuracy = accuracy + acc
    auc = auc + au
    print('CATEGORY => {} '.format(category),'AUC ROC score => {}'.format(au)," Accuracy => {}".format(acc))
    filename = str(modelNumber)+"_model.sav"
    joblib.dump(LR_pipeline, filename)          # saving individual classifiers for later use
    modelNumber += 1
print("-------------------------------------------------------------------------------------------------------------------------------")
print('Test averaged Accuracy => {}'.format(accuracy/len(df2.columns[2:-1])))
print('Test averaged AUC ROC Score => {}'.format(auc/len(df2.columns[2:-1])))

# By using the above code, we create classifiers for each product category, print its individual accuracy, AUC ROC, and overall accuracy of the model as a multiclassification model.


# Api for category prediction
def categoryPrediction(name,description):
  # performing necessary data cleaning operations od product name and description
  name = name.lower()
  name = decontract(name)
  name = cleanPunc(name)
  name = keepAlpha(name)
  name = removeStopWords(name)
  name = stemming(name)

  description = description.lower()
  description = decontract(description)
  description = cleanPunc(description)
  description = keepAlpha(description)
  description = removeStopWords(description)
  description = stemming(description)

  information = name + description  # creating information text
  X_api = vectorizer.transform([information]) # transforming using already trained vectorizing transformer

  # LR_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1))])
  accuracy = 0
  modelNumber = 1
  for category in df2.columns[2:-1]:
    # LR_pipeline.fit(X_train, y_train[category])
    filename = str(modelNumber)+"_model.sav"
    modelNumber += 1
    LR_pipeline = joblib.load(filename) # loading already saved model

    # calculating test accuracy
    prediction = LR_pipeline.predict(X_api)
    if prediction==1:
      print('CATEGORY {}'.format(category)) # if models predicts true print that category


#categoryPrediction('iphone','apple')

# Test for API
name = "Samsung - Galaxy S7 32GB - Black Onyx (Sprint)"
description = "Qualcomm Snapdragon 820 MSM8996 2.2GHz quad-core processorAndroid 6.0 Marshmallow operating system4G mobile hotspot capability with support for up to 10 devicesWiFi Capable 802.11 a/b/g/n/ac5.1\" WQHD touch-screen displayBluetooth 4.2"
print(categoryPrediction(name,description)

name = "Duracell - AAA Batteries (4-Pack)"
description = "Compatible with select electronic devices; AAA size; DURALOCK Power Preserve technology; 4-pack"
categoryPrediction(name,description)

name = "Motorola - Moto 360 2nd Generation Men's Smartwatch 42mm Stainless Steel - Gold Stainless Steel"
description = "Fits most wrist sizesCompatible with most Apple&#174; iOS and Android cell phones22mm stainless steel bandWater-resistant designAt-a-glance notifications"
categoryPrediction(name,description)


"""

############################################# 7. ML Common #############################################

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf

tokenizer = Tokenizer(num_words=5000,
                      lower=True)
# num_words : the maximum number of words to keep, based on word frequency.
# lower : boolean. Whether to convert the texts to lowercase.

tokenizer.fit_on_texts(df2[
                           'information'])
# encoding words of information to integers, Updates internal vocabulary based on a list of sequences.

sequences = tokenizer.texts_to_sequences(df2['information'])
x = pad_sequences(sequences, maxlen=500)  # provide tagging to make each sequence of length 500

# Splitting dataset in train/test set
X_train, X_test, y_train, y_test = train_test_split(x, df2[df2.columns[2:-1]],
                                                    test_size=0.3,
                                                    random_state=0)
most_common_cat['class_weight'] = len(most_common_cat) / most_common_cat['count']
class_weight = {}
most_common_cat.head()

print(categories)  # categories in which our results will belong to

num_classes = y_train.shape[1]
max_words = len(tokenizer.word_index) + 1
maxlen = 500

print(num_classes)  # number of categories/classe

print(max_words)  # max words/ total vocab + 1

import operator


# Utility function to get predictions using Neural Net model
def categoryPredictionNN(name, description):
    # Data cleaning process
    name = name.lower()
    name = decontract(name)
    name = cleanPunc(name)
    name = keepAlpha(name)
    name = removeStopWords(name)
    name = stemming(name)

    description = description.lower()
    description = decontract(description)
    description = cleanPunc(description)
    description = keepAlpha(description)
    description = removeStopWords(description)
    description = stemming(description)

    information = name + description
    # necessary data preprocessing steps
    sequences = tokenizer.texts_to_sequences([information])
    x = pad_sequences(sequences, maxlen=500)
    prediction = model.predict(x)
    predScores = [score for pred in prediction for score in pred]
    predDict = {}
    for cla, score in zip(classes, predScores):
        predDict[cla] = score

    return sorted(predDict.items(), key=operator.itemgetter(1), reverse=True)[:10]  # return top 10 results


############################################# 7. Deep Learning-Based Models #############################################

"""
model = Sequential()
# Turns positive integers (indexes) into dense vectors of fixed size, input_dim = 500, output_dim = 300
model.add(Embedding(max_words, 300, input_length=maxlen))
# model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
callbacks = [
    ReduceLROnPlateau(),  # Reduce learning rate when a metric has stopped improving
    # EarlyStopping(patience=10),
    ModelCheckpoint(filepath='model-neural-net.h5', save_best_only=True)
]

model.summary()

# from tensorflow.keras.utils import plot_model

# plot_model(model, to_file='model_nn_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, y_train,
                    class_weight=class_weight,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.3,
                    callbacks=callbacks)

# Plotting losses wrt epochs(time)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# Plotting accuracy wrt epochs(time)
plt.plot(history.history["auc"], label="Training AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")
plt.legend()
plt.show()

metrics = model.evaluate(X_test, y_test)
print("{}: {}".format(model.metrics_names[1], metrics[1]))

classes = [col for col in df2.columns]
classes = classes[2:-1]
print(classes)

name = "Motorola - Moto 360 2nd Generation Men's Smartwatch 42mm Stainless Steel - Gold Stainless Steel"
description = "Fits most wrist sizesCompatible with most Apple&#174; iOS and Android cell phones22mm stainless steel " \
              "bandWater-resistant designAt-a-glance notifications "
prediction = categoryPredictionNN(name, description)
print(prediction)

name = "Samsung - Galaxy S7 32GB - Black Onyx (Sprint)"
description = "Qualcomm Snapdragon 820 MSM8996 2.2GHz quad-core processorAndroid 6.0 Marshmallow operating system4G " \
              "mobile hotspot capability with support for up to 10 devicesWiFi Capable 802.11 a/b/g/n/ac5.1\" WQHD " \
              "touch-screen displayBluetooth 4.2 "
prediction = categoryPredictionNN(name, description)
print(prediction)

name = "Duracell - AAA Batteries (4-Pack)"
description = "Compatible with select electronic devices; AAA size; DURALOCK Power Preserve technology; 4-pack"
prediction = categoryPredictionNN(name, description)
print(prediction)
"""
############################################# 8. Conv Net Model #############################################
"""
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D

model = Sequential()
model.add(Embedding(max_words, 300, input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=1))
# This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal)
#   dimension to produce a tensor of outputs.
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

callbacks = [
    ReduceLROnPlateau(),
    ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
]

model.summary()

history = model.fit(X_train, y_train,
                    class_weight=class_weight,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.3,
                    callbacks=callbacks)

# Plotting losses wrt epochs(time)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
# Plotting accuracy wrt epochs(time)
plt.plot(history.history["auc_1"], label="Training AUC")
plt.plot(history.history["val_auc_1"], label="Validation AUC")
plt.legend()
plt.show()

metrics = model.evaluate(X_test, y_test)
print("{}: {}".format(model.metrics_names[1], metrics[1]))

name = "Motorola - Moto 360 2nd Generation Men's Smartwatch 42mm Stainless Steel - Gold Stainless Steel"
description = "Fits most wrist sizesCompatible with most Apple&#174; iOS and Android cell phones22mm stainless steel " \
              "bandWater-resistant designAt-a-glance notifications "
prediction = categoryPredictionNN(name, description)
print(prediction)

name = "Samsung - Galaxy S7 32GB - Black Onyx (Sprint)"
description = "Qualcomm Snapdragon 820 MSM8996 2.2GHz quad-core processorAndroid 6.0 Marshmallow operating system4G " \
              "mobile hotspot capability with support for up to 10 devicesWiFi Capable 802.11 a/b/g/n/ac5.1\" WQHD " \
              "touch-screen displayBluetooth 4.2 "
prediction = categoryPredictionNN(name, description)
print(prediction)

name = "Duracell - AAA Batteries (4-Pack)"
description = "Compatible with select electronic devices; AAA size; DURALOCK Power Preserve technology; 4-pack"
prediction = categoryPredictionNN(name, description)
print(prediction)

name = "Keurig - Green Mountain Coffee Organic Ethiopia Yirgacheffe K-Cups (16-Pack)"
description = "Compatible with Keurig single-serve K-Cup and 2.0 coffee brewers; notes of citrus and ginger; 16-pack"
prediction = categoryPredictionNN(name, description)
print(prediction)

name = "Kung Fu Panda: Showdown of Legendary Legends - Xbox One"
description = "Jump into an all-out brawl for honor, glory and legend status"
prediction = categoryPredictionNN(name, description)
print(prediction)

##### This model trained for 30 epochs and its performance over epoch is quite stable as compared to the previous model

"""
############################################# 9. LSTM + Glove Model #############################################

classes = [col for col in df2.columns]
classes = classes[2:-1]
print(classes)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
# Importing glove file
glove_file = open('data/glove.6B.100d.txt', encoding="utf8") # Reference : https://nlp.stanford.edu/projects/glove/
# GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((max_words, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


from keras.layers import Input
from keras.layers import Flatten, LSTM
from keras.models import Model

deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(max_words, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
dropout_embedding_layer = Dropout(0.2)(embedding_layer)
LSTM_Layer_1 = LSTM(300)(dropout_embedding_layer)
dropout_LSTM_Layer_1 =  Dropout(0.2)(LSTM_Layer_1)
dense_layer_1 = Dense(num_classes, activation='sigmoid')(dropout_LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

callbacks = [
    ReduceLROnPlateau(),
    ModelCheckpoint(filepath='model-LSTM.h5', save_best_only=True)
]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
model.summary()


history = model.fit(X_train, y_train.values,
                    class_weight=class_weight,
                    batch_size=32,
                    epochs=30,
                    validation_split=0.3,
                    callbacks=callbacks)

# Plotting losses wrt epochs(time)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
# Plotting accuracy wrt epochs(time)
plt.plot(history.history["auc_3"], label="Training AUC")  # TODO : ---> KeyError: 'auc_3'
plt.plot(history.history["val_auc_3"], label="Validation AUC")
plt.legend()
plt.show()

metrics = model.evaluate(X_test, y_test)
print("{}: {}".format(model.metrics_names[1], metrics[1]))


name = "Motorola - Moto 360 2nd Generation Men's Smartwatch 42mm Stainless Steel - Gold Stainless Steel"
description = "Fits most wrist sizesCompatible with most Apple&#174; iOS and Android cell phones22mm stainless steel bandWater-resistant designAt-a-glance notifications"
prediction = categoryPredictionNN(name,description)
print(prediction)

name = "Samsung - Galaxy S7 32GB - Black Onyx (Sprint)"
description = "Qualcomm Snapdragon 820 MSM8996 2.2GHz quad-core processorAndroid 6.0 Marshmallow operating system4G " \
              "mobile hotspot capability with support for up to 10 devicesWiFi Capable 802.11 a/b/g/n/ac5.1\" WQHD " \
              "touch-screen displayBluetooth 4.2 "
prediction = categoryPredictionNN(name,description)
print(prediction)

name = "Duracell - AAA Batteries (4-Pack)"
description = "Compatible with select electronic devices; AAA size; DURALOCK Power Preserve technology; 4-pack"
prediction = categoryPredictionNN(name,description)
print(prediction)


name = "Keurig - Green Mountain Coffee Organic Ethiopia Yirgacheffe K-Cups (16-Pack)"
description = "Compatible with Keurig single-serve K-Cup and 2.0 coffee brewers; notes of citrus and ginger; 16-pack"
prediction = categoryPredictionNN(name,description)
print(prediction)

name = "Kung Fu Panda: Showdown of Legendary Legends - Xbox One"
description = "Jump into an all-out brawl for honor, glory and legend status"
prediction = categoryPredictionNN(name,description)
print(prediction)


# Saving the model
model.save('model-LSTM.h5')

import pickle
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Epoch 30/30
# 791/791 [==============================] - 88s 111ms/step - loss: 0.0102 - auc: 0.9903 - val_loss: 0.0578 - val_auc: 0.9687 - lr: 1.5000e-04
# 485/485 [==============================] - 6s 13ms/step - loss: 0.0647 - auc: 0.9678
# auc: 0.9678394198417664
# ['Coffee Pods & Beans', 'Blenders', 'Projector Screens', 'Over-Ear Headphones', 'DSLR Lenses', 'Laptop Chargers & Adapters', 'Coffee Makers', 'iPhone Cases & Clips', 'Portable Chargers/Power Packs', 'Printer Ink', 'Electric Dryers', 'Gas Dryers', 'Camera Batteries', '3D Printer Filament', 'Food Preparation Utensils', 'Grills', 'Cookware', 'All TV Stands', 'Mice', 'Computer Keyboards', 'All Printers', 'Pedals', 'Gas Ranges', 'Toner', 'Systems', 'Wall Chargers & Power Adapters', 'Electric Ranges', 'Camera Bags & Cases', 'Sound Bars', 'All Point & Shoot Cameras', 'Tea Kettles', 'Wall Mount Range Hoods', 'Laptop Batteries', 'Mirrorless Lenses', 'Internal Batteries', 'Cases', 'Outdoor Speakers', 'In-Wall & In-Ceiling Speakers', 'Bookshelf Speakers', 'Car Chargers', 'USB Cables & Hubs', 'Security Camera Systems', 'Laptop Bags & Cases', 'On-Ear Headphones', 'Flashes & Accessories', 'Bakeware', 'External Hard Drives', 'Single Ovens', 'PC Laptops', 'Smartwatch Accessories', 'Luggage', 'Patio Furniture & Decor', 'Apple Watch Bands & Straps', 'Outdoor Heating', 'Dash Installation Kits', 'Deck Harnesses', 'Antennas & Adapters', 'Coffee Pods', 'Prime Lenses', 'Multi-Cup Coffee Makers', 'Universal Camera Bags & Cases', 'Electric Tea Kettles', 'In-Ceiling Speakers', 'Full-Size Blenders', 'Smartwatch Bands', 'Outdoor Seating', 'iPhone 6s Plus Cases', 'iPhone 6s Cases', 'Others']
# [('Others', 1.0), ('Coffee Pods & Beans', 0.0), ('Blenders', 0.0), ('Projector Screens', 0.0), ('Over-Ear Headphones', 0.0), ('DSLR Lenses', 0.0), ('Laptop Chargers & Adapters', 0.0), ('Coffee Makers', 0.0), ('iPhone Cases & Clips', 0.0), ('Portable Chargers/Power Packs', 0.0)]
# [('Coffee Pods & Beans', 0.0), ('Blenders', 0.0), ('Projector Screens', 0.0), ('Over-Ear Headphones', 0.0), ('DSLR Lenses', 0.0), ('Laptop Chargers & Adapters', 0.0), ('Coffee Makers', 0.0), ('iPhone Cases & Clips', 0.0), ('Portable Chargers/Power Packs', 0.0), ('Printer Ink', 0.0)]
# [('Others', 0.20479976), ('Grills', 0.0016639956), ('Wall Chargers & Power Adapters', 0.00046400557), ('Wall Mount Range Hoods', 0.00040534537), ('Pedals', 0.00018245062), ('All TV Stands', 0.00014094167), ('Luggage', 0.00010193935), ('Laptop Bags & Cases', 8.508938e-05), ('Sound Bars', 1.2588902e-05), ('Printer Ink', 1.1086758e-05)]
#
# Process finished with exit code 0