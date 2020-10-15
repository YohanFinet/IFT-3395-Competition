import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

train = pd.read_csv('train.csv', delimiter=',')
train['Abstract'] = [doc.lower() for doc in train['Abstract']]
train['Abstract'] = [word_tokenize(doc) for doc in train['Abstract']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for i,doc in enumerate(train['Abstract']):
    processed_words = []
    lemmatizer = WordNetLemmatizer()
    for word, tag in pos_tag(doc):
        if word not in stopwords.words('english') and word.isalpha():
            processed_word = lemmatizer.lemmatize(word,tag_map[tag[0]])
            processed_words.append(processed_word)
    train.loc[i,'Abstract_processed'] = str(processed_words)

train_x, test_x, train_y, test_y = model_selection.train_test_split(train['Abstract_processed'],train['Category'],test_size=0.3)

encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

tfidf_vect = TfidfVectorizer(max_features=4000)
tfidf_vect.fit(train['Abstract_processed'])
train_x_tfidf = tfidf_vect.transform(train_x)
test_x_tfidf = tfidf_vect.transform(test_x)

NBM = naive_bayes.MultinomialNB()
NBM.fit(train_x_tfidf,train_y)
predictions_NBM = NBM.predict(test_x_tfidf)
print("Multinomial Naive Bayes Accuracy Score (Validation) -> ",accuracy_score(predictions_NBM, test_y)*100)
predictions_NBM = NBM.predict(train_x_tfidf)
print("Multinomial Naive Bayes Accuracy Score (Train) -> ",accuracy_score(predictions_NBM, train_y)*100)

SVM = svm.SVC(kernel='linear')
SVM.fit(train_x_tfidf,train_y)
predictions_SVM = SVM.predict(test_x_tfidf)
print("SVM Accuracy Score (Validation) -> ",accuracy_score(predictions_SVM, test_y)*100)
predictions_SVM = SVM.predict(train_x_tfidf)
print("SVM Accuracy Score (Train) -> ",accuracy_score(predictions_SVM, train_y)*100)

Forest = RandomForestClassifier(max_depth=7)
Forest.fit(train_x_tfidf, train_y)
score_Forest = Forest.score(test_x_tfidf, test_y)
print("Forest Accuracy Score (Validation) -> ",score_Forest*100)
score_Forest = Forest.score(train_x_tfidf, train_y)
print("Forest Accuracy Score (Train) -> ",score_Forest*100)

MLP = MLPClassifier(hidden_layer_sizes=(100, 5), solver='lbfgs')
MLP.fit(train_x_tfidf, train_y)
score_MLP = MLP.score(test_x_tfidf, test_y)
print("Multi-Layer Perceptron Accuracy Score (Validation) -> ",score_MLP*100)
score_MLP = MLP.score(train_x_tfidf, train_y)
print("Multi-Layer Perceptron Accuracy Score (Train) -> ",score_MLP*100)