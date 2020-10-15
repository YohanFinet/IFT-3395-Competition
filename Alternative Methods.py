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
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(doc):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    train.loc[i,'Abstract_processed'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train['Abstract_processed'],train['Category'],test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=4000)
Tfidf_vect.fit(train['Abstract_processed'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

NBM = naive_bayes.MultinomialNB()
NBM.fit(Train_X_Tfidf,Train_Y)
predictions_NBM = NBM.predict(Test_X_Tfidf)
print("Multinomial Naive Bayes Accuracy Score (Validation) -> ",accuracy_score(predictions_NBM, Test_Y)*100)
predictions_NBM = NBM.predict(Train_X_Tfidf)
print("Multinomial Naive Bayes Accuracy Score (Train) -> ",accuracy_score(predictions_NBM, Train_Y)*100)

SVM = svm.SVC(kernel='linear')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score (Validation) -> ",accuracy_score(predictions_SVM, Test_Y)*100)
predictions_SVM = SVM.predict(Train_X_Tfidf)
print("SVM Accuracy Score (Train) -> ",accuracy_score(predictions_SVM, Train_Y)*100)

Forest = RandomForestClassifier(max_depth=7)
Forest.fit(Train_X_Tfidf, Train_Y)
score_Forest = Forest.score(Test_X_Tfidf, Test_Y)
print("Forest Accuracy Score (Validation) -> ",score_Forest*100)
score_Forest = Forest.score(Train_X_Tfidf, Train_Y)
print("Forest Accuracy Score (Train) -> ",score_Forest*100)

MLP = MLPClassifier(hidden_layer_sizes=(100, 5), solver='lbfgs')
MLP.fit(Train_X_Tfidf, Train_Y)
score_MLP = MLP.score(Test_X_Tfidf, Test_Y)
print("Multi-Layer Perceptron Accuracy Score (Validation) -> ",score_MLP*100)
score_MLP = MLP.score(Train_X_Tfidf, Train_Y)
print("Multi-Layer Perceptron Accuracy Score (Train) -> ",score_MLP*100)