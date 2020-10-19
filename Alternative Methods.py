import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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

small_cat_train_y = train_y
small_cat_train_y = small_cat_train_y.replace(to_replace='astro-ph', value='astro')
small_cat_train_y = small_cat_train_y.replace(to_replace='astro-ph.CO', value='astro')
small_cat_train_y = small_cat_train_y.replace(to_replace='astro-ph.GA', value='astro')
small_cat_train_y = small_cat_train_y.replace(to_replace='astro-ph.SR', value='astro')
small_cat_train_y = small_cat_train_y.replace(to_replace='cond-mat.mes-hall', value='cond-mat')
small_cat_train_y = small_cat_train_y.replace(to_replace='cond-mat.mtrl-sci', value='cond-mat')
small_cat_train_y = small_cat_train_y.replace(to_replace='hep-ph', value='hep')
small_cat_train_y = small_cat_train_y.replace(to_replace='hep-th', value='hep')
small_cat_train_y = small_cat_train_y.replace(to_replace='math.AP', value='math')
small_cat_train_y = small_cat_train_y.replace(to_replace='math.CO', value='math')

small_cat_test_y = test_y
small_cat_test_y = small_cat_test_y.replace(to_replace='astro-ph', value='astro')
small_cat_test_y = small_cat_test_y.replace(to_replace='astro-ph.CO', value='astro')
small_cat_test_y = small_cat_test_y.replace(to_replace='astro-ph.GA', value='astro')
small_cat_test_y = small_cat_test_y.replace(to_replace='astro-ph.SR', value='astro')
small_cat_test_y = small_cat_test_y.replace(to_replace='cond-mat.mes-hall', value='cond-mat')
small_cat_test_y = small_cat_test_y.replace(to_replace='cond-mat.mtrl-sci', value='cond-mat')
small_cat_test_y = small_cat_test_y.replace(to_replace='hep-ph', value='hep')
small_cat_test_y = small_cat_test_y.replace(to_replace='hep-th', value='hep')
small_cat_test_y = small_cat_test_y.replace(to_replace='math.AP', value='math')
small_cat_test_y = small_cat_test_y.replace(to_replace='math.CO', value='math')

"""
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
"""

tfidf_vect = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
tfidf_vect.fit(train['Abstract_processed'])
train_x_tfidf = tfidf_vect.transform(train_x)
test_x_tfidf = tfidf_vect.transform(test_x)

astro_train_x = train_x.loc[small_cat_train_y == 'astro']
astro_train_x = tfidf_vect.transform(astro_train_x)
astro_train_y = train_y.loc[small_cat_train_y == 'astro']
astro_test_x = test_x.loc[small_cat_test_y == 'astro']
astro_test_x = tfidf_vect.transform(astro_test_x)
astro_test_y = test_y.loc[small_cat_test_y == 'astro']

cond_mat_train_x = train_x.loc[small_cat_train_y == 'cond-mat']
cond_mat_train_x = tfidf_vect.transform(cond_mat_train_x)
cond_mat_train_y = train_y.loc[small_cat_train_y == 'cond-mat']
cond_mat_test_x = test_x.loc[small_cat_test_y == 'cond-mat']
cond_mat_test_x = tfidf_vect.transform(cond_mat_test_x)
cond_mat_test_y = test_y.loc[small_cat_test_y == 'cond-mat']

hep_train_x = train_x.loc[small_cat_train_y == 'hep']
hep_train_x = tfidf_vect.transform(hep_train_x)
hep_train_y = train_y.loc[small_cat_train_y == 'hep']
hep_test_x = test_x.loc[small_cat_test_y == 'hep']
hep_test_x = tfidf_vect.transform(hep_test_x)
hep_test_y = test_y.loc[small_cat_test_y == 'hep']

math_train_x = train_x.loc[small_cat_train_y == 'math']
math_train_x = tfidf_vect.transform(math_train_x)
math_train_y = train_y.loc[small_cat_train_y == 'math']
math_test_x = test_x.loc[small_cat_test_y == 'math']
math_test_x = tfidf_vect.transform(math_test_x)
math_test_y = test_y.loc[small_cat_test_y == 'math']

print("NBM")
NBM = naive_bayes.MultinomialNB()
NBM.fit(train_x_tfidf, small_cat_train_y)
predictions_NBM = NBM.predict(test_x_tfidf)
print("Multinomial Naive Bayes Accuracy Score (Validation) -> ", accuracy_score(predictions_NBM, small_cat_test_y) * 100)
predictions_NBM_train = NBM.predict(train_x_tfidf)
print("Multinomial Naive Bayes Accuracy Score (Train) -> ", accuracy_score(predictions_NBM_train, small_cat_train_y) * 100)
print()

alphas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
for a in alphas:
    print("NBM Astro alpha = ", a)
    NBM_astro = naive_bayes.MultinomialNB(alpha=a)
    NBM_astro.fit(astro_train_x, astro_train_y)
    predictions_NBM_astro = NBM_astro.predict(astro_test_x)
    print("Multinomial Naive Bayes Astro Accuracy Score (Validation) -> ", accuracy_score(predictions_NBM_astro, astro_test_y) * 100)
    predictions_NBM_astro_train = NBM_astro.predict(astro_train_x)
    print("Multinomial Naive Bayes Astro Accuracy Score (Train) -> ", accuracy_score(predictions_NBM_astro_train, astro_train_y) * 100)
    print()

C_list = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
for c in C_list:
    print("SVM Astro C = ", c)
    SVM = svm.SVC(C=c, kernel='rbf')
    SVM.fit(astro_train_x,astro_train_y)
    predictions_SVM = SVM.predict(astro_test_x)
    print("SVM Accuracy Score (Validation) -> ",accuracy_score(predictions_SVM, astro_test_y)*100)
    predictions_SVM = SVM.predict(astro_train_x)
    print("SVM Accuracy Score (Train) -> ",accuracy_score(predictions_SVM, astro_train_y)*100)
    print()

depths = [10, 20, 30, 40, 50, 60]
for d in depths:
    print("Forest Astro depth = ", d)
    Forest = RandomForestClassifier(max_depth=d)
    Forest.fit(astro_train_x,astro_train_y)
    score_Forest = Forest.score(astro_test_x, astro_test_y)
    print("Forest Accuracy Score (Validation) -> ",score_Forest*100)
    score_Forest = Forest.score(astro_train_x,astro_train_y)
    print("Forest Accuracy Score (Train) -> ",score_Forest*100)
    print()

neurones = [25, 50, 75, 100, 150]
activation = ['logistic', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
for n in neurones:
    for a in activation:
        for s in solver:
            print("MLP Astro", n, " hidden neurones, ", a, " activation, ", s, " solver")
            MLP = MLPClassifier(hidden_layer_sizes=(n), solver=s, max_iter=300, activation=a)
            MLP.fit(astro_train_x,astro_train_y)
            score_MLP = MLP.score(astro_test_x, astro_test_y)
            print("Multi-Layer Perceptron Accuracy Score (Validation) -> ",score_MLP*100)
            score_MLP = MLP.score(astro_train_x,astro_train_y)
            print("Multi-Layer Perceptron Accuracy Score (Train) -> ",score_MLP*100)
            print()

l1_ratios = [0.01, 0.1, 0.2, 0.3]
C_list = [0.01, 0.1, 0.2, 0.3]
for l in l1_ratios:
    for c in C_list:
        print("LR Astro l1 ratio = ", l, ", C = ", c)
        LR = LogisticRegression(penalty='elasticnet', l1_ratio=l, C=c, solver='saga')
        LR.fit(astro_train_x,astro_train_y)
        score_LR = LR.score(astro_test_x, astro_test_y)
        print("LR Accuracy Score (Validation) -> ", score_LR*100)
        score_LR = LR.score(astro_train_x,astro_train_y)
        print("LR Accuracy Score (Train) -> ", score_LR * 100)
        print()

for d in depths:
    print("GB Astro depth = ", d)
    GBC = GradientBoostingClassifier(max_depth=d)
    GBC.fit(astro_train_x,astro_train_y)
    score_GBC = GBC.score(astro_test_x, astro_test_y)
    print("GBC Accuracy Score (Validation) -> ", score_GBC * 100)
    score_GBC = GBC.score(astro_train_x,astro_train_y)
    print("GBC Accuracy Score (Train) -> ", score_GBC * 100)
    print()

print("NBM Cond-mat")
NBM_cond = naive_bayes.MultinomialNB()
NBM_cond.fit(cond_mat_train_x, cond_mat_train_y)
predictions_NBM_cond = NBM_cond.predict(cond_mat_test_x)
print("Multinomial Naive Bayes Cond-mat Accuracy Score (Validation) -> ", accuracy_score(predictions_NBM_cond, cond_mat_test_y) * 100)
predictions_NBM_cond_train = NBM_cond.predict(cond_mat_train_x)
print("Multinomial Naive Bayes Cond-mat Accuracy Score (Train) -> ", accuracy_score(predictions_NBM_cond_train, cond_mat_train_y) * 100)
print()

print("NBM Hep")
NBM_hep = naive_bayes.MultinomialNB()
NBM_hep.fit(hep_train_x, hep_train_y)
predictions_NBM_hep = NBM_hep.predict(hep_test_x)
print("Multinomial Naive Bayes Hep Accuracy Score (Validation) -> ", accuracy_score(predictions_NBM_hep, hep_test_y) * 100)
predictions_NBM_hep_train = NBM_hep.predict(hep_train_x)
print("Multinomial Naive Bayes Hep Accuracy Score (Train) -> ", accuracy_score(predictions_NBM_hep_train, hep_train_y) * 100)
print()

print("NBM Math")
NBM_math = naive_bayes.MultinomialNB()
NBM_math.fit(math_train_x, math_train_y)
predictions_NBM_math = NBM_math.predict(math_test_x)
print("Multinomial Naive Bayes Math Accuracy Score (Validation) -> ", accuracy_score(predictions_NBM_math, math_test_y) * 100)
predictions_NBM_math_train = NBM_math.predict(math_train_x)
print("Multinomial Naive Bayes Math Accuracy Score (Train) -> ", accuracy_score(predictions_NBM_math_train, math_train_y) * 100)
print()

predict_astro_test_x = test_x.loc[predictions_NBM == 'astro']
predict_astro_test_x = tfidf_vect.transform(predict_astro_test_x)
predict_astro_test_y = NBM_astro.predict(predict_astro_test_x)

print("----------predictions NBM--------------")
print(predictions_NBM)
print()
print("---------------predict astro test x--------------")
print(predict_astro_test_x)
print()
print("---------------predict astro test y--------------")
print(predict_astro_test_y)
print()

for i in predict_astro_test_y.index:
    predictions_NBM.at[i] = predict_astro_test_y[i]

predict_cond_mat_test_x = test_x.loc[predictions_NBM == 'cond_mat']
predict_cond_mat_test_x = tfidf_vect.transform(predict_cond_mat_test_x)
predict_cond_mat_test_y = NBM_cond.predict(predict_cond_mat_test_x)
for i in predict_cond_mat_test_y.index:
    predictions_NBM.at[i] = predict_cond_mat_test_y[i]

predict_hep_test_x = test_x.loc[predictions_NBM == 'hep']
predict_hep_test_x = tfidf_vect.transform(predict_hep_test_x)
predict_hep_test_y = NBM_hep.predict(predict_hep_test_x)
for i in predict_hep_test_y.index:
    predictions_NBM.at[i] = predict_hep_test_y[i]

predict_math_test_x = test_x.loc[predictions_NBM == 'math']
predict_math_test_x = tfidf_vect.transform(predict_math_test_x)
predict_math_test_y = NBM_math.predict(predict_math_test_x)
for i in predict_math_test_y.index:
    predictions_NBM.at[i] = predict_math_test_y[i]

print("Final Prediction -> ", accuracy_score(predictions_NBM, test_y))

"""
alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for a in alphas:
    print("MNB alpha = ", a)
    NBM = naive_bayes.MultinomialNB(alpha=a)
    NBM.fit(train_x_tfidf,train_y)
    predictions_NBM = NBM.predict(test_x_tfidf)
    print("Multinomial Naive Bayes Accuracy Score (Validation) -> ",accuracy_score(predictions_NBM, test_y)*100)
    predictions_NBM = NBM.predict(train_x_tfidf)
    print("Multinomial Naive Bayes Accuracy Score (Train) -> ",accuracy_score(predictions_NBM, train_y)*100)
    print()

C_list = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
for c in C_list:
    print("SVM C = ", c)
    SVM = svm.SVC(C=c, kernel='rbf')
    SVM.fit(train_x_tfidf,train_y)
    predictions_SVM = SVM.predict(test_x_tfidf)
    print("SVM Accuracy Score (Validation) -> ",accuracy_score(predictions_SVM, test_y)*100)
    predictions_SVM = SVM.predict(train_x_tfidf)
    print("SVM Accuracy Score (Train) -> ",accuracy_score(predictions_SVM, train_y)*100)
    print()


poly = [3, 4, 5, 6]
for i in poly:
    print("poly ", i)
    SVM = svm.SVC(kernel='poly', degree=i)
    SVM.fit(train_x_tfidf,train_y)
    predictions_SVM = SVM.predict(test_x_tfidf)
    print("SVM Accuracy Score (Validation) -> ",accuracy_score(predictions_SVM, test_y)*100)
    predictions_SVM = SVM.predict(train_x_tfidf)
    print("SVM Accuracy Score (Train) -> ",accuracy_score(predictions_SVM, train_y)*100)
    print()



Forest = RandomForestClassifier(max_depth=35)
Forest.fit(train_x_tfidf, train_y)
score_Forest = Forest.score(test_x_tfidf, test_y)
print("Forest Accuracy Score (Validation) -> ",score_Forest*100)
score_Forest = Forest.score(train_x_tfidf, train_y)
print("Forest Accuracy Score (Train) -> ",score_Forest*100)



neurones = [25, 50, 75, 100, 150]
activation = ['logistic', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
for n in neurones:
    for a in activation:
        for s in solver:
            print("MLP", n, " hidden neurones, ", a, " activation, ", s, " solver")
            MLP = MLPClassifier(hidden_layer_sizes=(n), solver=s, max_iter=300, activation=a)
            MLP.fit(train_x_tfidf, train_y)
            score_MLP = MLP.score(test_x_tfidf, test_y)
            print("Multi-Layer Perceptron Accuracy Score (Validation) -> ",score_MLP*100)
            score_MLP = MLP.score(train_x_tfidf, train_y)
            print("Multi-Layer Perceptron Accuracy Score (Train) -> ",score_MLP*100)
            print()


NBM = naive_bayes.MultinomialNB()
SVM = svm.SVC(kernel='rbf', probability=True)
Forest = RandomForestClassifier(max_depth=35)
MLP = MLPClassifier(hidden_layer_sizes=(75), solver='adam', max_iter=300, activation='relu')
vote = ['hard', 'soft']
for v in vote:
    print("voting all: ", v)
    VC = VotingClassifier(estimators=[('nbm', NBM), ('svm', SVM), ('forest', Forest), ('mlp', MLP)], voting=v)
    VC.fit(train_x_tfidf, train_y)
    score_VC = VC.score(test_x_tfidf, test_y)
    print("Voting Classifier Accuracy Score (Validation) -> ", score_VC * 100)
    score_VC = VC.score(train_x_tfidf, train_y)
    print("Voting Classifier Accuracy Score (Train) -> ", score_VC * 100)
    print()


for v in vote:
    print("voting NBM SVM MLP: ", v)
    VC = VotingClassifier(estimators=[('nbm', NBM), ('svm', SVM), ('mlp', MLP)], voting=v)
    VC.fit(train_x_tfidf, train_y)
    score_VC = VC.score(test_x_tfidf, test_y)
    print("Voting Classifier Accuracy Score (Validation) -> ", score_VC * 100)
    score_VC = VC.score(train_x_tfidf, train_y)
    print("Voting Classifier Accuracy Score (Train) -> ", score_VC * 100)
    print()


for v in vote:
    print("voting NBM SVM Forest: ", v)
    VC = VotingClassifier(estimators=[('nbm', NBM), ('svm', SVM), ('forest', Forest)], voting=v)
    VC.fit(train_x_tfidf, train_y)
    score_VC = VC.score(test_x_tfidf, test_y)
    print("Voting Classifier Accuracy Score (Validation) -> ", score_VC * 100)
    score_VC = VC.score(train_x_tfidf, train_y)
    print("Voting Classifier Accuracy Score (Train) -> ", score_VC * 100)
    print()

for v in vote:
    print("voting NBM SVM: ", v)
    VC = VotingClassifier(estimators=[('nbm', NBM), ('svm', SVM)], voting=v)
    VC.fit(train_x_tfidf, train_y)
    score_VC = VC.score(test_x_tfidf, test_y)
    print("Voting Classifier Accuracy Score (Validation) -> ", score_VC * 100)
    score_VC = VC.score(train_x_tfidf, train_y)
    print("Voting Classifier Accuracy Score (Train) -> ", score_VC * 100)
    print()


for v in vote:
    print("voting NBM MLP: ", v)
    VC = VotingClassifier(estimators=[('nbm', NBM), ('mlp', MLP)], voting=v)
    VC.fit(train_x_tfidf, train_y)
    score_VC = VC.score(test_x_tfidf, test_y)
    print("Voting Classifier Accuracy Score (Validation) -> ", score_VC * 100)
    score_VC = VC.score(train_x_tfidf, train_y)
    print("Voting Classifier Accuracy Score (Train) -> ", score_VC * 100)
    print()

for v in vote:
    print("voting SVM MLP: ", v)
    VC = VotingClassifier(estimators=[('svm', SVM), ('mlp', MLP)], voting=v)
    VC.fit(train_x_tfidf, train_y)
    score_VC = VC.score(test_x_tfidf, test_y)
    print("Voting Classifier Accuracy Score (Validation) -> ", score_VC * 100)
    score_VC = VC.score(train_x_tfidf, train_y)
    print("Voting Classifier Accuracy Score (Train) -> ", score_VC * 100)
    print()


GBC = GradientBoostingClassifier(max_depth=35)
GBC.fit(train_x_tfidf, train_y)
score_GBC = GBC.score(test_x_tfidf, test_y)
print("GBC Accuracy Score (Validation) -> ", score_GBC * 100)
score_GBC = GBC.score(train_x_tfidf, train_y)
print("GBC Accuracy Score (Train) -> ", score_GBC * 100)
print()


l1_ratios = [0.01, 0.1, 0.2, 0.3]
C_list = [0.01, 0.1, 0.2, 0.3]
for l in l1_ratios:
    for c in C_list:
        LR = LogisticRegression(penalty='elasticnet', l1_ratio=l, C=c, solver='saga')
        LR.fit(train_x_tfidf, train_y)
        score_LR = LR.score(test_x_tfidf, test_y)
        print("LR Accuracy Score (Validation) -> ", score_LR*100)
        score_LR = LR.score(train_x_tfidf, train_y)
        print("LR Accuracy Score (Train) -> ", score_LR * 100)
        print()
"""