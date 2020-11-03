#Bernouli
import numpy as np
import pandas as pd
import string
import re
data=pd.read_csv('C:\\Users\\Yassine\\Desktop\\Étude\\3395\\competition\\train.csv', sep=',',header=0).to_numpy()
n_classes = np.unique(data[:,2])
n_classes_n = len(n_classes)
table = str.maketrans(dict.fromkeys(string.punctuation))
train_data_1 = data
test_data = pd.read_csv('C:\\Users\\Yassine\\Desktop\\Étude\\3395\\competition\\test.csv', sep=',',header=0).to_numpy()

no_word = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he'
           , 'him', 'his', 'himself', 'she', 'her','hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs'
           ,'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am','is', 'are', 'was', 'were'
           , 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and'
           , 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against'
           , 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in'
           , 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why'
           , 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
           'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm'
           , 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 
           'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']


vocabulaire_1 = {}
greek= ["alpha", "beta", "gamma", "delta", "epsilon", "zata", "eta", "theta", "iota","kappa", "lambda", "mu", "nu", "xi", "omikron", "pi", "rho", "sigma", "tau", "upsilon", "phi","chi", "psi", "omega"]


#Trouve le vocabulaire des documents et TRAITEMENT DE DONNÉES

        
for i in train_data_1:
    for j in i[1].split():
        #Enlever les cractére speciaux de l'élement j
        word = j.translate(table).lower()
        # Verifier si 'word' n'est pas un nombre 
        if not word.isdigit():
            #Enlever tous les chiffre existant dans 'word' 
            for k in re.split(r'(\d+)', word):
                wordf = ''.join(i for i in k if not i.isdigit())
                #verifier si 'wordf' est un mot non important ou si c'est une chaine vide 
                if (wordf not in no_word )and(wordf != ''):
                    q =1
                    #verifier si 'wordf' contient un nom d'une lettre greque et les remplacer par cette lettre 
                    #si c'est le cas, on ajoute le mot au dictionnaire 
                    for m in greek:
                        if m in wordf:
                            vocabulaire_1[m]= 0
                            q=0
                    #on ajoute le mot au dictionnaire        
                    if q == 1:
                        vocabulaire_1[wordf]= 0
                    



    
#Pour chaque document de chaque classe on fait son traitement et on indique 
#la présence d'un mot (de notre vocabulaire) par un 1
result_1 = {}
for i in n_classes:
    matrix = []
    for j in range(len(train_data_1)):
        bag = vocabulaire_1.copy()
        #verifier que le document appartien a la classe i
        if train_data_1[j][2] == i:
            #faire le meme traitement des données
            for k in train_data_1[j][1].split(): 
                for s in re.split(r'(\d+)', k):
                    word = ''.join(i for i in s.translate(table).lower() if not i.isdigit()) 
                    for m in greek:
                        if m in word:
                            word = m
                    
                    if bag.get(word,5) != 5:
                        bag[word] = 1
            vector = np.array([bag.get(k) for k in bag])
            matrix.append(vector)
    result_1[i]= matrix
    print(i)
    
    

#Déterminer la probabilité de la présence d'un mot dans chaque classe 
classes_prob_1 =[]
for i in result_1:

    a=np.zeros(len(vocabulaire_1))
    for j in result_1[i]:
        a=a+j
    #notre alpha = 0,15 : c le meilleur qu'on a trouvé
    classes_prob_1.append((a+0.15)/(len(result_1[i])))

    #Trouver les mots qui sont plus fréquents que les autres.  
#sa probabilité doit étre supérieur au double de tous les probabilités des autres mots
best_words =[]
for i in classes_prob_1:
    s = np.ones(len(vocabulaire_1))
    for j in classes_prob_1:
        if False in (i==j):
            for k in range(len(j)):
                if (i[k] /j[k] <2):
                    s[k] = 0
    best_words.append(s)
                    
                    

#Traitement des données pour le Test 
matrix_1 = []
for j in range(len(test_data)):
    bag = vocabulaire_1.copy()
    for k in test_data[j][1].split():
        for s in re.split(r'(\d+)', k):
            word = ''.join(i for i in s.translate(table).lower() if not i.isdigit())
            for m in greek:
                if m in word:
                    word = m
            
            if (bag.get(word ,5) != 5):
                bag[word ] = 1

    vector = np.array([bag.get(i) for i in bag])
    matrix_1.append(vector)
    


#fair la prediction     
predict=[]
for i in range(len(matrix_1)):
    best_class_accuracy=-np.Inf
    class_number=-1
    for j in range(len(classes_prob_1)):
        class_accuracy=0
        # si un mot existe on va remplacer le 1 par la probabilité de ce mot dans notre 'classes_prob_1'
        # sinon on va remplacer le 0 par (1-la probabilité de ce mot dans notre 'classes_prob_1')
        # pour faire ce calcul on a : np.abs(1-(matrix_1[i]+classes_prob_1[j]))
        # et puis on double la probabilité des mots important dans best_words qui existe dans le document 
        m = np.abs(1-(matrix_1[i]+classes_prob_1[j]))+(best_words[j]*matrix_1[i])
        
        # on utilise log pour calculer la somme des log de probabilité au lieu du produit    
        class_accuracy = np.log(m).sum()
        
            
        #on choisit la meilleur classe 
        if class_accuracy>best_class_accuracy:
            class_number = j
            best_class_accuracy = class_accuracy
    predict.append([i,n_classes[class_number]])
