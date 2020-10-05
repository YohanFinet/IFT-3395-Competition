import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, train_data_file_name):
        self.train_data = pd.read_csv(train_data_file_name, delimiter=',', header=0).to_numpy()

    def concatenate_documents(self):
        pass

    def compute_conditionnal_prob(self):
        pass

    def compute_class_prob(self):
        pass

    def compute_predictions(self, conditional_prob, class_prob, test_data):
        pass

#nb = NaiveBayesClassifier('train.csv')
#print(nb.train_data[:,2])