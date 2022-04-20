import os
import sys
import traceback
import re
import torch
import joblib
import sklearn
import numpy as np
from sentence_transformers import SentenceTransformer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SVMTextClassifier:
    def __init__(self,model_name,device):
        self.model = joblib.load("./libs/svm_v1.pkl")
        self.embedder = SentenceTransformer(f'../model/{model_name}')     
        self.device = device
    
    def classify(self,text):
        self.embedding = self.embedder.encode(text).reshape(1,-1)
        prediction = self.model.predict(self.embedding)
        test_one_hot_class = np.zeros((prediction.size, prediction.max()+1))
        test_one_hot_class[np.arange(prediction.size),prediction] = 1
        return test_one_hot_class