import os
import sys
import traceback
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HuggingFaceTextClassifier:
    def __init__(self,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(f'../tokenizer/{model_name}')
        self.model = AutoModelForTokenClassification.from_pretrained(f'../model/{model_name}')
        
    def classify(self,text): 
        print(text)
        input_ids = self.tokenizer.encode(text,max_length=512,truncation=True, padding=True)
        input_tensor = torch.tensor(input_ids).to(device).unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = self.model.to(device)(input_tensor)
        classes = outputs[0][:,0,:]
        return classes.cpu().numpy()