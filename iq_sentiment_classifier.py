from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import socket
import sys
from urllib.parse import urlparse, parse_qs
from libs.HF_classifier import HuggingFaceTextClassifier
from libs.SVM_classifier import SVMTextClassifier
import json
import numpy as np
import pandas as pd
import torch
import cgi
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--host',type=str,default='0.0.0.0')
parser.add_argument('--port',type=int,default=80)
parser.add_argument('--mode',type=str,default='svm',help = 'svm or huggingface')
parser.add_argument('--device',type=str,default='gpu',help = 'cpu or gpu')
args = parser.parse_args()

MY_IP =args.host#'0.0.0.0'
PORT = args.port#80
mode = args.mode#'svm'
if args.device=='gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.device=='cpu':
    device = torch.device('cpu')

if mode == 'huggingface':    
    model_name = 'Tatyana/rubert-base-cased-sentiment-new'.replace('/','_')
    classifier = HuggingFaceTextClassifier(model_name,device)
    
if mode =='svm':    
    model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'.replace('/','_')
    classifier = SVMTextClassifier(model_name,device)
    
def create_message(text):    
    text_class = classifier.classify(text)
    print(text_class)
    columns = ['NEUTRAL','POSITIVE','NEGATIVE']
    message = pd.DataFrame(data =text_class, columns=columns).iloc[0].to_dict()
    message['received'] = 'ok' 
    return message

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)     
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):     
        self._set_headers()
        parse_dict = parse_qs(urlparse(self.path).query)
        str_data = parse_dict
        print(str_data)
        message = create_message(str_data['text'][0])
        self.wfile.write(json.dumps(message).encode('utf-8'))
        return

    def do_HEAD(self):
        self._set_headers()
        return

    def do_POST(self):   
        self._set_headers()  
        length = int(self.headers.get('Content-Length'))
        data = self.rfile.read(length)
        str_data = json.loads(data.decode()) 
        print(str_data)
        message = create_message(str_data['text'])
        self.wfile.write(json.dumps(message).encode('utf-8'))
        return 

def run(server_class=HTTPServer, handler_class=Server, port=PORT):
    server_address = (MY_IP, port)
    httpd = server_class(server_address, handler_class)
    
    print(f'http://YOUR_IP:{port}/')
    httpd.serve_forever()
    
if __name__ == "__main__":
    run()