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

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)     
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()        
        self.wfile.write(json.dumps({'what should i do': 'you should do POST request', 'is this thing working': 'yes'}).encode('utf-8'))
        return

    def do_HEAD(self):
        self._set_headers()
        return

    def do_POST(self):        
        parse_dict = parse_qs(urlparse(self.path).query)
        str_data = parse_dict
        print(str_data)
        self._set_headers()
        text_class = classifier.classify(str_data['text'][0])
        print(text_class)
        columns = ['NEUTRAL','POSITIVE','NEGATIVE']
        message = pd.DataFrame(data =text_class, columns=columns).iloc[0].to_dict()
        message['received'] = 'ok'        
        self.wfile.write(json.dumps(message).encode('utf-8'))
        return 

def run(server_class=HTTPServer, handler_class=Server, port=PORT):
    print(MY_IP, port)
    server_address = (MY_IP, port)
    httpd = server_class(server_address, handler_class)
    
    print(f'Starting httpd on port {MY_IP}:{port}...')
    httpd.serve_forever()
    
if __name__ == "__main__":
    run()