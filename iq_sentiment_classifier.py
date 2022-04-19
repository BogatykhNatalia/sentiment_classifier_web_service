from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import socket
import sys
from urllib.parse import urlparse, parse_qs
from libs.HF_classifier import HuggingFaceTextClassifier
import json
import numpy as np
import pandas as pd
# from tqdm import tqdm
import cgi

model_name = 'Tatyana/rubert-base-cased-sentiment-new'.replace('/','_')
MY_IP = '0.0.0.0'
PORT = 80

classifier = HuggingFaceTextClassifier(model_name)

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)     
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()        
        self.wfile.write(json.dumps({'hello': 'world', 'received': 'ok'}).encode('utf-8'))
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
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()