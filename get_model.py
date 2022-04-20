from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer

model_name = 'Tatyana/rubert-base-cased-sentiment-new'

model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(f"../model/{model_name.replace('/','_')}")
tokenizer.save_pretrained(f"../tokenizer/{model_name.replace('/','_')}")

model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
model = SentenceTransformer(model_name)
model.save(f"../model/{model_name.replace('/','_')}")