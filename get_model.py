from transformers import AutoTokenizer, AutoModelForTokenClassification
model_name = 'Tatyana/rubert-base-cased-sentiment-new'

model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(f"../model/{model_name.replace('/','_')}")
tokenizer.save_pretrained(f"../tokenizer/{model_name.replace('/','_')}")

print(f"\nModel and tokenizer saved in /model/{model_name.replace('/','_')} and /tokenizer/{model_name.replace('/','_')} directories")