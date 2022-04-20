Для того, чтобы построить докер надо исполнить строчку

docker build -t iqmen:classifier .


Чтобы запустить этот докер надо исполнить строчку

docker run -p 8080:80 -it iqmen:classifier --mode huggingface

где параметр *-p 8080:80* перенаправляет данные из порта localhost:8080 на порт 80 в докере


Исполняемый скрипт iq_sentiment_classifier.py имеет параметры

*--host default='0.0.0.0'*

*--port default=80*

*--mode default='svm'* может принимать значения "svm" (классификатор SVM, обученный на эмбедднигах от sentence-transformers/distiluse-base-multilingual-cased-v1) или "huggingface" (классификатор с huggingface от Tatyana/rubert-base-cased-sentiment-new)

*--device default='gpu'* может принимать значения "cpu" или "gpu". Если выбрано значение "gpu", но torch не может запустить модель на GPU, тогда скрипт выведет сообщение об этом и завершится.

Веб-сервис будет запущен на localhost, текст для классификации он принимает в параметрах POST запроса http://0.0.0.0:8080?text=текст
