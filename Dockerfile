FROM pytorch/pytorch

ENV LANG=ru_RU.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get install -y python3-pip libpq5 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install numpy --no-cache-dir pandas transformers sentence_transformers argparse sklearn --no-cache-dir 

WORKDIR /

COPY /get_model.py /
RUN python get_model.py && rm -r ~/.cache/huggingface/transformers/

COPY /iq_sentiment_classifier.py /
COPY libs/ libs/
RUN mkdir /output/

ENTRYPOINT ["python", "iq_sentiment_classifier.py"]