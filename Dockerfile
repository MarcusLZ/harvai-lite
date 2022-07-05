FROM python:3.8.13-buster
COPY harvai /harvai
COPY raw_data /raw_data
COPY api /api
COPY requirements.txt /requirements.txt
COPY faiss_document_store.db /faiss_document_store.db
RUN pip install -r requirements.txt
RUN pip install --upgrade protobuf==3.20.0

RUN python -m nltk.downloader stopwords && python -m nltk.downloader punkt  && \
    python -m nltk.downloader averaged_perceptron_tagger

CMD uvicorn api.fast:app --port 8080 --host 0.0.0.0
