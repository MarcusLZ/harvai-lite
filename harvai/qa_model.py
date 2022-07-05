from contextvars import Context
from re import A
from webbrowser import get

from harvai.data import preprocessing_user_input
from harvai.nn_model import Nn_model
import requests

API_URL = "https://api-inference.huggingface.co/models/etalab-ia/camembert-base-squadFR-fquad-piaf"
headers = {"Authorization": "Bearer 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"}


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def get_answer(question,retriever,article_number,digits=False):
    """ Instanciate and use the transformer model"""

    context, parsed_context , article_reference = get_context(question, retriever,article_number,digits)
    output = query({
	"inputs": {
		"question": question,
		"context": context
	},
    })

    return output, parsed_context , context , article_reference


def get_context(question, retriever,article_number,digits=False):
    """calling the research model/function"""

    retriever_dictonnary =  {"KNN" : Nn_model(article_number,digits)}
    retriever = retriever_dictonnary[retriever]
    retriever.clean_data()
    retriever.fit()
    question = preprocessing_user_input(question)
    retriever.predict(question)
    context = retriever.get_articles_text_only()
    parsed_context = retriever.get_articles_parsed() # Liste d'articles
    article_reference = retriever.get_article_reference()

    return context, parsed_context , article_reference

if __name__ == "__main__":


    answer, parsed_context, context,article_reference = get_answer("quelle est la vitesse normale autorisée sur l'autoroute ?", "KNN",2,digits=False)
    print (answer, parsed_context, context, article_reference)
