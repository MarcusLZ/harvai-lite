from re import A
#from tkinter.tix import LabelEntry
from urllib import request
import streamlit as st
from streamlit_chat import message
import requests
import os
from decouple import config
import time
from harvai.qa_model import get_answer

ARTICLES = None
START = None
END = None

st.set_page_config(
    #initial_sidebar_state="collapsed",
    layout ="wide",
    page_title="HarvAI",
    page_icon=":desktop_computer:"
)

#load local env variable if not on heroku else load heroku env variable
is_prod = os.environ.get('IS_HEROKU', None)


#headers = {"Authorization": f"Bearer {API_TOKEN}"}


# ------------ Side Bar------------

from PIL import Image
image = Image.open('images/robot_reading.png')
st.sidebar.image(image)
st.sidebar.markdown("<div><h1 style='text-align: center; color: white;'>HarvAI</h1></div>", unsafe_allow_html=True)
st.sidebar.markdown("This chat bot allows find answer on your french traffic regulation question using Python and Streamlit.")
st.sidebar.markdown("To get started : <ol>Write the question you wish to ask in the question bar and press enter</ol>",unsafe_allow_html=True)

# ------------ Parameters------------


st.sidebar.markdown("[Github](https://github.com/MarcusLZ/harvai)")


# ------------ Chat Box ------------


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("Question : ","", key="input")
    return input_text

col1, col_separateur, col2= st.columns((5,1,5))

with col1 :
    user_input = get_text()

    if user_input:

        answer, parsed_context, context, article_reference = get_answer(user_input, "KNN", 4)
        output = {"question": user_input, "answer": answer, "parsed_context" : parsed_context, "context" : context , "article_reference" : article_reference}

        #output = requests.get("http://127.0.0.1:8000/answer", params={'question': user_input, 'retriever' : retriever, 'article_number' : nb_articles})
        print(output)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output['answer']['answer'])


        ARTICLES = output["parsed_context"]
        ARTICLES_REFERENCE = output["article_reference"]
        START = output['answer']['start']
        END = output['answer']['end']

        print(ARTICLES_REFERENCE)
        print(output['answer']['answer'], output['parsed_context'])


    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))


# ------------ Articles Expander ------------


def hightlight(articles, start, end, reference):
    new = ""
    for number, article in enumerate(articles):
        new = new + "<b>" + str(reference[number]) + ":" + "</b> <br>"
        for count, word in enumerate(list(article)):
            if count == int(start) and start > 0:
                new = new + " <mark style='background-color: DodgerBlue;'>" + word
            elif count == int(end) and end > 0:
                new = new + word + "</mark>"
            else:
                new = new + word
        new = new + "<p></p>"
        start = start - len(article)
        end = end - len(article)
    return new

with col2:
    st.markdown("Articles Returned :")
    if ARTICLES is not None:
        st.markdown(hightlight(ARTICLES, START, END, ARTICLES_REFERENCE), unsafe_allow_html=True)
