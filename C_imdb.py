
# carregando as bibliotecas
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import re
import nltk
import pickle

#import streamlit.components.v1 as components
import PIL
from PIL import Image
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('rslp')
from nltk import word_tokenize
from nltk.stem import RSLPStemmer
#from nltk.tokenize import word_tokenize
from nltk import FreqDist

import mglearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer





###### NLP ######
st.markdown("Análise de sentimentos")

col1,col2,col3 = st.columns([1,2,3])
col1,col2,col3 = st.columns([1,2,3])
    # col1,col2,col3 = st.columns([1,2,3])
uploaded_file = st.file_uploader("escolha um arquivo *.csv")
if uploaded_file is not None:
        df2 = pd.read_csv(uploaded_file)
        df2=df2['text_pt'].to_list()
        #print(df2) # checar a saída no terminal

        # Funções
        def re_breakline(text_list):
            return [re.sub('[\n\r]', ' ', r) for r in text_list]
        reviews = df2
        reviews_breakline = re_breakline(reviews)
        
        def re_hiperlinks(text_list):
            pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            return [re.sub(pattern, ' link ', r) for r in text_list]
        reviews_hiperlinks = re_hiperlinks(reviews_breakline)
        
        def re_dates(text_list):
            pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
            return [re.sub(pattern, ' data ', r) for r in text_list]
        reviews_dates = re_dates(reviews_breakline)

        def re_money(text_list):
            pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
            return [re.sub(pattern, ' dinheiro ', r) for r in text_list]
        reviews_money = re_money(reviews_dates)

        def re_numbers(text_list):
            return [re.sub('[0-9]+', ' numero ', r) for r in text_list]
        reviews_numbers = re_numbers(reviews_money)

        def re_negation(text_list):
            return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', r) for r in text_list]
        reviews_negation = re_negation(reviews_numbers)

        def re_special_chars(text_list):
            return [re.sub('\W', ' ', r) for r in text_list]
        reviews_special_chars = re_special_chars(reviews_negation)

        def re_whitespaces(text_list):
            white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
            white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
            return white_spaces_end
        reviews_whitespaces = re_whitespaces(reviews_special_chars)

        def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
            return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]
        reviews_stopwords = [' '.join(stopwords_removal(review)) for review in reviews_whitespaces]

        def stemming_process(text, stemmer=RSLPStemmer()):
            return [stemmer.stem(c) for c in text.split()]
        reviews_stemmer = [' '.join(stemming_process(review)) for review in reviews_stopwords]
        #print(reviews_stemmer)

        #carregando o modelo de predição
        modelo = pickle.load(open('model_logistic.pkl','rb'))
        y_pred = modelo.predict(reviews_stemmer)
        total = len(y_pred)
        st.write("Comentários analisados:")
        st.write("Total: ", total)

        negativo = (y_pred ==0).sum()
        print(negativo)
        positivo = (y_pred ==1).sum()
        print(positivo)
        porc_positiva = (positivo/total)*100
        porc_negativa= (negativo/total)*100

        st.write("Positivos (%): ", round(porc_positiva,2))
        st.write("Negativos(%): ", round(porc_negativa,2))
        #st.markdown("*poderá haver um erro de margem de 10pts para cima ou para baixo.")
        col1,col2,col3 = st.columns([1,2,3])
        col1,col2,col3 = st.columns([1,1,4])

    #Inserindo dado novo
    #     col1,col2,col3= st.columns(3)
    #     col1,col2,col3= st.columns(3)
    #     col1,col2,col3= st.columns(3)
    #     Dado_novo= st.text_input("ou cole/digite um comentário", key="dado_novo")
    #     Dado_novo = Dado_novo.split(',')
    #     print(Dado_novo)
    
    #     if Dado_novo is not None:

    # #### funções ####
    #         def re_breakline(text_list):
    #             return [re.sub('[\n\r]', ' ', r) for r in text_list]
    #         reviews = Dado_novo
    #         reviews_breakline = re_breakline(reviews)

    #         def re_hiperlinks(text_list):
    #             pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #             return [re.sub(pattern, ' link ', r) for r in text_list]
    #         reviews_hiperlinks = re_hiperlinks(reviews_breakline)

    #         def re_dates(text_list):
    #             pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    #             return [re.sub(pattern, ' data ', r) for r in text_list]
    #         reviews_dates = re_dates(reviews_breakline)

    #         def re_money(text_list):
    #             pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    #             return [re.sub(pattern, ' dinheiro ', r) for r in text_list]
    #         reviews_money = re_money(reviews_dates)

    #         def re_numbers(text_list):
    #             return [re.sub('[0-9]+', ' numero ', r) for r in text_list]
    #         reviews_numbers = re_numbers(reviews_money)

    #         def re_negation(text_list):
    #             return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', r) for r in text_list]
    #         reviews_negation = re_negation(reviews_numbers)

    #         def re_special_chars(text_list):
    #             return [re.sub('\W', ' ', r) for r in text_list]
    #         reviews_special_chars = re_special_chars(reviews_negation)

    #         def re_whitespaces(text_list):
    #             white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    #             white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    #             return white_spaces_end
    #         reviews_whitespaces = re_whitespaces(reviews_special_chars)

    #         def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    #             return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]
    #         reviews_stopwords = [' '.join(stopwords_removal(review)) for review in reviews_whitespaces]

    #         def stemming_process(text, stemmer=RSLPStemmer()):
    #             return [stemmer.stem(c) for c in text.split()]
    #         reviews_stemmer = [' '.join(stemming_process(review)) for review in reviews_stopwords]
    #         print(reviews_stemmer)


    #     #carregando o modelo de predição
    #     modelo = pickle.load(open('3modelo20220127.pkl','rb'))

    #     #predição do modelo
    #     y_pred = modelo.predict(reviews_stemmer)
    #     #print(y_pred)

    #     #st.write(y_pred)
    #     unique, counts = np.unique(y_pred, return_counts= True)
    #     result = np.column_stack((unique, counts))
    #     print(result)
    #     print ("o Result é: ", result[0])
    #     print("tamanho de result: ", len(result))

    #     #bora começar a testar o negativo
  
        
    #     if len(result) == 2:
    #         negativos = result[0][1]
    #         positivos = result[1][1]
    #         print("Sucesso negativo e positivo")
    #         print("mensagem negativa e positiva", negativos, positivos)
    #     else:
    #         if result[0][0] == 0:
    #             negativos = result[0][1]
    #             positivos = 0
    #             print("Sucesso negativo!")
    #             print("mensagem negativa ", negativos)
    #             print("mensagem positiva ", positivos)
    #             #nao viajamos, o hotel não deu suporte. nao conseguimos viajar, péssimo atendimento

    #         if result[0][0] == 1:
    #             print("Sucesso positivo")
    #             negativos = 0
    #             positivos = result[0][1]
    #             print("mensagem negativa ", negativos)
    #             print("mensagem positiva ", positivos)
    #             #amei a estadia, a alimentação e tudo! obrigada pela oportunidade!

    #     if negativos > positivos:
    #         st.write("😖😫😩 Conteúdo negativo")
    #     elif negativos < positivos:
    #         st.write('😃😄😁 Conteúdo positivo')
    #     else:
    #         st.write("Conteúdo neutro")


