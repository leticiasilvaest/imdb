
# carregando as bibliotecas
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



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

st.set_page_config(
    page_title="NLP",
    #page_icon="",
    layout="centered",
    initial_sidebar_state='auto',
    menu_items=None)



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

##### Ngramns #####
        # def ngrams_count(corpus, ngram_range, n=-1, cached_stopwords=stopwords.words('portuguese')):
        #     """
        #     Args
        #     ----------
        #     corpus: text to be analysed [type: pd.DataFrame]
        #     ngram_range: type of n gram to be used on analysis [type: tuple]
        #     n: top limit of ngrams to be shown [type: int, default: -1]
        #     """
            
        #     # Using CountVectorizer to build a bag of words using the given corpus
        #     vectorizer = CountVectorizer(stop_words=cached_stopwords, ngram_range=ngram_range).fit(corpus)
        #     bag_of_words = vectorizer.transform(corpus)
        #     sum_words = bag_of_words.sum(axis=0)
        #     words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        #     words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        #     total_list = words_freq[:n]
            
        #     # Returning a DataFrame with the ngrams count
        #     count_df = pd.DataFrame(total_list, columns=['ngram', 'count'])
        #     return count_df



        # # Splitting the corpus
        #     comments = df2
        #     print(comments)
        #     # Extracting the top 10 unigrams
        #     unigrams_pos = ngrams_count(comments, (1, 1), 10)
            
        #     # Extracting the top 10 biigrams
        #     bigrams_pos = ngrams_count(comments, (2, 2), 10)
            
        #     # Extracting the top 10 trigram
        #     trigrams_pos = ngrams_count(comments, (3, 3), 10)
            
        #     # Joining everything in a python dictionary to make the plots easier
        #     ngram_dict_plot = {
        #         'Top  Comments': unigrams_pos,
        #         'Top Bigrams Comments': bigrams_pos,
        #         'Top Trigrams Comments': trigrams_pos,
        #     }

        #     # Plotting the ngrams analysis
        #     fig, axs = st.subplots(nrows=3, ncols=2, figsize=(15, 18))
        #     i, j = 0, 0
        #     colors = ['Blues_d', 'Reds_d']
        #     for title, ngram_data in ngram_dict_plot.items():
        #         ax = axs[i, j]
        #         st.barplot(x='count', y='ngram', data=ngram_data, ax=ax, palette=colors[j])
                
        #         # Customizing plots
        #         #format_spines(ax, right_border=False)
        #         ax.set_title(title, size=14)
        #         ax.set_ylabel('')
        #         ax.set_xlabel('')
                
        #         # Incrementing the index
        #         j += 1
        #         if j == 2:
        #             j = 0
        #             i += 1
        #     st.tight_layout()
        #     st.show()




        ##########################


#Inserindo dado novo
col1,col2,col3= st.columns(3)
col1,col2,col3= st.columns(3)
col1,col2,col3= st.columns(3)
Dado_novo= st.text_input("ou cole/digite um comentário", key="dado_novo")
Dado_novo = Dado_novo.split(',')
print(Dado_novo)
    
if Dado_novo is not None:

    #### funções ####
        def re_breakline(text_list):
            return [re.sub('[\n\r]', ' ', r) for r in text_list]
        reviews = Dado_novo
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
        print(reviews_stemmer)


        #carregando o modelo de predição
        modelo = pickle.load(open('3modelo20220127.pkl','rb'))

        #predição do modelo
        y_pred = modelo.predict(reviews_stemmer)
        #print(y_pred)

        #st.write(y_pred)
        unique, counts = np.unique(y_pred, return_counts= True)
        result = np.column_stack((unique, counts))
        print(result)
        print ("o Result é: ", result[0])
        print("tamanho de result: ", len(result))

        #bora começar a testar o negativo
  
        
        if len(result) == 2:
            negativos = result[0][1]
            positivos = result[1][1]
            print("Sucesso negativo e positivo")
            print("mensagem negativa e positiva", negativos, positivos)
        else:
            if result[0][0] == 0:
                negativos = result[0][1]
                positivos = 0
                print("Sucesso negativo!")
                print("mensagem negativa ", negativos)
                print("mensagem positiva ", positivos)
                #nao viajamos, o hotel não deu suporte. nao conseguimos viajar, péssimo atendimento

            if result[0][0] == 1:
                print("Sucesso positivo")
                negativos = 0
                positivos = result[0][1]
                print("mensagem negativa ", negativos)
                print("mensagem positiva ", positivos)
                #amei a estadia, a alimentação e tudo! obrigada pela oportunidade!

        if negativos > positivos:
            st.write("😖😫😩 Conteúdo negativo")
        elif negativos < positivos:
            st.write('😃😄😁 Conteúdo positivo')
        else:
            st.write("Conteúdo neutro")


