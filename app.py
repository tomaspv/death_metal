#librerias
import spacy
import string
from spacy.lang.en import English
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import re
import nltk
nltk.download('stopwords')
import spacy.cli
spacy.cli.download("en_core_web_sm")
import streamlit as st

#hacer una funcion generica para obtener las canciones por separado
@st.cache(persist=True, suppress_st_warning=True)
def separar_canciones(df):
    lines = df.readlines()

    # remove /n at the end of each line
    for index, line in enumerate(lines):
          lines[index] = line.strip()

    # guardo cancion por cancion
    texto = ''
    i = 0
    while i <len(lines):
        if((re.search(r'\d.',lines[i])!=None)):
            texto+= ' | '
        else:
            texto+= ' ' + lines[i]
        i = i + 1
    canciones = texto.split(" | ")
    df = pd.DataFrame(canciones)
    return df

@st.cache(persist=True, suppress_st_warning=True)
def load_data():
    cannibal_corpse = open('DATASET/DEATH METAL/CANNIBAL CORPSE lyrics.txt', "r")
    celtic_frost = open('DATASET/DEATH METAL/CELTIC FROST lyrics.txt', "r")
    death = open('DATASET/DEATH METAL/DEATH lyrics.txt', "r")
    jesus_martyr = open('DATASET/DEATH METAL/JESUS MARTYR lyrics.txt', "r")
    master = open('DATASET/DEATH METAL/MASTER lyrics.txt', "r")
    morbid_angel = open('DATASET/DEATH METAL/MORBID ANGEL lyrics.txt', "r")
    possessed = open('DATASET/DEATH METAL/POSSESSED lyrics.txt', "r")
    sodom = open('DATASET/DEATH METAL/SODOM lyrics.txt', "r")
    venom = open('DATASET/DEATH METAL/VENOM lyrics.txt', "r")
    
    avantasia = open('DATASET/OTRO GENERO/AVANTASIA lyrics.txt', "r")
    hammerfall = open('DATASET/OTRO GENERO/HAMMERFALL lyrics.txt', "r")
    iron_maiden = open('DATASET/OTRO GENERO/IRON MAIDEN lyrics.txt', "r")
    rhapsody = open('DATASET/OTRO GENERO/RHAPSODY lyrics.txt', "r")
    slipknot = open('DATASET/OTRO GENERO/SLIPKNOT lyrics.txt', "r")
    twisted_sister = open('DATASET/OTRO GENERO/TWISTED SISTER lyrics.txt', "r")
    
    
    #canciones del genero death metal
    df_death_metal = pd.DataFrame() 
    df_cannibal_corpse = separar_canciones(cannibal_corpse)
    df_celtic_frost = separar_canciones(celtic_frost)
    df_death = separar_canciones(death)
    df_jesus_martyr = separar_canciones(jesus_martyr)
    df_master = separar_canciones(master)
    df_morbid_angel = separar_canciones(morbid_angel)
    df_possessed = separar_canciones(possessed)
    df_sodom = separar_canciones(sodom)
    df_venom = separar_canciones(venom)
    
    #canciones de genero metal pero distinto a death metsl
    df_otros_metal = pd.DataFrame()
    df_avantasia = separar_canciones(avantasia)
    df_hammerfall = separar_canciones(hammerfall)
    df_iron_maiden = separar_canciones(iron_maiden)
    df_rhapsody = separar_canciones(rhapsody)
    df_slipknot = separar_canciones(slipknot)
    df_twisted_sister = separar_canciones(twisted_sister)
    
    cannibal_corpse.close()
    celtic_frost.close()
    death.close()
    jesus_martyr.close()
    master.close()
    morbid_angel.close()
    possessed.close()
    sodom.close()
    venom.close()
    avantasia.close()
    hammerfall.close()
    iron_maiden.close()
    rhapsody.close()
    slipknot.close()
    twisted_sister.close()
    
    #generamos los df
    df_death_metal = df_death_metal.append([df_cannibal_corpse,
                                            df_celtic_frost,
                                            df_death,
                                            df_jesus_martyr,
                                            df_master,
                                            df_morbid_angel,
                                            df_possessed,
                                            df_sodom,df_venom])
    
    df_otros_metal = df_otros_metal.append([df_avantasia,
                                            df_hammerfall,
                                            df_iron_maiden,
                                            df_rhapsody,
                                            df_slipknot,
                                            df_twisted_sister])
    df_death_metal.rename(columns={0:'texto'}, inplace=True)
    df_otros_metal.rename(columns={0:'texto'}, inplace=True)
    
    df_death_metal['death'] = 1
    df_otros_metal['death'] = 0
    
    
    df_metal = df_death_metal.append(df_otros_metal)
    
    #Realizamos un tratamiento al texto de las canciones
    # removemos simbolos de puntuacion
    df_metal['texto']= df_metal['texto'].map(lambda x: re.sub('[,\.!?]', '', x))
    # convertimos a lowercase
    df_metal['texto'] = df_metal['texto'].map(lambda x: x.lower())
    #elimino las filas con *
    patternDel = "\*"
    filter_ast = df_metal['texto'].str.contains(patternDel)
    df_metal = df_metal[~filter_ast]
    
    #elimino las filas con intrumental
    patternDel_2 = ".nstrumental|INSTRUMENTAL"
    filter_inst = df_metal['texto'].str.contains(patternDel_2)
    df_metal = df_metal[~filter_inst]
    
    #elimino las filas con un solo " "
    df_metal = df_metal[df_metal['texto'].map(len) > 1]

    return df_metal

@st.cache(persist=True, suppress_st_warning=True)
def model(df_metal):
    # # Spacy + Pipeline
    
    # Create our list of punctuation marks
    punctuations = string.punctuation
    
    # Create our list of stopwords
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    
    # Load English tokenizer, tagger, parser, NER and word vectors
    parser = English()
    
    # Creating our tokenizer function
    def spacy_tokenizer(sentence):
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = parser(sentence)
    
        # Lemmatizing each token and converting each token into lowercase
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    
        # Removing stop words
        mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
        # return preprocessed list of tokens
        return mytokens
    
    # Custom transformer using spaCy
    class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            # Cleaning Text
            return [clean_text(text) for text in X]
    
        def fit(self, X, y=None, **fit_params):
            return self
    
        def get_params(self, deep=True):
            return {}
    
    # Basic function to clean the text
    def clean_text(text):
        # Removing spaces and converting text into lowercase
        return text.strip().lower()
    
    
    bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    
    X = df_metal['texto'] # the features we want to analyze
    ylabels = df_metal['death'] # the labels, or answers, we want to test against
    
    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state = 0)
    
    
    classifier = MultinomialNB()
    
    # Create pipeline using Bag of Words
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', bow_vector),
                     ('classifier', classifier)])
    
    # model generation
    pipe.fit(X_train,y_train)
    
    return pipe

    
    
def consultar_si_es_death_metal(pipe,letra):
        
    resultado = pipe.predict(letra)
    
    if(resultado[0]==0):
        return ("La cancion para el clasificador NO es del genero Death Metal")
    else:
        return ("La cancion para el clasificador claramente SI es del genero Death Metal ")
    
    

def main():
    df_data = load_data()
    my_model = model(df_data)
    
    st.header("¿La cancion tiene contenido de death metal?")
    #st.subheader("The number of the beast")
    st.image("https://www.nuclearblast.de/static/articles/171/171294.jpg/1000x1000.jpg", width=300)


    #if st.checkbox("show first rows of the data & shape of the data"):
    #    st.write(df_data.head())
    #    st.write(df_data.shape)
        
        
    sentence = st.text_input('Escribi la letra de la cancion y dale enter amiguero!!!:') 

    if sentence:
        st.write(consultar_si_es_death_metal(my_model,sentence))
    
    
    


if __name__ == "__main__":
    main()


