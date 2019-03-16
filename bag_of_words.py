#!-*- coding: utf8 -*-
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from collections import Counter
import nltk
from nltk.stem.porter import PorterStemmer
import math

def vetorize_text(text, translate):
    # cria um vetor do tamanho do dicionario com todas as posições zeradas
    vector = [0] * len(translate)
    for word in text:
        if len(word) > 0:
            base_word = stemmer.stem(word)
            if base_word in translate:
                pos = translate[base_word]
                vector[pos] += 1
    return vector

def calculate_frequency_document(word_index, phrases_vectors):
    freq = 0
    for phrase_vector in phrases_vectors:
        if phrase_vector[word_index] > 0:
            freq += 1
    return freq

def normalize_text(phrase_vector, phrases_vectors):
    # cria um vetor do tamanho do dicionario com todas as posições zeradas
    vector = [0] * len(translate)

    # percorre cada posição do vetor que representa a frequencia da palavra na frase
    for i in range(0, len(phrase_vector)):
        if phrase_vector[i] > 0:
            vector[i] = phrase_vector[i] * math.log10(len(phrases_vectors)/calculate_frequency_document(i, phrases_vectors))
    return vector

def normalize_text_l2(phrase_vector, l2_norm):
    vector = [0] * len(translate)            
    for i in range(0,len(phrase_vector)):
        if(phrase_vector[i] > 0):
            vector[i] = phrase_vector[i]/l2_norm
    return vector

def cosine_similarity(phrase_vector1, phrase_vector2):
    cosine = 0
    for word1, word2 in zip(phrase_vector1, phrase_vector2):
        if word1 > 0 and word2 > 0:
            cosine = np.dot(phrase_vector1,phrase_vector2) / (np.linalg.norm(phrase_vector1) * np.linalg.norm(phrase_vector2))
    return cosine

def test_cosine(vectors1, vectors2, value):
    vector_cosine = []
    for i in range(0,len(vectors1)):
        for j in range(0, len(vectors2)):
            cosine = cosine_similarity(vectors1[i], vectors2[j])
            if cosine > value:
                l = [i, j]
                sum = np.sum(l)
                if(sum/2 == i):
                    vector_cosine.append(l)
    return vector_cosine

classifications = pd.read_csv('perguntas.csv', encoding = 'utf-8')

textsQuestion1 = classifications['Pergunta 1'].str.lower()
textsQuestion2 = classifications['Pergunta 2'].str.lower()

# nltk.download('punkt')
textsTokenQuestion1 = [nltk.tokenize.word_tokenize(phrase) for phrase in textsQuestion1]
textsTokenQuestion2 = [nltk.tokenize.word_tokenize(phrase) for phrase in textsQuestion2]

# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words("english")

# remove a raiz das palavras
# nltk.download('rslp')
# stemmer = nltk.stem.RSLPStemmer()
stemmer = PorterStemmer()

# set = conjunto (não permite elemento repetido)
dictionary = set()

for text in textsTokenQuestion1:
    valids = [stemmer.stem(word) for word in text if word not in stopwords and len(word) > 2]
    dictionary.update(valids)

for text in textsTokenQuestion2:
    valids = [stemmer.stem(word) for word in text if word not in stopwords and len(word) > 2]
    dictionary.update(valids)

countWords = len(dictionary)
# associa um indice com cada palavra do dictionary
tuplas = zip(dictionary, range(countWords))

# #criando dictionary {'ajudar':1}
translate = {word:index for word,index in tuplas}

textVectorsQuestion1 = [vetorize_text(text, translate) for text in textsTokenQuestion1]
textVectorsQuestion2 = [vetorize_text(text, translate) for text in textsTokenQuestion2]

# normalizando os dados com L2 Norm
normalVectorQuestion1L2 = [normalize_text_l2(vector, np.linalg.norm(vector)) for vector in textVectorsQuestion1]
normalVectorQuestion2L2 = [normalize_text_l2(vector, np.linalg.norm(vector)) for vector in textVectorsQuestion2]

# normalizando os dados com tf-idf
normalVectorQuestion1 = [normalize_text(vector, textVectorsQuestion1) for vector in textVectorsQuestion1]
normalVectorQuestion2 = [normalize_text(vector, textVectorsQuestion2) for vector in textVectorsQuestion2]

vector_cosine_l2 = test_cosine(normalVectorQuestion1L2, normalVectorQuestion2L2, 0.5)
vector_cosine_tfidf = test_cosine(normalVectorQuestion1, normalVectorQuestion2, 0.5)

taxa_acerto_l2 = len(vector_cosine_l2)/len(normalVectorQuestion1L2)
taxa_acerto_tfidf = len(vector_cosine_tfidf)/len(normalVectorQuestion1)

print("Taxa de acerto L2: {}".format(taxa_acerto_l2))
print("Taxa de acerto TF-IDF: {}".format(taxa_acerto_tfidf))