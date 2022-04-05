import pandas as pd
import numpy as np
import ast
import nltk
import gensim
import pickle
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
import time


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
features_df = pd.read_csv('./Data/features_clustered.csv')

# Load the model
model_save_path = './Model/kmeans_model.pkl'
with open(model_save_path, 'rb') as f:
    kmeans_model = pickle.load(f)

# Load Google's pre-trained Word2Vec model, trained on news articles
# from CSIL lab machine
bdenv_loc = Path('/usr/shared/CMPT/big-data')
bdata = bdenv_loc / 'data'
model = gensim.models.KeyedVectors.load_word2vec_format(
    bdata / 'GoogleNews-vectors-negative300.bin.gz', binary=True)

# from local path 
# model = KeyedVectors.load_word2vec_format('./Model/GoogleNews-vectors-negative300.bin.gz', binary=True)



def text_to_wordlist(text, remove_stopwords=True, lemmatize_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return text

def text2vec(text):
    # Get the list of keys available in the model
    vocab_keys = list(model.index_to_key)
    return np.mean([model.get_vector(x, norm=True) for x in text.split() if x in vocab_keys], axis=0).reshape(1,-1)

def get_column_tranformer():
    # Normalizing the ratings and polarity score columns through column transformer
    column_transformer = ColumnTransformer(
        [
        ('ratings_transformer', Normalizer(), ['rating']),
        ('polarity_score_transformer', Normalizer(), ['polarity_score'])
        ],
        remainder='drop')
    return column_transformer

def concat_vectors_with_column_transformer(vector, column_data):
    # vector = np.array(vector)
    vector_reshaped = np.vstack(vector)
    X = np.concatenate((vector_reshaped, column_data),axis=1)
    return X


def get_similar_items(ingredients, items_to_predict=10):
    # ingredients = ['soy sauce', 'rice', 'vinegar', 'sesame oil', 'wasabe powder', 'water']
    ingredients = " ".join(ingredients)
    ratings_df = features_df[features_df['ingredients'].str.contains(ingredients)]
    
    lemmatized = text_to_wordlist(ingredients)
    vectorized = text2vec(lemmatized)
    column_transformer = get_column_tranformer()
    input_x = column_transformer.fit_transform(ratings_df[['rating', 'polarity_score']])
    data_x = concat_vectors_with_column_transformer(vectorized, input_x)
    # prediction
    predicted_cluster = kmeans_model.predict(data_x)
    if predicted_cluster.size != 0:
        predicted_recipes_cluster = features_df[features_df['clusters'] == predicted_cluster[0]]
        predcited_df = predicted_recipes_cluster.sample(n=items_to_predict)
    else:
        print(f"unable to find similar items, so returning random {items_to_predict} samples from dataset")
        predcited_df = features_df.sample(n=items_to_predict)
    # print(list(predcited_df['clusters'].values))
    return list(predcited_df['id'].values)

# imput sample
# ingredients = ['winter squash', 'mexican seasoning', 'mixed spice', 'honey', 'butter', 'olive oil', 'salt']
# start_time = time.time()
# print(get_similar_items(ingredients))
# print("Predicted in: ", time.time()-start_time)


