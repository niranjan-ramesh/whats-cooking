import pandas as pd
import numpy as np
import random

def get_items():
    pass

def get_iteractions(users = 12000, recipes = 15000, size = 15000, k=200000):

    reviews = pd.read_csv('Data/user_interactions.csv')
    
    interactions = reviews.sample(k)
    recipes_count = interactions[['recipe_id', 'rating']].groupby('recipe_id', as_index=False).size() \
        .sort_values('size', ascending=False)
    pop_recipes = recipes_count['recipe_id'].values[:recipes]
    
    temp = interactions[interactions['recipe_id'].isin(pop_recipes)]
    data = temp.sort_values('user_id').iloc[:size]

    num = len(data['recipe_id'].unique())

    return data, num

def splitter(reviews):
    num = len(reviews['recipe_id'].unique())
    train = reviews.pivot(index='user_id', values='rating', columns='recipe_id').fillna(0)
    total_data = train.values
    random.shuffle(total_data)
    n = len(total_data)
    size_train = int(n * 0.75)
    X_train = total_data[:size_train]
    valid = total_data[size_train:]
    return train, valid, num
