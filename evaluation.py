import pandas as pd
import numpy as np

class Evaluation:
    """Evaluation module for the reccomendations"""

    def __init__(self, interactions):
        self.interactions = interactions

    def recall_at_k(self, actual, predictions, k = 10, relevance=2.5):
        recall = 0.0
        for a, p in zip(actual, predictions):
            hits = 0.0
            miss = 0.0
            ratings = zip(a, p)
            for pair in ratings:
                if(pair[0] >= relevance and pair[1] >= relevance):
                    hits += 1
                elif(pair[0] >= relevance):
                    miss +=1
        if(miss == 0):
            return recall
        recall = (hits/miss)
        return recall

    def precision_at_k(self, actual, predictions, k = 10, relevance=2.5):
        precision = 0.0
        total = 0.0
        for a, p in zip(actual, predictions):
            hits = 0.0
            ratings = zip(a, p)
            for pair in ratings:
                if(pair[0] >= relevance and pair[1] >= relevance):
                    hits += 1
                if(pair[1] >= relevance):
                    total += 1
        if(total == 0):
            return precision
        else:
            precision = (hits/ total) 
        return precision  

    def coverage(self, predictions):
        allpredictions = [recipe_id for user_preds in predictions for recipe_id in user_preds]
        n = len(predictions)
        unique_recipes = len(set(allpredictions))
        return (unique_recipes/n) * 100
