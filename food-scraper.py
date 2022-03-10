# from threading import Thread
import pandas as pd
# import time
import requests
import math

from time import sleep
from dataprep.connector import connect
import asyncio
import os

if not os.path.exists('source'):
    os.mkdir('source')

NUM_WORKERS = 1000
TOTAL = 52380

async def fetch_records(queries_to_fetch):
    results = asyncio.gather(*queries_to_fetch)
    df = pd.concat(await results)
    return df

food_connector = connect("./food", _concurrency=100)

response = requests.get('https://api.food.com/services/mobile/fdc/search/sectionfront?recordType=Recipe')

total_num = int(response.json()['response']['totalResultsCount'])
total_pages = math.ceil(total_num/10)
print(total_pages)
result = pd.DataFrame()
orig_batch_size = 500
batch_size = orig_batch_size
queries = []


for i in range(1, total_pages + 1):
    if(batch_size == 0):
        df = asyncio.run(fetch_records(queries))
        result = result.append(df)
        print(result.shape)
        queries = []
        batch_size = orig_batch_size
        sleep(0.5)
    query = food_connector.query('food', pn = str(i), recordType='Recipe')
    queries.append(query)
    batch_size = batch_size - 1
    # print(batch_size)

df = asyncio.run(fetch_records(queries))
result = result.append(df)
print(result.shape)
result.to_csv('result.csv', index=False)



