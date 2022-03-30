from time import sleep
from dataprep.connector import connect

import pandas as pd
import asyncio
import math
import requests

async def fetch_records(queries_to_fetch):
        results = asyncio.gather(*queries_to_fetch)
        df = pd.concat(await results)
        return df

df = pd.read_csv('result.csv')
start = 0
end = 500000
rec_id = df['recipe_id'][start:end]
result = pd.DataFrame()
count = 0
for recipeId in rec_id:
	print ("Recipe Number ---> ", count)
	count = count + 1
	try:
	    food_connector = connect("./config", _concurrency=50)
	    INITIAL_URL = 'https://api.food.com/external/v1/recipes/{}/feed/reviews?pn=1'.format(recipeId)
	    response = requests.get(INITIAL_URL)
	    total_num = int(response.json()['total'])
	    total_pages = math.ceil(total_num/20)
	    orig_batch_size = 300
	    batch_size = orig_batch_size
	    queries = []
	    for i in range(1, total_pages + 1):
	        if(batch_size == 0):
	            df = asyncio.run(fetch_records(queries))
	            result = result.append(df)
	            queries = []
	            batch_size = orig_batch_size
	            sleep(0.5)
	        query = food_connector.query('food_reviews', pn = str(i), recipeId=recipeId)
	        queries.append(query)
	        batch_size = batch_size - 1
	    df = asyncio.run(fetch_records(queries))
	    result = result.append(df)

	except Exception as e:
		print (e)


result.to_csv('food_reviews_full_dataset.csv', index=False)
