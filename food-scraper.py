# from threading import Thread
import pandas as pd
# import time
import requests

from threading import Thread
from time import sleep
from dataprep.connector import connect
import asyncio
import os

if not os.path.exists('source'):
    os.mkdir('source')


# titles = []
# owners = []
# dish_urls = []
# ratings = []
# owner_urls = []
# num_ratings = []

# # p = re.compile('var initialData = (.*?);')


# def worker(name):
#     # try:
#     dishes = []
#     URL = 'https://api.food.com/services/mobile/fdc/search/sectionfront?pn={}&recordType=Recipe'.format(name)
#     response = requests.get(URL)
#     # if(name == 1):
#     data = response.json()
#     results = data['response']['results']
#     for dish in results:
#         titles.append(dish['title'])
#         owners.append(dish['main_username'])
#         dish_urls.append(dish['record_url'])
#         ratings.append(dish['main_rating'])
#         owner_urls.append(dish['recipe_user_url'])
#         num_ratings.append(dish['main_num_ratings'])
#     dishes.append(data)
#     # else:
#     #     text = response.text
#     #     print(text)
#     #     responseXml = ET.fromstring(text)
#     #     results = responseXml.find('response').findall('results')
#     #     for dish in results:
#     #         titles.append(dish.find('title'))
#     #         owners.append(dish.find('main_username'))
#     #         dish_urls.append(dish.find('record_url'))
#     #         ratings.append(dish.find('main_rating'))
#     #         owner_urls.append(dish.find('recipe_user_url'))
#     #         num_ratings.append(dish.find('main_num_ratings'))
#     #     dishes.append(data)

#     df = pd.DataFrame({
#         'titles': titles,
#         'owners': owners,
#         'dish_urls': dish_urls,
#         'ratings': ratings,
#         'owner_urls': owner_urls,
#         'num_ratings': num_ratings
#     })
#     df.to_csv('source/{}.csv'.format(name))

# for i in range(1, 52381):
#     if((i%100) == 0):
#         time.sleep(10)
#     t = Thread(target=worker, args=(i, ))
#     t.start()


NUM_WORKERS = 1000
TOTAL = 52380

async def fetch_records(queries_to_fetch):
    results = asyncio.gather(*queries_to_fetch)
    df = pd.concat(await results)
    # df = await query
    df.to_csv('result.csv', index=False)
    # print('Written {} file'.format(i))
    print('Written file')

# def worker(queries_to_fetch, i):
#     # food_connector = connect("./food")
#     # query = food_connector.query('food', pn = str(i), recordType='Recipe')
#     asyncio.run(fetch_records(queries_to_fetch, i))


running = [TOTAL//NUM_WORKERS] * NUM_WORKERS
running.append(TOTAL % NUM_WORKERS)

# print(running)

# for index, runs in enumerate(running):
#     if((index % 100) == 0):
#         sleep(40)
#     queries = []
#     for j in range(1, runs + 1):
#         pn = (index * (TOTAL//NUM_WORKERS)) + j
#         food_connector = connect("./food")
#         query = food_connector.query('food', pn = str(pn), recordType='Recipe')
#         queries.append(query)
#     t = Thread(target=worker, args=(queries, index, ))
#     t.start()

# for index in range(1, TOTAL + 1):
#     if((index%100) == 0):
#         sleep(5)
#     # query = anime_connector.query('food', pn = str(i), recordType='Recipe')
#     t = Thread(target=worker, args=(index, ))
#     t.start()

def worker(n, tot_requests):
    titles = []
    owners = []
    dish_urls = []
    ratings = []
    owner_urls = []
    num_ratings = []
    for j in tot_requests:
        pn = (n * (TOTAL//NUM_WORKERS) + j)
        payload = {'recordType': 'Recipe', 'pn': str(pn)}
        URL = 'https://api.food.com/services/mobile/fdc/search/sectionfront'
        response = requests.get(URL, params=payload)
        data = response.json()
        results = data['response']['results']
        for dish in results:
            titles.append(dish['title'])
            owners.append(dish['main_username'])
            dish_urls.append(dish['record_url'])
            ratings.append(dish['main_rating'])
            owner_urls.append(dish['recipe_user_url'])
            num_ratings.append(dish['main_num_ratings'])
    df = pd.DataFrame({
        'titles': titles,
        'owners': owners,
        'dish_urls': dish_urls,
        'ratings': ratings,
        'owner_urls': owner_urls,
        'num_ratings': num_ratings
    })
    df.to_csv('source/{}.csv'.format(n))


queries = []
food_connector = connect("./food", _concurrency=100)
for index, runs in enumerate(running): # range(1, 52381):
    if((index%100)==0):
        sleep(5)
    t = Thread(target=worker, args=(index, runs, ))
    t.start()
    # query = food_connector.query('food', pn = str(i), recordType='Recipe')
    # queries.append(query)

asyncio.run(fetch_records(queries))
# print('blah')

