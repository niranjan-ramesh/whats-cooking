from dataprep.connector import connect
from bs4 import BeautifulSoup
import re
import pandas as pd
import asyncio
import math
import time

connect_api = connect("./config", _concurrency=10)

async def fetch_records(queries_to_fetch):
    results = asyncio.gather(*queries_to_fetch)
    df = pd.concat(await results)
    return df


metadata_query = [connect_api.query('allrecipes', page=1)]
metadata = asyncio.run(fetch_records(metadata_query))
result_set = metadata.iloc[0]['totalResults']
perPage = 24
total_requests = math.ceil(result_set/perPage)
print(total_requests)

def extract_recipes_from_html(df):
    print(df.shape)
    recipes_df = pd.DataFrame(columns = ['title', 'dish_url', 'rating', 'users_rated', 'owner', 'owner_url'])
    recipe_data = {'title' : '', 'dish_url' : '', 'rating' : '', 'users_rated' : '', 'owner' : '', 'owner_url': ''}
    for html in df['html']:
        html_soup = BeautifulSoup(html, 'html.parser')
        # print(html_soup)
        recipes_container = html_soup.find_all('div', class_ = 'card__detailsContainer-left')
        for recipe_container in recipes_container:
            recipe_data['title'] = recipe_container.a.text.strip()
            recipe_data['dish_url'] = recipe_container.a['href']
            rating_container = recipe_container.find('div', class_ = 'card__ratingContainer')
            if(rating_container):
                # print(rating_container)
                rating = rating_container.find('span', class_ = 'review-star-text visually-hidden').text
                # recipe_data['rating'] = re.findall("\d+\.\d+", rating)[0]
                recipe_data['rating'] = re.search('Rating:(.*)stars', rating)
                recipe_data['rating'] = recipe_data['rating'].group(1).strip()
                recipe_data['users_rated'] = rating_container.find('span', class_ = 'ratings-count elementFont__details').text.strip()
                
            owner_container = recipe_container.find('a', class_ = 'card__authorNameLink elementFont__details--bold elementFont__detailsLinkOnly--underlined')
            if(owner_container):
                recipe_data['owner_url'] = owner_container['href']
                recipe_data['owner'] = owner_container.span.text.strip()
            # print(recipe_data)
            recipes_df = recipes_df.append(recipe_data, ignore_index=True)
    return recipes_df


def collect_data(start_page, end_page):
    # Connecting to Web API using dataprep
    queries = []
    for i in range(start_page, end_page):
        query = connect_api.query('allrecipes', page=i)
        queries.append(query)

    df = asyncio.run(fetch_records(queries))
    return df

batch_size = 100
batches = math.ceil(total_requests/batch_size)
result_df = pd.DataFrame()
for i in range(1, batches+1):
    start_page = (i-1)*batch_size + 1
    end_page = i*batch_size
    end_page = end_page if end_page<total_requests else total_requests-1
    df = collect_data(start_page, end_page)
    extracted_df = extract_recipes_from_html(df)
    print("extracted_df.shape", extracted_df.shape)
    result_df = pd.concat([result_df, extracted_df])
    print(f"processed for page {start_page}, {end_page}")
    print(result_df.shape)
    time.sleep(5)
    if(extracted_df.shape[0] == 0):
        break
result_df['id'] = result_df.index
print(f"final shape {result_df.shape}")
result_df.to_csv('allrecipes.csv', index=False)
