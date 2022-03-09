from dataprep.connector import Connector
import requests
import pandas as pd
import time
import requests
from numpy import *
from requests import get
from bs4 import BeautifulSoup
import asyncio
import math
from itertools import chain


inputs = sys.argv[1]
output = sys.argv[2]
df1 = pd.read_csv(inputs)

# static data
config_file_path = './config'
con = Connector(config_file_path)
renderTemplate1 = 'feedback/partials/reviews'
type1 = 'reviews'
getUserEndpoint1 = '/user-proxy/getreviewbyuserandcontent?mappingType=myReview'
postReactionEndpoint1 = '/user-proxy/savereaction'
postGenerateSignedUrlEndpoint1 = '/user-proxy/generatesignedurl'
postFeedbackPhotoUploadEndpoint1 = '/element-api/content-proxy/generic-photo-upload'
contentType1 = 'recipe'
page1 = '1'
ofContent1 = 'alrcom'

rev_df = pd.DataFrame(columns = ['review_text', 'username', 'rating', 'review_date'])
count = 1

async def fetch_records(queries_to_fetch):
    results = asyncio.gather(*queries_to_fetch)
    df = pd.concat(await results)
    return df

def collect_data(start_page, end_page):
    queries = []
    id_list = []
    for i in range(start_page, end_page):
        print("processing for recipe: ", i)
        val = df1.iloc[i-1].values.flatten().tolist()
        url = val[1]
        id = val[6]

        response = get(url)
        fr = response.text
        html_soup = BeautifulSoup(fr, 'html.parser')
        #rev_count = html_soup.find('a', class_ = "ugc-ratings-link elementFont__detailsLink--underlined ugc-reviews-link")
        rev_count_tag = html_soup.find('span', attrs={'class':'feedback__total'})
        if rev_count_tag :
            #rc = rev_count.text.strip().split(' ')[0]
            rc = rev_count_tag.text
            itemsPerPage1 =  rc
            itemsToRender1 = rc
            id = int(id)
            id_list.append([id] * int(float(rc)))

            a = url.split('recipe/')
            brandval = a[1].split("/")[0]              
            url1 = url

            query = con.query("reviews",
                                renderTemplate = renderTemplate1, type = type1,
                                itemsPerPage = itemsPerPage1, itemsToRender = itemsToRender1,getUserEndpoint = getUserEndpoint1,
                                postReactionEndpoint = postReactionEndpoint1, postGenerateSignedUrlEndpoint = postGenerateSignedUrlEndpoint1,
                                postFeedbackPhotoUploadEndpoint = postFeedbackPhotoUploadEndpoint1, contentType = contentType1, url = url1,
                                page = page1,num = brandval)
            queries.append(query)

    df = asyncio.run(fetch_records(queries))
    return df,id_list

    
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

total_requests = df1.shape[0]
#total_requests = 10
batch_size = 100
batches = math.ceil(total_requests/batch_size)
rev_df = pd.DataFrame()
id1 = []
for i in range(1, batches+1):
    start_page = (i-1)*batch_size + 1
    end_page = i*batch_size
    end_page = end_page if end_page<total_requests else total_requests-1
    df,id_list = collect_data(start_page, end_page)
    ids = list(chain.from_iterable(id_list))
    id1.append(ids) 
    rev_df = pd.concat([rev_df, df])
    print(f"processed for page {start_page}, {end_page}")

ids = list(chain.from_iterable(id1))
rev_df['id'] = ids
rev_df.to_csv('reviews.csv',encoding='utf-8',index=False)