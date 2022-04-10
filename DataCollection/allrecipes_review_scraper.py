from dataprep.connector import Connector
import pandas as pd
from numpy import *
from requests import get
from bs4 import BeautifulSoup
import asyncio
import math

df1 = pd.read_csv("allrecipes.csv")

# static data
config_file_path = './config'
con = Connector(config_file_path, _concurrency=100)
renderTemplate1 = 'feedback/partials/reviews'
type1 = 'reviews'
getUserEndpoint1 = '/user-proxy/getreviewbyuserandcontent?mappingType=myReview'
postReactionEndpoint1 = '/user-proxy/savereaction'
postGenerateSignedUrlEndpoint1 = '/user-proxy/generatesignedurl'
postFeedbackPhotoUploadEndpoint1 = '/element-api/\
content-proxy/generic-photo-upload'
contentType1 = 'recipe'
page1 = '1'
ofContent1 = 'alrcom'

rev_df = pd.DataFrame(
    columns=['review_text', 'username', 'rating', 'review_date'])
count = 1


async def fetch_records(queries_to_fetch):
    results = asyncio.gather(*queries_to_fetch)
    df = pd.concat(await results, ignore_index=True)
    return df


async def collect_data(start_page, end_page):
    queries = []
    print(start_page, end_page)
    for i in range(start_page, end_page):
        print("processing for recipe: ", i)
        val = df1.iloc[i].values.flatten().tolist()
        url = val[1]

        response = get(url)
        fr = response.text
        html_soup = BeautifulSoup(fr, 'html.parser')

        rev_count_tag = html_soup.find(
            'span', attrs={'class': 'feedback__total'})
        if rev_count_tag:

            rc = rev_count_tag.text
            itemsPerPage1 = rc
            itemsToRender1 = rc

            a = url.split('recipe/')
            brandval = a[1].split("/")[0]
            url1 = url

            query = con.query(
                "reviews",
                renderTemplate=renderTemplate1,
                type=type1,
                itemsPerPage=itemsPerPage1,
                itemsToRender=itemsToRender1,
                getUserEndpoint=getUserEndpoint1,
                postReactionEndpoint=postReactionEndpoint1,
                postGenerateSignedUrlEndpoint=postGenerateSignedUrlEndpoint1,
                postFeedbackPhotoUploadEndpoint=postFeedbackPhotoUploadEndpoint1,
                contentType=contentType1,
                url=url1,
                page=page1,
                num=brandval
            )
            queries.append(query)

    df = asyncio.run(fetch_records(queries))
    return df

total_requests = df1.shape[0]
batch_size = 100
batches = math.ceil(total_requests/batch_size)
rev_df = pd.DataFrame()
id1 = []
try:
    for i in range(1, batches+1):
        start_page = (i-1)*batch_size
        end_page = i*batch_size
        # end_page = end_page if end_page<total_requests else total_requests-1
        df = asyncio.run(collect_data(start_page, end_page))
        rev_df = pd.concat([rev_df, df])
        print(f"processed for page {start_page}, {end_page}")

except Exception as e:
    print(e)
    rev_df.to_csv('reviews_exception.csv', encoding='utf-8', index=False)
rev_df.to_csv('reviews.csv', encoding='utf-8', index=False)
