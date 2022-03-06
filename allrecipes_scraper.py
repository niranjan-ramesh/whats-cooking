from dataprep.connector import connect
# from datetime import datetime
# import matplotlib.pyplot as plt
import pandas as pd
import asyncio
import math

# "authorization": "Bearer",
# "pagination": {
#             "type": "page",
#             "pageKey": "page",
#             "limitKey": "pageSize",
#             "maxCount": 5000000
#         } 
# "pagination": {
#             "type": "offset",
#             "offsetKey": "f",
#             "limitKey": "h",
#             "maxCount": 100000
#         }   
# Provide your API key here for TAs to reproduce your results
# API_key = "8D73AS9T8U2JKXRP"

connect_api = connect("./config", _concurrency=100)

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


# Connecting to Web API using dataprep
queries = []
for i in range(1, total_requests):
    query = connect_api.query('allrecipes', page=i)
    queries.append(query)

df = asyncio.run(fetch_records(queries))
df.to_csv('allrecipes.csv', sep='\t')