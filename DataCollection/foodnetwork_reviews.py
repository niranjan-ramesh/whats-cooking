# Import statements
import os
import sys
import glob
import json
import time
import pandas as pd
from urllib.request import urlopen


# Get Current Directory Path
PATH = os.getcwd()
# URL to the reviews for recipe
REVIEW_INDEX_URL = "https://api.sni.foodnetwork.com/moderation-chitter-proxy/"\
    + "v1/comments/brand/FOOD/type/recipe/id/"
# Directory for list of reviews
REVIEW_LIST_DIR = PATH + "/datasets/food-network/review-lists/"
# Director for reviews
REVIEW_DIR = PATH + "/datasets/food-network/reviews/"
# Columns for final dataframe
review_columns = ['username', 'rating', 'review_date',
                  'recipe_id', 'review_text']


# Make Directory if does not exist
def make_check_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# Get the response and return json format
def get_response(url):
    response = urlopen(url)
    data_json = json.loads(response.read())
    return data_json


# Parse the Dataframe
def parse_dataframe(df, recipe_id):
    df_reviews = df.copy()
    df_reviews['username'] = df_reviews['userReference']\
        .apply(lambda x: str(x['name']) if x is not None else "None")
    df_reviews['rating'] = df_reviews['authorAssetRatings']\
        .apply(lambda x: str(x[0]['value']) if len(x) > 0 else "None")
    df_reviews['review_date'] = df_reviews['timestamp']
    df_reviews['recipe_id'] = recipe_id
    df_reviews['review_text'] = df_reviews['text']
    df_reviews = df_reviews[review_columns]
    return df_reviews


# Returns the json converted dataframe of reviews of one recipe
def get_reviews(url):
    comments_df = pd.DataFrame()
    has_next = True
    cursor_id = ""
    cursors = []
    while(has_next):
        json_data = get_response(url+"?sort=NEWEST&cursor="+cursor_id)
        page_info = json_data['pageInfo']
        has_next = page_info['hasNextPage']
        try:
            if page_info['endCursor'] not in cursors:
                cursor_id = page_info['endCursor']
                cursors.append(cursor_id)
            else:
                has_next = False
        except Exception:
            cursor_id = ""
        if "comments" in json_data:
            temp_df = pd.DataFrame(json_data['comments'])
            comments_df = pd.concat([comments_df, temp_df], ignore_index=True)
    return comments_df


# Returns the complete dataframe with reviews of one recipe
def read_reviews(recipe_id, recipe_unique_id):
    print("Reading Recipe Id: ", recipe_id)
    parsed_reviews_df = pd.DataFrame()
    try:
        recipe_url = REVIEW_INDEX_URL+recipe_unique_id
        raw_reviews_df = get_reviews(recipe_url)
        if raw_reviews_df.shape[0] > 0:
            parsed_reviews_df = parse_dataframe(raw_reviews_df, recipe_id)
        print("Successfully Read Recipe Id: ", recipe_id)
    except Exception as e:
        print("Error in reading Recipe Id: ", recipe_id)
        print("Error: ", e)
    return parsed_reviews_df


# Reads the filename, executes scraping and saves the result
def read_reviews_by_recipes(file_path, file_name):
    print("Reading reviews for recipes starting with: ", file_name)
    inital_df = pd.read_csv(file_path)
    res_df = pd.DataFrame([], columns=review_columns)
    count = 1
    for row in inital_df.itertuples():
        recipe_id = row.id
        unique_id = row.unique_id
        reviews = read_reviews(recipe_id, unique_id)
        res_df = pd.concat([res_df, reviews], ignore_index=True)
        if count % 200 == 0:
            print("Sleeping for 10 Seconds")
            time.sleep(10)
    res_df.to_csv(REVIEW_DIR+file_name+".csv")
    print("Saved Reviews File for initial: ", file_name)


def merge_all_reviews():
    filenames = glob.glob(REVIEW_DIR+"I.csv")
    all_reviews_df = pd.DataFrame([], columns=review_columns)
    for file in filenames:
        try:
            temp_df = pd.read_csv(file)
            all_reviews_df = pd.concat([all_reviews_df, temp_df],
                                       ignore_index=True)
        except Exception as e:
            print("Error in reading the file: ", file)
            print("Error: ", e)
    all_reviews_df.to_csv("food_network_interactions.csv")


# Main Function
def main(filename):
    make_check_directory(REVIEW_LIST_DIR)
    make_check_directory(REVIEW_DIR)
    # initial_file_name = filename.split('/')[-1].split(".")[0]
    read_reviews_by_recipes(REVIEW_LIST_DIR+filename+".csv", filename)
    # merge_all_reviews()


# Execution
if __name__ == "__main__":
    # Input file name for single execution
    filename = sys.argv[1]
    # Run the program
    main(filename)
