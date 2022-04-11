# Import statements
import os
import sys
import json
import time
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

# Get Current Directory Path
PATH = os.getcwd()
# URL to the website
INDEX_URL = "http://www.foodnetwork.com/recipes/recipes-a-z"
# Directory for list of recipes
RECIPE_LIST_DIR = PATH + "/datasets/food-network/recipe-lists/"
# Directory for recipes csv
RECIPE_DIR = PATH + "/datasets/food-network/recipes/"
# Directory for list of reviews
REVIEW_LIST_DIR = PATH + "/datasets/food-network/review-lists/"


# Make Directory if does not exist
def make_check_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# Returns web page contents
def get_web_content(url):
    # Open the url and read
    try:
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webContent = urlopen(request).read().decode('UTF-8')
        return webContent
    except Exception as e:
        print("Error in requesting web page: ", e)
        raise


# Columns for the final dfs
recipe_columns = ['id', 'name', 'time_taken', 'servings', 'ingredients',
                  'directions', 'categories']
review_columns = ['id', 'unique_id']


# Reading each recipe
# Output: Returns a dataframe containing information about the given recipe
def read_recipe(url):

    try:
        # Get the html page
        webContent = get_web_content(url)
        # Parse the html page
        soup = BeautifulSoup(webContent, 'html.parser')

        # Start scraping information

        # Empty data list
        data = []

        recipe_id = url.split('-')[-1]
        data.append(recipe_id)

        # Recipe content
        recipe_content = soup.find('div', class_="content-well")

        # Recipe Info
        recipe_info = recipe_content.find('div', class_="recipeInfo")

        # Recipe Body
        recipe_body = recipe_content.find('div', class_="recipe-body")

        # Title of recipe
        recipe_title = recipe_content.find('span', class_="o-AssetTitle" +
                                           "__a-HeadlineText")
        if recipe_title is not None:
            recipe_title = recipe_title.text
        else:
            recipe_title = ""
        data.append(recipe_title)

        # Time taken to cook recipe
        recipe_time = recipe_info\
            .find('span', class_="o-RecipeInfo__a-Description" +
                  " m-RecipeInfo__a-Description--Total")
        if recipe_time is not None:
            recipe_time = recipe_time.text.strip()
        else:
            recipe_time = ""
        data.append(recipe_time)

        # Total servings per recipe
        recipe_yield = recipe_info.find('ul', class_="o-RecipeInfo__m-Yield")
        if recipe_yield is not None:
            recipe_yield = recipe_yield\
                .find('span', class_="o-RecipeInfo__a-Description")
            if recipe_yield is not None:
                recipe_yield = recipe_yield.text.split(" ")[0].strip()
            else:
                recipe_yield = ""
        else:
            recipe_yield = ""
        data.append(recipe_yield)

        # Ingredients
        recipe_ingredients = []
        recipe_ingredients_body = recipe_body\
            .find('div', class_="o-Ingredients__m-Body")
        if recipe_ingredients_body is not None:
            recipe_ingredients_body = recipe_ingredients_body\
                .find_all('p', class_="o-Ingredients__a-Ingredient")[1:]
            for ingredient in recipe_ingredients_body:
                recipe_ingredient = ingredient\
                    .find('span',
                          class_="o-Ingredients__a-Ingredient" +
                          "--CheckboxLabel").text
                if recipe_ingredient is not None:
                    recipe_ingredients.append(recipe_ingredient)
        data.append(recipe_ingredients)

        # Recipe Steps
        recipe_steps = []
        recipe_description = recipe_body.find('div', class_="o-Method__m-Body")
        if recipe_description is not None:
            recipe_description = recipe_description\
                .find_all('li', class_="o-Method__m-Step")
            for step in recipe_description:
                if step is not None:
                    recipe_steps.append(step.text.strip())
        data.append(recipe_steps)

        # Recipe Categories
        recipe_categories = []
        recipe_body_footer = recipe_content\
            .find('div', class_='recipe-body-footer')
        if recipe_body_footer is not None:
            recipe_category_list = recipe_body_footer\
                .find('div', class_='o-Capsule__m-TagList m-TagList')
            if recipe_category_list is not None:
                recipe_category_list = recipe_category_list\
                    .find_all('a', class_='o-Capsule__a-Tag a-Tag')
                for category in recipe_category_list:
                    if category is not None:
                        recipe_categories.append(category.text.strip())
        data.append(recipe_categories)

        review_df = pd.DataFrame([], columns=review_columns)

        # Get the review - ratings Id
        div = soup.find('div', class_='o-ReviewSummary')
        if div is not None:
            rating_section = div.find('script')
            if rating_section is not None:
                json_data = json.loads(rating_section.text)
                unique_id = json_data['assetId']
                review_df = pd.DataFrame([[recipe_id, unique_id]],
                                         columns=review_columns)

        # Convert the scraped information into dataframes
        recipe_df = pd.DataFrame(data, recipe_columns)
        recipe_df = recipe_df.T

    except Exception:
        print("Error in reading recipe")
        raise

    # Return the created dataframe
    return recipe_df, review_df


# Get recipes for each letter
async def get_recipes_by_file(file_name):
    recipes_file_df = pd.DataFrame([], columns=recipe_columns)
    reviews_file_df = pd.DataFrame([], columns=review_columns)

    input_df = pd.read_csv(file_name)
    title_list = input_df['recipe_title'].values.tolist()
    links_list = input_df['recipe_url'].values.tolist()

    for i in range(len(links_list)):
        if i % 1000 == 0:
            time.sleep(5)
            print("***Sleeping for 5 Seconds***")
        print("Reading " + str(i+1) + " of " + str(len(links_list))
              + " recipes")
        print("Reading recipe: ", title_list[i])
        try:
            recipe, reviews = read_recipe(links_list[i])
            recipes_file_df = pd.concat([recipes_file_df, recipe],
                                        ignore_index=True)
            reviews_file_df = pd.concat([reviews_file_df, reviews],
                                        ignore_index=True)
        except Exception as e:
            print("Error in reading recipe: ", title_list[i])
            print("Error: ", e)

    return recipes_file_df, reviews_file_df


# Handle file reading and writing
async def get_recipes(file_name):
    recipes_df = pd.DataFrame([], columns=recipe_columns)
    reviews_df = pd.DataFrame([], columns=review_columns)
    file = RECIPE_LIST_DIR + file_name + ".csv"
    try:
        file_name = file.split('/')[-1]
        file_name = file_name.split('.')[0].strip()
        print("#########################")
        print("Started Reading Recipes starting with ", file_name)
        print("#########################")
        recipes_df, reviews_df = await get_recipes_by_file(file)
        recipes_df.to_csv(RECIPE_DIR+file_name+".csv")
        reviews_df.to_csv(REVIEW_LIST_DIR+file_name+".csv")
        print("#########################")
        print("Successfully Completed Reading Recipes starting with ",
              file_name)
        print("#########################")
    except Exception as e:
        print("Error in reading filename: ", file_name)
        print("Error: ", e)


# Main Function
async def main(filename):
    make_check_directory(RECIPE_DIR)
    make_check_directory(RECIPE_LIST_DIR)
    make_check_directory(REVIEW_LIST_DIR)
    await get_recipes(filename)

if __name__ == "__main__":
    # Input file name for single execution
    filename = sys.argv[1]
    # Run the program in asynchronous mode
    asyncio.run(main(filename))
