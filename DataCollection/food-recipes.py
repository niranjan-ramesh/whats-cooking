from bs4 import BeautifulSoup
import pandas as pd
import requests
import argparse


def get_data(page_source):
    soup = BeautifulSoup(page_source, 'html.parser')
    try:
        prep_time = soup.find('div', class_='recipe-facts__time').text
        left_pane = soup.find('div', class_='recipe-layout__content-left')
        ingredients = left_pane.find_all(
            'li', class_='recipe-ingredients__item')
        ingredients = [ingredient.text for ingredient in ingredients]
        recipe = soup.find_all('li', class_='recipe-directions__step')
        recipe = [step.text for step in recipe]
        nutrients = left_pane.find_all('p', class_='recipe-nutrition__item')
        nutrients = [nutrient.text for nutrient in nutrients]
    except Exception as e:
        print(e)
        prep_time = "empty"
        ingredients = "empty"
        recipe = "empty"
        nutrients = "empty"

    return (prep_time, ingredients, recipe, nutrients)


def scrape_recipe(URL):
    response = requests.get(URL)
    page_source = response.text
    prep_time, ingredients, recipe, nutrients = get_data(page_source)
    return (prep_time, ingredients, recipe, nutrients)


df = pd.read_csv('output_result.csv')
parser = argparse.ArgumentParser()
parser.add_argument("-start_pos", action="store", dest="start_pos", type=int)
parser.add_argument("-end_pos", action="store", dest="end_pos", type=int)
parser.add_argument("-dataset_num", action="store",
                    dest="dataset_num", type=str)
args = parser.parse_args()
start = args.start_pos
end = args.end_pos
datasetnum = args.dataset_num

rec_id = df['recipe_id'][start:end]
url = df['dish_url'][start:end]

prep_times = []
ingredients_s = []
recipes = []
nutrients_s = []

reviews_df = pd.DataFrame()
rec_df = pd.DataFrame()
count = 0

for title, url in zip(rec_id, url):
    print("Iteration ---> ", count)
    count = count + 1
    prep_time, ingredients, recipe, nutrients = scrape_recipe(url)
    prep_times.append(prep_time)
    ingredients_s.append(ingredients)
    recipes.append(recipe)
    nutrients_s.append(nutrients)


rec_df['prep_time'] = prep_times
rec_df['ingredients'] = ingredients_s
rec_df['recipe'] = recipes
rec_df['nutrients'] = nutrients_s
rec_df.to_csv('temp/complete_food_dataset'+datasetnum+'.csv', index=False)
