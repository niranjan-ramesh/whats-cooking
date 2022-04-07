from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys

inputs = "allrecipes.csv"
df1 = pd.read_csv(inputs)
df2 = df1.dish_url
recipes_df = pd.DataFrame(columns = ['Dish Name', 'cook', 'prep', 'total', 'Servings', 'Yield', 'Ingredients', 'Directions' , 'Nutrition'])
recipe_data = {'Dish Name':'', 'cook':'', 'prep':'', 'total':'', 'Servings':'', 'Yield':'', 'Ingredients':'', 'Directions' :'', 'Nutrition':''}
count = 0

for url in df2:
    print ("***** Iteration for dish recipe ------>>>>> ",count)
    response = get(url)
    fr = response.text
    html_soup = BeautifulSoup(fr, 'html.parser')
    dish_name = html_soup.find_all('h1',class_ = 'headline heading-content elementFont__display')
    first_box = html_soup.find_all('div', class_ = 'recipe-meta-item-header elementFont__subtitle--bold elementFont__transformCapitalize')
    first_box_val = html_soup.find_all('div', class_ = "recipe-meta-item-body elementFont__subtitle")
    ing = html_soup.find_all('li', class_ = "ingredients-item")
    dir = html_soup.find_all('li', class_ = "subcontainer instructions-section-item")
    tag = html_soup.find_all('span', attrs={'class':'elementFont__details--bold elementFont__transformCapitalize'})
    nut_val = html_soup.find_all('span', attrs={'class':'nutrient-value'})

    head = []
    data = []
    ingred = []
    val = []

    if (dish_name and first_box and first_box_val and ing and dir and tag and nut_val):
        dname = dish_name[0].text
        recipe_data['Dish Name'] = dname
        a = len(str(first_box).split(","))

        for i in range(a):
            header = first_box[i].text.strip()
            dt = first_box_val[i].text.strip()
            if header == 'cook:':
                recipe_data['cook'] = dt
            if header == 'prep:':
                recipe_data['prep'] = dt
            if header == 'total:':
                recipe_data['total'] = dt
            if header == 'Servings:':
                recipe_data['Servings'] = dt
            if header == 'Yield:':
                recipe_data['Yield'] = dt

        ing_len = len(str(ing).split("li>,"))
        for i in range(ing_len):
            ingred.append(ing[i].label.span.text.strip())

        recipe_data['Ingredients'] = ingred

        dir_len = len(str(dir).split("li>,"))
        for i in range(dir_len):
            steps = dir[i].label.span.text.strip()
            x = dir[i].find('p')
            x = x.text.strip()
            val.append(steps+": "+x)

        recipe_data['Directions'] = val

        x = []
        y = []
        for i in tag:
            x.append(' '.join(i.stripped_strings))

        for i in nut_val:
            y.append(' '.join(i.stripped_strings))

        nut = dict(zip(x, y))
        recipe_data['Nutrition'] = nut
        recipes_df = recipes_df.append(recipe_data, ignore_index=True)
        count = count +1

recipes_df.to_csv('dish_recipes.csv',encoding='utf-8',index=False)