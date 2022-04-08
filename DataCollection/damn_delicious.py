# Import statements
import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.request import Request, urlopen

# Get Current Directory Path
PATH = os.getcwd()
# URL to the website
INDEX_URL = "https://damndelicious.net/recipe-index/"
# Directory for recipes csv
DIR_PATH = PATH + "/datasets/"
# Create path if doesn't exists
if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)

# Columns for the recipes dataset
recipe_columns = [
    'recipe_id',
    'title',
    'date_posted',
    'yields',
    'prep_time',
    'cook_time',
    'total_time',
    'ratings',
    'ratings_count',
    'n_ingredients',
    'ingredients',
    'n_steps',
    'directions',
    'nutrition',
    'tags'
]
# Columns for reviews dataset
review_columns = [
    "recipe_id",
    "username",
    "review_date",
    "rating",
    "review_text"
]


# Get the HTML Web Content of the Page
def get_web_content(url):
    # Open the url and read
    try:
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webContent = urlopen(request).read().decode('UTF-8')
        return webContent
    except Exception as e:
        print("Error in requesting web page: ", e)
        return False


# Scraping the Recipe Web Page
def read_recipe(url, recipe_id):
    # Empty DataFrame
    empty_df = pd.DataFrame([], columns=recipe_columns)
    # Empty Reviews DataFrame
    empty_reviews_df = pd.DataFrame([], columns=review_columns)
    # Tokens from url
    tokens = url.split('/')
    try:
        # Get the html page
        webContent = get_web_content(url)
        # Parse the html page
        soup = BeautifulSoup(webContent, 'html.parser')
        # Check if page contains a recipe
        recipe_content = soup.find('div', class_='recipe-content')
        if recipe_content is None:
            # Return empty as none found
            print("No Recipe Found in ", tokens[-2])
            return empty_df, False
        # Data for dataframe
        data = []
        data.append(recipe_id)
        review_data = []
        # Title
        title = ""
        title_h1 = soup.find('h1', class_="post-title")
        if title_h1 is not None:
            title = title_h1.text
        data.append(title)
        # Date Posted
        date_posted = ""
        if len(tokens) == 8:
            date_posted = tokens[3]+"/"+tokens[4]+"/"+tokens[5]
            date_posted = datetime.strptime(date_posted, '%Y/%m/%d')\
                .isoformat()
        data.append(date_posted)
        # Yields and Time
        yields = ""
        prep_time = ""
        cook_time = ""
        total_time = ""
        # Get the Post Meta Time Section
        post_meta_time = soup.find('div', {"class": ["time"]})
        if post_meta_time is not None:
            time_p_tags = post_meta_time.find_all('p')
            for time_p in time_p_tags:
                if time_p is not None:
                    strong = time_p.find('strong')
                    if strong is not None:
                        if strong.text[:-1] == "Yield":
                            yields = time_p.span.text
                        if strong.text[:-1] == "prep time":
                            prep_time = time_p.span.text
                        if strong.text[:-1] == "cook time":
                            cook_time = time_p.span.text
                        if strong.text[:-1] == "total time":
                            total_time = time_p.span.text
        # Yields
        data.append(yields)
        # Prep Time
        data.append(prep_time)
        # Cook Time
        data.append(cook_time)
        # Total Time
        data.append(total_time)
        # Rating
        rating_average = ""
        ratings_average_span = soup.find('span',
                                         class_='wprm-recipe-rating-average')
        if ratings_average_span is not None:
            rating_average = ratings_average_span.text.strip()
        data.append(rating_average)
        # Ratings Count
        ratings_count = ""
        ratings_count_span = soup.find('span',
                                       class_='wprm-recipe-rating-count')
        if ratings_count_span is not None:
            ratings_count = ratings_count_span.text.strip()
        data.append(ratings_count)
        # Ingredients
        ingredients = []
        ingredients_li = soup.find_all('li', itemprop="ingredients")
        for ingredient in ingredients_li:
            ingredients.append(ingredient.text.strip())
        data.append(len(ingredients))
        data.append(ingredients)
        # Directions
        directions = []
        directions_div = soup.find('div', class_="instructions")
        if directions_div is not None:
            directions_li = directions_div.find_all('li')
            for direction in directions_li:
                if direction is not None:
                    directions.append(direction.text)
        data.append(len(directions))
        data.append(directions)
        # Reviews
        comments_li = soup.find_all('li', {'class': ['depth-1']})
        if comments_li is not None:
            for comment_li in comments_li:
                review = []
                username = ""
                rating = ""
                review_date = ""
                comments = ""
                comment_meta = comment_li.div.find('div',
                                                   class_="comment-meta")
                if comment_meta is not None:
                    username = comment_meta.find('strong').text
                if comment_meta.a.time is not None:
                    review_date = comment_meta.a.time['datetime']
                comment_cont = comment_li.div.find('div',
                                                   class_="comment-content")
                if comment_cont is not None:
                    comment_rating = comment_cont\
                        .find('div', class_="wpsso-rar")
                    if comment_rating is not None:
                        comment_rating = comment_rating.div
                        if comment_rating is not None:
                            rating = comment_rating.text
                    comment_ps = comment_cont.find_all('p')
                    for p in comment_ps:
                        if p is not None:
                            comment = p.text.strip()
                            if len(comment) > 0:
                                comments = comments + " " + comment
                    review.append(recipe_id)
                    review.append(username)
                    review.append(review_date)
                    review.append(rating)
                    review.append(comments.strip())
                    review_data.append(review)
        # Nutrition
        nutrition = []
        nutrition_div = soup.find('div', class_="wp-nutrition-label")
        if nutrition_div is not None:
            span_left = nutrition_div.find_all('span', class_='f-left')
            for span in span_left:
                if span is not None:
                    pairs = span.text.strip()
                    pairs = pairs.rsplit(" ", 1)
                    nutrition.append([pairs[0], pairs[1].strip()])
        data.append(nutrition)
        # Categories
        categories = []
        categories_a = soup.find_all('a', rel="category")
        for category in categories_a:
            if category is not None:
                categories.append(category.text)
        data.append(categories)
        # Create a dataframe
        recipe_df = pd.DataFrame(data=[data], columns=recipe_columns)
        review_df = pd.DataFrame(data=review_data, columns=review_columns)
        # Processing reviews
        review_df['rating'] = review_df['rating'].str.split(" ").str[1]
        review_df['rating'] = review_df['rating'].fillna("")
        # Success Message
        print("Recipe Loaded successfully: ", tokens[-2])
        # Return statement
        return recipe_df, review_df, True
    except Exception as e:
        print("Error in loading Recipe: ", tokens[-2])
        print("Error in loading Web page: ", e)
        return empty_df, empty_reviews_df, False


# Getting Recipes for each Ingredient
def get_recipes_by_ingredient(url, title_list):
    # Ingredient's recipes DataFrame
    recipes_by_ingredient_df = pd.DataFrame([], columns=recipe_columns)
    reviews_by_ingredient_df = pd.DataFrame([], columns=review_columns)
    try:
        # Get the html page
        webContent = get_web_content(url)
        # Parse the html page
        soup = BeautifulSoup(webContent, 'html.parser')
        # Get all the recipes for the ingredient
        archives = soup.find('div', class_='archives')
        if archives is not None:
            archive_posts = archives.find_all('div', 'archive-post')
            count = 0
            for post in archive_posts:
                if post is not None:
                    post_a = post.find('a')
                    if post_a is not None:
                        title = post_a['title']
                        # Check for duplication of recipes
                        if title not in title_list:
                            recipe_url = post_a['href']
                            recipe_ingredient_df, review_ingredient_df,\
                                success = read_recipe(recipe_url, count)
                            count += 1
                            if success:
                                title_list.append(title)
                                recipes_by_ingredient_df = pd.concat(
                                    [recipes_by_ingredient_df,
                                     recipe_ingredient_df], ignore_index=True)
                                reviews_by_ingredient_df = pd.concat(
                                    [reviews_by_ingredient_df,
                                     review_ingredient_df], ignore_index=True)
    except Exception as e:
        print("Error in loading recipes from page: ", url)
        print("Error: ", e)
    finally:
        return recipes_by_ingredient_df, reviews_by_ingredient_df


# Getting Ingredients List from Index Page
def get_ingredients_list(url):
    # List of Lead Ingredients Title and URL
    ingredients_list = []
    try:
        # Get the html page
        webContent = get_web_content(url)
        # Parse the html page
        soup = BeautifulSoup(webContent, 'html.parser')
        # <Li> List
        ingredients_li = []
        # Get all the archives <UL> Tags
        archivelist_ul = soup.find_all('ul', 'archiveslist')
        for archive in archivelist_ul:
            if archive is not None:
                archive_li = archive.find_all('li')
                ingredients_li = ingredients_li + archive_li
        # Get the Ingredients URL and Title
        for ingredient in ingredients_li:
            if ingredient is not None:
                ingredient_a = ingredient.find('a')
                if ingredient_a is not None:
                    ingredients_list.append([ingredient_a['title'],
                                            ingredient_a['href']])
    except Exception as e:
        print("Error in Loading the index page: ", e)
    finally:
        return ingredients_list


# Merging all ingredients' recipes into one set
def main():
    # The Index of all recipes
    index_url = "https://damndelicious.net/recipe-index/"
    try:
        ingredients_list = get_ingredients_list(index_url)
    except Exception:
        print("Error in getting ingredients list! Terminating..")
        exit()
    # Titles of all recipes we find
    title_list = []
    # Final Dataframe
    damn_delicious_recipes_df = pd.DataFrame([], columns=recipe_columns)
    damn_delicious_reviews_df = pd.DataFrame([], columns=review_columns)
    print("Start..")
    # Get Recipes for each ingredient
    for ingredient in ingredients_list:
        # Ingredient Details
        ingredient_title = ingredient[0]
        ingredient_url = ingredient[1]
        print("##################################################")
        print("Reading recipes for Ingredient: ", ingredient_title)
        print("##################################################")
        try:
            ingredient_recipe_df, ingredient_review_df = \
                get_recipes_by_ingredient(ingredient_url, title_list)
        except Exception as e:
            print("##################################################")
            print("Error in Reading recipes for Ingredient: ",
                  ingredient_title)
            print("Error = ", e)
            print("##################################################")
        print("##################################################")
        print("Completed reading recipes for Ingredient: ", ingredient_title)
        print("##################################################")
        try:
            damn_delicious_recipes_df = pd.concat([damn_delicious_recipes_df,
                                                   ingredient_recipe_df],
                                                  ignore_index=True)
            damn_delicious_reviews_df = pd.concat([damn_delicious_reviews_df,
                                                   ingredient_review_df],
                                                  ignore_index=True)
        except Exception as e:
            print("Error in concating dataframe for ingredient: ",
                  ingredient_title)
            print("Error = ", e)
    # Save the final dataframe as csv file
    damn_delicious_recipes_df.to_csv(DIR_PATH+'damn_delicious_recipes.csv',
                                     index=False)
    damn_delicious_reviews_df.to_csv(DIR_PATH+'damn_delicious_reviews.csv',
                                     index=False)
    print("Finished!")


if __name__ == "__main__":
    main()
