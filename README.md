
# Whats cooking?

  
## Table of Contents

  
- Installation


- Problem Statement

  
- File Structure

  
- Outcomes of Analysis


- Licenses and Acknowledgement


## Installation


Welcome to the DataAffairs Whats Cooking? project. The following serves as a guide to execute the workflow.


## Executing the scripts


Navigate to the DataCollection directory, run the following commands,


### Data Collection:


1. Web Scraper Scripts for Food.com

```
python3 food-scraper.py
```

As there are 700k recipes, the script can be run multiple times by setting the starting and ending position and iteratively icreasing for every run.

```
python3 food-review-scrapper.py -start_pos 0 -end_pos 10 -dataset_num 1
```

Since there are 500k recipes, the script can be run multiple times by setting the starting and ending position and iteratively icreasing for every run.

```
python3 food-recipes.py -start_pos 0 -end_pos 10 -dataset_num 1
```

```
python3 recipes_id_appender.py
```

```
python3 food_etl.py
```

3. Web scraper scripts for Allrecipes.com
```
python3 allrecipes_scrapper.py
```
```
python3 allrecipes_review_scraper.py
```
```
python3 scrapper.py
```
```
python3 allrecipes_etl.py
```
```
python3 allrecipes_formatter.py
```

4. Damn Delicious Dataset
  
  To scrape the dataset from DamnDelicious.net, run the "damn_delicious_scrapper.ipynb" file from start to end. This will save recipes and reviews datasets.
  Alternatively run the following for python execution:
  
```
python3 damn_delicious.py
```

4. Food Network Dataset

To scrape dataset from FoodNetwork.com, (Not .ca), run the complete file named "food_network_scrapper.ipynb" to get the recipes and reviews dataset.

The flow of the scrapper is as follows: 
-> Scrape for the Recipes List for each Initial
-> Scrape Recipes and Reviews list for each initial using the recipes list .
-> Scrape Reviews for each initial using the reviews lists.

The Recipe and Reviews scraping after getting the list takes a lot of time to run, give it good 24 hours time. Alternatively to make it run in less amount of time, follow the steps:

Run the following for each initial
```
python3 foodnetwork_recipes.py "initial_name" 
```

Example:
```
python3 foodnetwork_recipes.py A 
```

Once all the recipes have been scraped, Run the following:
```
python3 foodnetwork_reveiws.py "initial_name" 
```

Example:
```
python3 foodnetwork_reviews.py XYZ
```

5. Once all the datasets have been scraped, to find the common recipes between two datasets, run the following:
```
python3 matching_datasets.py "dataset1_path" "dataset2_path"
```
This will remove the matching recipes between dataset1 and dataset2 from the dataset1.

6. To generate the single common dataset to be used for recommendation, merge the datasets using:

```
python3 merge_datasets.py
```

This will merge the datasets we have scrapped.


### EDA


EDA can be performed by running the "EDA/EDA.ipynb file".


### User Interface


Kindly download and place the contents of [Data](https://drive.google.com/drive/folders/1TAzenFjyOwpMU2wS7g6CC5WC4d-KVUPb?usp=sharing) in a folder named "Data" and [Model](https://drive.google.com/drive/folders/1VZXJQyvU48Udp84QIQ8ecDOk0VvF2uLz?usp=sharing) in a folder named "Model" and place it in the working directory.

```
./ui.sh
```
 

## Problem Statement


In the modern world, time is of the essence and most people require healthy options for their dining needs. A meal recommendation system can help people tackle both these needs, saving people’s time when a system can help them figure out what to cook. Now there is one less worry for people to think about.

1. Given user activity and preference (for example cuisine), can we recommend the best food dishes users must try?

2. For a new user, can we recommend Top highly rated dished of all time?

3. Chosen a recipe, can we recommend similar items to the user?
  

## File Structure


Please find the file structure below,
  

```

├── Data

│ └── CuisinePred

│ └── train.json

├── DataCollection

│ ├── allrecipes_etl.py

│ ├── allrecipes_formatter.py

│ ├── allrecipes_review_scraper.py

│ ├── allrecipes_scraper.py

│ ├── config

│ │ ├── _meta.json

│ │ ├── allrecipes.json

│ │ ├── food.json

│ │ ├── food_reviews.json

│ │ └── reviews.json

│ ├── food-recipes.py

│ ├── food-review-scrapper.py

│ ├── food-scraper.ipynb

│ ├── food-scraper.py

│ ├── food_etl.py

│ ├── recipes_id_appender.py

│ ├── scraping 2.ipynb

│ ├── scraping.ipynb

│ └── scrapper.py

├── EDA

│ └── EDA.ipynb

├── KMeans_analysis.png

├── KMeans_training.ipynb

├── README.md

├── RatingReviewAnalysis

│ └── ratinganalysis.ipynb

├── UI

│ ├── frontend.py

│ └── ui.sh

├── predict_similar_items.py

├── rbm.py

├── rbmpy.py

├── recc.ipynb

├── requirements.txt

├── result.csv

├── utils.py

└── vae.py

```

## Outcome of Analysis


The user interface for the food recommendation system can be found [here] (IP).


## Licenses and Acknowledgement


The data set was scraped from multiple recipe websites including [Food.com](https://www.food.com/?ref=nav), [Allrecipes.com](https://www.allrecipes.com/), [Foodnetwork.com](https://www.foodnetwork.com/) and [DamnDelicious](https://damndelicious.net/). Data processing, storage and visualization are done using Google Cloud Platform and Colab.