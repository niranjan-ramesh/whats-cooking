## Installation

  

Welcome to the DataAffair's Whats Cooking? project. The following serves as a guide to execute the workflow.

  

## Initializing the environment

1. pip install virtualenv (if you don't already have virtualenv installed)

2. virtualenv whats_cooking to create your new environment (called 'whats_cooking' here)

3. source whats_cooking/bin/activate to enter the virtual environment

4. pip install -r requirements.txt to install the requirements in the current environment

  
  

## Executing the scripts

  
  
  
  

### Data Collection:

Navigate to the DataCollection directory, run the following commands,

  
  

1. Web Scraper Scripts for Food.com

  

```

python3 food-scraper.py

```

  

As there are 700k reviews, the script can be run multiple times by setting the starting and ending position and iteratively icreasing for every run. Sample command has been shown below:

  

```

python3 food-review-scrapper.py -start_pos 0 -end_pos 10 -dataset_num 1

```

  

Since there are 500k recipes, the script can be run multiple times by setting the starting and ending position and iteratively icreasing for every run. Sample command has been shown below:

  

```

python3 food-recipes.py -start_pos 0 -end_pos 10 -dataset_num 1

```

  

```

python3 recipes_id_appender.py

```

  

```

python3 food_etl.py

```

  

2. Web scraper scripts for Allrecipes.com

```

python3 allrecipes_scraper.py

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

  

3. Damn Delicious Dataset

To scrape the dataset from DamnDelicious.net, run the [damn_delicious_scrapper.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/DataCollection/damn_delicious_scrapper.ipynb) file from start to end. This will save recipes and reviews datasets.

Alternatively run the following for python execution:

```

python3 damn_delicious.py

```

After executing python file, return back to ipynb for the ETL Section.
  

4. Food Network Dataset

  

To scrape dataset from FoodNetwork.com, (Not .ca), run the complete file named [food_network_scrapper.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/DataCollection/food_network_scrapper.ipynb) to get the recipes and reviews dataset.

  

The flow of the scrapper is as follows:

-> Scrape for the Recipes List for each letter in the alphabet(initial)

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

  

Once all the recipes have been scraped, Run the following to scrape reviews:

```

python3 foodnetwork_reviews.py "initial_name"

```

  

After all recipes and reviews scraping have been completed, return to the python notebook for merging the scraped parts and processing them.

  

Example:

```

python3 foodnetwork_reviews.py XYZ

```

  

5. Once all the datasets have been scraped, to find the common recipes between two datasets, run the following:

```

python3 matching_datasets.py "dataset1_path" "dataset2_path"

```

This will remove the matching recipes between dataset1 and dataset2 from the dataset1. Similarly, we can do for all 4 datasets.

  

6. To generate the single common dataset to be used for recommendation, merge the datasets by running the notebook [merge_datasets.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/DataCollection/merge_datasets.ipynb).

  
  

This will merge the datasets we have scrapped.

  
  

### EDA

  
  

EDA can be performed by running the "EDA/EDA.ipynb" file.

  

### Item-item prediction with KMeans

##### VADER Sentiment Analysis

1. We have to run the notebook [ratinganalysis.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/RatingReviewAnalysis/ratinganalysis.ipynb) before we can start training the KMeans clustering algorithm

2. The previous step will give us a compound score metric using VADER Sentiment Analyzer, so that the skewed reviews are normalized.

3. This process will create an intermediary output(Data/ratings_with_polarity_score.csv) for the KMeans algorithm.

##### Cuisine Prediction
[Notebook](https://github.com/niranjan-ramesh/whats-cooking/blob/master/CuisinePrediction.ipynb) for Support Vector Classifier implementation to predict the cuisine of a food item based on ingredients.

##### KMeans Clustering

KMeans algorithm can be trained by running the notebook [KMeans_training.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/KMeans_training.ipynb)

##### Restricted Boltzmann Machine
Notebook to train a Restricted Boltzmann machine: [rbm_training.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/rbm_training.ipynb)

##### Variational AutoEncoder
Notebook to train a DeepAutoEncoder: [vae_training.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/vae_training.ipynb)

##### KMeans Prediction

The trained model can be used to predict by calling get_similar_items(ingredients) in predict_similar_items.py

  

### User Interface

  
  

Kindly download and place the contents of [Data](https://drive.google.com/drive/folders/1TAzenFjyOwpMU2wS7g6CC5WC4d-KVUPb?usp=sharing) in a folder named "Data" and [Model](https://drive.google.com/drive/folders/1VZXJQyvU48Udp84QIQ8ecDOk0VvF2uLz?usp=sharing) in a folder named "Model" and place it in the working directory and then run:

```

./ui.sh

```

##### FastAI training for comparison: 
Note book to train a dot product model using FASTAi library: [dotproduct-training.ipynb](https://github.com/niranjan-ramesh/whats-cooking/blob/master/dotproduct-training.ipynb)
