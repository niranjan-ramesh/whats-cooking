import pandas as pd
import glob

def spl(id):
    id1 = id.split("_",5)
    return id1[5]

def rev_text(rev):
    rev1 = rev.split("<p>",1)
    rev2 = rev1[1]
    return (rev2.split("</p>",1)[0])

def urlval(link):
    link1 = link.split("/",4)
    return (link1[4].split("/",1)[0])

li = []
dish = []
all_files = glob.glob("reviews*.csv")
for filename in all_files:
    df = pd.read_csv(filename)
    id = df['id']
    df = df.drop('id',axis = 1)
    df['recipe_id'] = id.apply(spl).to_frame()

    rev = df['review_text']
    df = df.drop('review_text',axis = 1)  
    df['review_text'] = rev.apply(rev_text).to_frame()
    li.append(df)

all_recipe_files = glob.glob("output_dishrec*.csv")
for recipes in all_recipe_files:
    rec_df = pd.read_csv(recipes)
    dish.append(rec_df)

frame = pd.concat(li, axis=0, ignore_index=True)

dish_frame = pd.concat(dish, axis=0, ignore_index=True)
dish_frame = dish_frame.rename(columns={"Dish Name": "title"})

base_df = pd.read_csv("allrecipes.csv")
base_df = base_df.drop_duplicates(subset = 'title')
base_df = base_df.drop('id',axis = 1)
url = base_df['dish_url']
base_df['id'] = url.apply(urlval).to_frame()
idname = base_df[['id','title','owner']]
result = pd.merge(idname, dish_frame, on="title")
res = result.drop_duplicates(subset = 'title')

frame.to_csv('final_reviews.csv',encoding='utf-8',index=False)
res.to_csv('final_recipes.csv',encoding='utf-8',index=False)