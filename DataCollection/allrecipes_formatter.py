import pandas as pd
import json,re

useless_words = ['bag'\
,'box'\
,'bunch'\
,'can'\
,'cup'\
,'cups'\
,'dash'\
,'dozen'\
,'drop'\
,'envelope'\
,'fluid'\
,'g'\
,'gallon'\
,'head'\
,'inch'\
,'jar'\
,'kg'\
,'kgs'\
,'large'\
,'lb'\
,'lbs'\
,'leaf'\
,'liter'\
,'loaf'\
,'medium'\
,'ml'\
,'ounce'\
,'ounces'\
,'package'\
,'packages'\
,'packet'\
,'packets'\
,'pinch'\
,'pint'\
,'pints'\
,'quart'\
,'quarts'\
,'scoop'\
,'scoops'\
,'sheet'\
,'sheets'\
,'slice'\
,'slices'\
,'small'\
,'sprig'\
,'sprigs'\
,'stalk'\
,'stalks'\
,'tablespoon'\
,'tablespoons'\
,'teaspoon'\
,'teaspoons'\
, 'or'\
,'whole'
]


dfrec = pd.read_csv('final_recipes.csv')
dfrev = pd.read_csv('final_reviews.csv')
dfrec1 = dfrec.drop(["cook","prep","Servings","Yield"], axis=1)
dfrec2 = dfrec1.rename(columns={"title":"recipe_name","id":"recipe_id",
                            "total":"minutes","owner":"contributor_id",
                            "Directions":"steps"})
allrecdf = dfrec2[['recipe_name', 'recipe_id', 'minutes', 'contributor_id', 'Nutrition', 'steps',"Ingredients"]]

ing = allrecdf[['steps']]
allrecdf.drop("steps", axis=1, inplace=True)
ing['steps'] =  ing['steps'].apply(eval)
finaling = []
stepfilt = ing['steps'] 
finalsteps = []
for i in stepfilt :
    finalsteps.append(len(i))

fclean =[]
n = []
for i in stepfilt :
     n = []
     for j in i:
         stepsplit = j.split(":")[1]
         n.append(stepsplit.strip())
     fclean.append(n)

allrecdf['steps'] = fclean
allrecdf['n_steps'] = finalsteps


ing = allrecdf[['Ingredients']]
allrecdf.drop("Ingredients", axis=1, inplace=True)
ing['Ingredients'] =  ing['Ingredients'].apply(eval)
finaling = []
ingfilt = ing['Ingredients'] 
ingsteps = []

for i in ingfilt :
    ingsteps.append(len(i))

for i in ingfilt :
    n = []
    for j in i:
        in1 = re.sub('[^a-z]+'," ",j)
        split = in1.split(" ")
        ing_str = ''
        for s in split:
            if s not in useless_words:
                ing_str = str(ing_str) +" " +str(s)
        n.append(ing_str.strip())
    finaling.append(n)

allrecdf['ingredients'] = finaling
allrecdf['n_ingredients'] = ingsteps


n1 = allrecdf['Nutrition']
allrecdf.drop("Nutrition", axis=1, inplace=True)
nutri = []
n = []
finalnutri  = []
for i in n1:
    temp = str(i).replace("'", '"')
    jval = json.loads(temp)
    st = list(jval.values())
    n = []
    for j in st:
        n.append(re.sub("[mgIUc]","",j))
    finalnutri.append(n)

allrecdf['nutrition'] = finalnutri
allrecdfin = allrecdf[['recipe_name', 'recipe_id', 'minutes', 'contributor_id', 'nutrition', 'n_steps', 'steps',"ingredients","n_ingredients"]]
dfrev = dfrev.rename(columns={"username":"user_id","review_text":"review"})
revfinal = dfrev[['user_id','recipe_id','rating','review']]

allrecdfin.to_csv('full_allrecipes.csv',encoding='utf-8',index=False)
revfinal.to_csv('full_allrecipes_reviews.csv',encoding='utf-8',index=False)