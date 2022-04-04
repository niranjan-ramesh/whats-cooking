import pandas as pd
import re
import numpy as np

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

dfid = pd.read_csv('complete_food_full_dataset.csv')
dfr = pd.read_csv('output_result.csv')
dfrev = pd.read_csv('food_reviews_full_dataset.csv')
val = dfid['prep_time'].str.split(':').str[1]
dfid.drop("prep_time", axis=1, inplace=True)

time_list = []
for each in val:
    hr = str(each).strip().split(" ")
    if (len(hr)>1):
        hour = hr[0].split("hrs")[0]
        mins = hr[1].split("mins")[0]
        if not mins.isdigit():
            mins = hr[1].split("min")[0]
        if hour.isdigit():
            t2 = int(hour) * 60
            fin = t2 + int(mins)
            time_list.append(str(fin)+"mins")
        else:
            t2 = hr[0].split("hr")[0]
            if t2.isdigit():
                t3 = int(t2) * 60
                fin = t3 + int(mins)
                time_list.append(str(fin)+"mins")
    elif (hr[0] == "1hr"):
        onehr = 60
        time_list.append(str(onehr)+"mins")
    else:
        hour = hr[0].split("hrs")[0]
        if hour.isdigit():
            t2 = int(hour) * 60
            fin = t2 + int(mins)
            time_list.append(str(fin)+"mins")
        else:
            time_list.append(hr[0])

dfid['prep_time'] = time_list

def getminnum(mins):
    return str(mins).split("m")[0]

dfid = dfid.drop(['dishurl'],axis=1)
title = dfr['title'][0:500000]
title = title.tolist()
dfid['recipe_name'] = title

own = dfr['owner'][0:500000]
own = own.tolist()
dfid['contributor_id'] = own

dfid['minutes']= dfid['prep_time'].apply(getminnum)

print ("*** Cleaned Prep Time ***")

nut = dfid[['nutrients']]
dfid.drop("nutrients", axis=1, inplace=True)
nut['nutrients'] = nut['nutrients'].apply(lambda x : x.replace("empty","['abc']"))
nut['nutrients'] =  nut['nutrients'].apply(eval)

n1 = nut['nutrients']
n = []
finnut = []
for i in n1:
    n = []
    for j in i:
        st = j.strip().replace("\n","")
        st1 = re.sub("\s\s+"," ",st)
        n.append(re.sub('[^0-9.]+',"",st1))
    finnut.append(n)

print ("*** Cleaned Nutrients ***")

ing = dfid[['ingredients']]
dfid.drop("ingredients", axis=1, inplace=True)
ing1 = ing[ing['ingredients'] != 'empty']
ing1['ingredients'] =  ing1['ingredients'].apply(eval)
finaling = []
ingfilt = ing1['ingredients'] 

ingsteps = []

for i in ingfilt :
    ingsteps.append(len(i))

for i in ing1['ingredients'] :
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

idx = np.where(ing['ingredients'] == 'empty')
idx1 = list(idx[0])
for each in idx1:
    finaling.insert(each, 'empty ingredients')
    ingsteps.insert(each,"0")

print ("*** Cleaned Ingrediens ***")

dfid = dfid.rename(columns={"recipe":"steps"})
s1 = dfid[['steps']]
s2 = s1[s1['steps'] != 'empty']
s2['steps'] =  s2['steps'].apply(eval)
finalsteps = []
s2filt = s2['steps'] 

for i in s2filt :
    finalsteps.append(len(i))

idx = np.where(s1['steps'] == 'empty')
idx1 = list(idx[0])
for each in idx1:
    finalsteps.insert(each, '0')

print ("*** Cleaned Steps ***")

dfid['ingredients'] = finaling
dfid['nutrients'] = finnut
dfid['n_step'] = finalsteps
dfid['n_ingredients'] = ingsteps
dfid = dfid.rename(columns={"nutrients":"nutrition"})
finaldf = dfid[['recipe_name', 'recipe_id', 'minutes', 'contributor_id', 'nutrition', 'n_step', 'steps',"ingredients","n_ingredients"]]

dfrev = dfrev.rename(columns={"id":"user_id","review_text":"review"})
revfinal = dfrev[['user_id','recipe_id','rating','review']]

finaldf.to_csv('full_food.csv',encoding='utf-8',index=False)
revfinal.to_csv('user_interactions.csv',encoding='utf-8',index=False)