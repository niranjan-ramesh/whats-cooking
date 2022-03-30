import pandas as pd

df = pd.read_csv('result.csv')
dfout = pd.read_csv('complete_food_dataset.csv')

recid = dfid['recipe_id'][0:500000]
dishurl = dfid['dish_url'][0:500000]
recid = recid.tolist()
dishurl = dishurl.tolist()
dfout['recipe_id'] = recid
dfout['dishurl'] = dishurl
dfout.to_csv('complete_food_full_dataset', index=False)