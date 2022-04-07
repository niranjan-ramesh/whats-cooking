import pandas as pd
import glob

dfid = pd.read_csv('output_result.csv')
all_files = glob.glob('temp/complete_food_dataset*.csv')
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

dfout = pd.concat(li, axis=0, ignore_index=True)

recid = dfid['recipe_id'][0:500000]
dishurl = dfid['dish_url'][0:500000]
recid = recid.tolist()
dishurl = dishurl.tolist()
dfout['recipe_id'] = recid
dfout['dishurl'] = dishurl
dfout.to_csv('temp/complete_food_full_dataset', index=False)