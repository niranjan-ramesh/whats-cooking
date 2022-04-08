# import statements
import pandas as pd
import time
import sys
import ast
import re


class SimilarityJoin:

    def __init__(self, dataframe1, dataframe2):
        self.df1 = dataframe1
        self.df2 = dataframe2

    def preprocess_df(self, df, cols):
        new_df = df.copy()
        # Create an empty column
        new_df['join'] = " "
        # Concatenate the cols to new column
        for col in cols:
            new_df['join'] = new_df["join"] + " " + new_df[col].astype(str)
        # Perform Regex Operation
        new_df['joinKey'] = new_df['join'].\
            apply(lambda x: re.split(r'\W+', x.strip()))
        new_df['joinKey'] = new_df['joinKey'].\
            apply(lambda y: [x for x in y if x != 'nan'])
        new_df.drop('join', axis=1, inplace=True)
        new_df.to_csv('ses.csv')
        return new_df

    def filtering(self, df1, df2):
        new_df1 = df1.copy()
        new_df1 = new_df1[['id', 'joinKey']].copy()
        new_df1['join'] = new_df1['joinKey']
        new_df1 = new_df1.explode('join')
        new_df1 = new_df1.dropna()
        new_df1['combined1'] = new_df1["id"]\
            .astype(str) + ' + ' + new_df1["joinKey"].astype(str)
        new_df1 = new_df1.groupby("join")\
            .agg({'combined1': lambda x: x.tolist()})
        new_df2 = df2.copy()
        new_df2 = new_df2[['id', 'joinKey']].copy()
        new_df2['join'] = new_df2['joinKey']
        new_df2 = new_df2.explode('join')
        new_df2 = new_df2.dropna()
        new_df2['combined2'] = new_df2["id"]\
            .astype(str) + ' + ' + new_df2["joinKey"].astype(str)
        new_df2 = new_df2.groupby("join")\
            .agg({'combined2': lambda x: x.tolist()})
        new_df = pd.merge(new_df1, new_df2, on=["join"])
        new_df = new_df.explode('combined1')
        new_df = new_df.explode('combined2')
        new_df = new_df.drop_duplicates()
        new_df[['id1', 'joinKey1']] = new_df['combined1']\
            .str.split('+', 1, expand=True)
        new_df[['id2', 'joinKey2']] = new_df['combined2']\
            .str.split('+', 1, expand=True)
        new_df['joinKey1'] = new_df['joinKey1']\
            .apply(lambda x: ast.literal_eval(x.strip()))
        new_df['joinKey2'] = new_df['joinKey2']\
            .apply(lambda x: ast.literal_eval(x.strip()))
        result = new_df.copy()
        result = result[['id1', 'joinKey1', 'id2', 'joinKey2']]
        return result

    def verification(self, cand_df, threshold):
        cand_df['intersect'] = cand_df.apply(lambda row:
                                             set(row.joinKey1)
                                             .intersection(set(row.joinKey2)),
                                             axis=1)
        cand_df['union'] = cand_df.apply(lambda row:
                                         set(row.joinKey1) |
                                         set(row.joinKey2), axis=1)
        cand_df['intersect_count'] = cand_df['intersect']\
            .apply(lambda x: len(x))
        cand_df['union_count'] = cand_df['union'].apply(lambda x: len(x))

        cand_df['jaccard'] = cand_df['intersect_count'] /\
            cand_df['union_count']
        result_df = cand_df.copy()
        result_df = result_df[['id1', 'joinKey1', 'id2', 'joinKey2',
                              'jaccard']]
        result_df = result_df[result_df.jaccard > threshold]
        result_df.to_csv('result.csv', index=False)
        return result_df

    def jaccard_join(self, cols1, cols2, threshold):
        pd.set_option("display.max_rows", 10, "display.max_columns", 10)
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print("Before filtering: %d pairs in total" % (self.df1.shape[0]
                                                       * self.df2.shape[0]))
        cand_df = self.filtering(new_df1, new_df2)
        # cand_df.to_csv('candf.csv', index=False)
        print("After Filtering: %d pairs left" % (cand_df.shape[0]))
        result_df = self.verification(cand_df, threshold)
        print("After Verification: %d similar pairs" % (result_df.shape[0]))
        return result_df


# Converts Ingredients column to String
def convert_str(df):
    df['ingredients'] = df['ingredients'].astype(str)
    return df

# Get the scores of the matching pairs
def get_scores(df1, df2):
    scores = pd.DataFrame([])
    for i in range(int(df1.shape[0] / 1000)+1):
        df_1_temp = df1[i*1000:(i+1)*1000]
        for j in range(int(df2.shape[0]/1000)+1):
            print("Iteration: "+str(i)+":"+str(j))
            df_2_temp = df2[j*1000:(j+1)*1000]
            start_time = time.time()
            er = SimilarityJoin(df_1_temp, df_2_temp)
            matches = er.jaccard_join(['ingredients'], ['ingredients'], 0.75)
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Matches Found: ", matches.shape[0])
            scores = pd.concat([scores, matches], ignore_index=True)
    return scores


# Main function
def main(df1_path, df2_path):
    # Read the input dfs
    print("Reading the input csvs ***")
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    # Process Ingredients Column
    print("Processing Ingredients ****")
    processed_df1 = convert_str(df1)
    processed_df2 = convert_str(df2)
    # Getting the scores of matching pairs
    print("Finding Matching Pairs ****")
    score_match = get_scores(processed_df1, processed_df2)
    # Reducing the matching rows
    print("Reducing the rows from first frame ***")
    id_list = score_match['id1'].unique().tolist()
    reduced_df = df1[~df1['id'].isin(id_list)]
    # Writing and replacing the csv
    print("Writing to csv ***")
    reduced_df.to_csv(df1_path)
    print("Done!")


if __name__ == "__main__":
    path_1 = sys.argv[1]
    path_2 = sys.argv[2]
    main(path_1, path_2)
