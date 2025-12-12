'''
process the PERC dataset
'''


import pandas as pd
from sklearn.model_selection import train_test_split


md_perc = pd.read_excel("PERC_mendelly.xlsx")
md_perc = md_perc.rename(columns={"Emotion": "label"})

labels = ["sad", "joy", "love", "anger", "fear", "surprise"]
label_map = {
    "sad": 0,
    "sadness": 0,
    "joy": 1,
    "happy": 1,
    "love": 2,
    "romantic": 2,
    "anger": 3,
    "angry": 3,
    "fear": 4,
    "surprise": 5,
    "excited": 5
}

target_numeric_labels = set(label_map.values())

def adjust_labels(df, label_col, label_map, target_nums):
    df[label_col] = df[label_col].map(label_map)
    
    return df[df[label_col].isin(target_nums)].reset_index(drop=True)

md_perc = adjust_labels(md_perc, "label", label_map, target_numeric_labels)
md_perc.to_csv('perc.csv')

train_df, test_df = train_test_split(md_perc, train_size=0.8)
train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

train_df.to_csv('perc_train.csv')
test_df.to_csv('perc_test.csv')
print('PERC dataset processing finished')
