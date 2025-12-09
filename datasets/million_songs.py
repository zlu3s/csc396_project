import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('lyrics_mood.csv')
df = df.drop(columns=['artist', 'song', 'link'])
df.rename(columns={'mood': 'label'}, inplace=True)


#labels = ["sad", "joy", "love", "anger", "fear", "surprise"]
label_map = {
    "<pad> sad": 0,
    "<pad> sadness": 0,
    "<pad> joy": 1,
    "<pad> happy": 1,
    "<pad> love": 2,
    "<pad> romantic": 2,
    "<pad> anger": 3,
    "<pad> angry": 3,
    "<pad> fear": 4,
    "<pad> surprise": 5,
    "<pad> excited": 5
}

target_numeric_labels = set(label_map.values())

def adjust_labels(df, label_col, label_map, target_nums):
    df[label_col] = df[label_col].map(label_map)
    
    return df[df[label_col].isin(target_nums)].reset_index(drop=True)

df = adjust_labels(df, "label", label_map, target_numeric_labels)
df.to_csv('msongs_cleaned.csv')

train_df, test_df = train_test_split(df, train_size=0.8)
train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

train_df.to_csv('msongs_train.csv')
test_df.to_csv('msongs_test.csv')
print('million songs dataset processing finished')