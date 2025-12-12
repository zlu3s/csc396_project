'''
process the 500songs dataset
'''

import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split

def clean(cell):
    if isinstance(cell, str) and cell.startswith("["):
        try:
            tokens = ast.literal_eval(cell)
            if isinstance(tokens, list):
                return " ".join(tokens)
        except:
            return ''
    return str(cell).strip()

def max_sentiment(series):
    return series.idxmax()

songs = pd.read_csv('500songs.csv')
original_sentiment_cols = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Love']
songs.columns = ['lyrics'] + original_sentiment_cols
songs['lyrics'] = songs['lyrics'].apply(clean)
sentiment_cols = original_sentiment_cols

songs.loc[:, 'label_str'] = songs.loc[:, sentiment_cols].apply(max_sentiment, axis=1)

sentiment_to_id = {
    'Joy': 1,
    'Sadness': 0,
    'Anger': 3,
    'Fear': 4,
    'Surprise': 5,
    'Love': 2
}

# Create the final 'label' column with numeric IDs
songs.loc[:, 'label'] = songs['label_str'].map(sentiment_to_id)

# Delete unnecessary columns
columns_to_drop = sentiment_cols + ['label_str']
songs.drop(columns=columns_to_drop, inplace=True)

songs.to_csv('500songs_cleaned.csv')

train_df, test_df = train_test_split(songs, train_size=0.8)
train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)
train_df.to_csv('500songs_train.csv')
test_df.to_csv('500songs_test.csv')
print('Dataset processing finished')