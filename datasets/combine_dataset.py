import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Read the two CSV files
try:
    df_perc = pd.read_csv('perc.csv')
    df_songs = pd.read_csv('500songs_cleaned.csv')
except FileNotFoundError as e:
    print(f"Error: One of the files was not found: {e}")
    # Inspect the available files if there's a FileNotFoundError
    # import os
    # print(os.listdir('.'))

# 2. Inspect the initial DataFrames (optional but good practice)
print("--- df_perc Head ---")
print(df_perc.head())
print("\n--- df_perc Info ---")
df_perc.info()

print("\n--- df_songs Head ---")
print(df_songs.head())
print("\n--- df_songs Info ---")
df_songs.info()

# 3. Standardize Columns and prepare for concatenation

# For df_perc, select the relevant columns and rename 'Poem' to 'text'
df_perc_clean = df_perc[['Poem', 'label']].rename(columns={'Poem': 'text'})

# For df_songs, select the relevant columns and rename 'lyrics' to 'text'
df_songs_clean = df_songs[['lyrics', 'label']].rename(columns={'lyrics': 'text'})

# 4. Concatenate the two DataFrames vertically (stack them)
combined_df = pd.concat([df_perc_clean, df_songs_clean], ignore_index=True)

# 5. Shuffle the resulting combined DataFrame
# frac=1 means return all rows in random order
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Inspect the final DataFrame
print("\n--- Final Shuffled DataFrame Head ---")
print(shuffled_df.head())
print("\n--- Final Shuffled DataFrame Info ---")
shuffled_df.info()

train_df, test_df = train_test_split(shuffled_df, train_size=0.8)
train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)

train_df.to_csv('conbined_train.csv')
test_df.to_csv('conbined_test.csv')
print('Dataset processing finished')