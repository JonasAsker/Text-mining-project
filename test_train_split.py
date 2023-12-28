import pandas as pd
import numpy as np

def get_even_distribution(df):
    pos_df = df[df['label'] == 'positive']
    neg_df = df[df['label'] == 'negative']
    neut_df = df[df['label'] == 'neutral']
    min_length = min(len(pos_df), len(neg_df), len(neut_df))
    pos_df = pos_df[:min_length]
    neg_df = neg_df[:min_length]
    neut_df = neut_df[:min_length]
    df = pd.concat([pos_df, neg_df, neut_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split_train_test(df):
    test_size = 0.2  # 20% of the dataset
    num_test_samples = int(len(df) * test_size)
    shuffled_indices = np.random.permutation(len(df))
    test_indices = shuffled_indices[:num_test_samples]
    train_indices = shuffled_indices[num_test_samples:]
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    return [train_df, test_df]

df = pd.read_csv('Data/data.csv')
df = get_even_distribution(df)
train_test = split_train_test(df)
train_test[0].to_csv('Data/train.csv', index=False)
train_test[1].to_csv('Data/test.csv', index=False)
