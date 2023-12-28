import glob
import re
import pandas as pd
from scraper import prep_dataframe

def clean_song_lyric(lyric: str):
    try:
        split_lyric = lyric.split('\n')
        split_lyric = split_lyric[1:]
        last_line = split_lyric[-1]
        last_line = re.sub(r'\d+', '', last_line)[:-5]
        split_lyric[-1] = last_line
        out = ' '.join(split_lyric)
        out = re.sub(r'\[.*?\]', '', out)
        return out
    except:
        return None

df = prep_dataframe(pd.read_csv('Data/spotify_songs.csv'), subset=['track_name', 'track_artist'])
df['track_artist'] = df['track_artist'].str.lower()
txt_files = glob.glob('Data/Songs/*.txt')
songs = list(map(lambda x: x.split('/')[2].split('_')[0], txt_files))
artists = list(map(lambda x: x.split('/')[2].split('_')[1][:-4], txt_files))
df['song_artist'] = list(zip(df['track_name'], df['track_artist']))
df.to_csv('Data/songs_test.csv', index=False)
to_drop = list(zip(songs, artists))
df = df[df['song_artist'].isin(to_drop)]
lyrics_df = pd.DataFrame(columns=['song_artist', 'lyrics'])

for i, lyric in enumerate(txt_files):
    with open(lyric, 'r') as f:
        song_lyric = f.read()
        lyrics_df.loc[len(lyrics_df)] = [to_drop[i], song_lyric]

lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(clean_song_lyric)
df = df.merge(lyrics_df, on='song_artist')
df.drop('song_artist', axis=1, inplace=True)
df = df[df['speechiness'] < 0.66] # drop rows which are not music
df.to_csv('Data/songs_with_lyrics.csv', index=False)
df = df[['track_name', 'track_artist', 'valence', 'lyrics']]
df = df.dropna()
out = pd.cut(df['valence'], 3, labels=['negative', 'neutral', 'positive'])
df['label'] = out
df = df[['track_name', 'track_artist', 'valence', 'label', 'lyrics']]
df.to_csv('Data/data.csv', index=False)