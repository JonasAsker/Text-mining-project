import lyricsgenius
import os
import glob
import random
import pandas as pd
from langdetect import detect

os.environ["GENIUS_ACCESS_TOKEN"] = "" # insert token

def clean_title(title):
    try:
        return title.split('__')[0].strip()
    except:
        return title

def prep_dataframe(df: pd.DataFrame, subset = []):
    df['track_name'] = df['track_name'].str.replace('-', '__', regex=False) # make search match genius
    df['track_name'] = df['track_name'].str.replace('(', '__', regex=False)
    df['track_name'] = df['track_name'].apply(clean_title)
    df['track_name'] = df['track_name'].str.lower()
    df = df.dropna()
    if subset:
        df = df.drop_duplicates(subset=subset)
    return df

def select_rows(df: pd.DataFrame):
    txt_files = glob.glob('Data/Songs/*.txt')
    with open('notfound.txt', 'r') as f:
        lines = f.readlines()
        not_found_songs = list(map(lambda x: x.split(',')[1].lstrip(' '), lines))
    songs = list(map(lambda x: x.split('/')[2].split('_')[0], txt_files))
    songs.extend(not_found_songs)
    songs = list(map(lambda x: x.lower(), songs))
    filtered_df = df[~df['track_name'].isin(songs)]
    return filtered_df

def get_song_lyrics(df, genius):
    for i, row in df.iterrows():
        artist = row['track_artist']
        title = row['track_name']

        try:
            song = genius.search_song(title = title, 
                                        artist = artist, 
                                        get_full_info=False)
            song_dict = song.to_dict()
            if artist.lower() not in song_dict['artist_names'].lower():
                with open("Data/notfound.txt", 'a') as f:
                    f.write(f"Song not found, {title}, got artist {song_dict['artist_names']}, expected artist {artist}" + '\n')
                    continue

        except:
            with open("Data/notfound.txt", 'a') as f:
                f.write(f"Song not found error, {title}, artist {artist}" + '\n')
                continue

        if '/' in title or '\\' in title or '/' in artist or '\\' in artist: # these characters make the search fail
            with open("Data/notfound.txt", 'a') as f:                     # they cant be replaced because then it
                f.write(f"Weird name, {title}, artist {artist}" + '\n')      # does not match genius
            continue

        sample_size = min(20, len(song.lyrics.split()))
        sample_from = song.lyrics.split()
        sample = ' '.join(random.choices(sample_from, k=sample_size)) # check the language on maximum 20 words of the lyrics

        if detect(sample) != 'en':
            with open("Data/notfound.txt", 'a') as f:
                f.write(f"Song not in english, {title}, artist {artist}" + '\n')
            continue
        file_name =  'Data/Songs/' + ''.join(title.lower()) + '_' + ''.join(artist.lower()) + '.txt'
        with open(file_name, 'w') as f:
            f.write(song.lyrics)

if __name__ == "__main__":
    genius = lyricsgenius.Genius(timeout=1000, retries=5)
    genius.remove_section_headers = True
    df = pd.read_csv('../Data/spotify_songs.csv')
    df = prep_dataframe(df)
    df = select_rows(df, ['track_name', 'track_artist'])
    get_song_lyrics(df, genius)
