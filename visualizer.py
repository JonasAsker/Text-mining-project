import torch
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

def extract_embeddings(sentences: list):
    vectors = []
    for sen in sentences:
        output = extractor(sen, truncation = True, padding = True)[0]
        vector = torch.tensor(output[0])
        vectors.append(vector)
    return vectors

def plot_tsne(vectors, labels, perplexity=30, n_iter=1000):
    tsne = TSNE(verbose=True, perplexity=perplexity, n_iter=n_iter)
    data = torch.stack(vectors)
    X_embedd = tsne.fit_transform(data)
    data = pd.DataFrame(data=X_embedd, columns=["Dim 1", "Dim 2"])
    data["label"] = labels
    sns.scatterplot(x="Dim 1", y="Dim 2", hue="label", data=data)

def plot_histograms(df: pd.DataFrame, column: str):
    df[column][df['label'] == 'negative'].hist(label='Negative')
    df[column][df['label'] == 'neutral'].hist(alpha=0.8, label='Neutral')
    df[column][df['label'] == 'positive'].hist(alpha=0.6, label='Positive')
    return

def plot_tsne(vectors, labels, perplexity=30, n_iter=1000):
    tsne = TSNE(verbose=True, perplexity=perplexity, n_iter=n_iter)
    X_embedd = tsne.fit_transform(vectors)
    data = pd.DataFrame(data=X_embedd, columns=["Dim 1", "Dim 2"])
    data["label"] = labels
    sns.scatterplot(x="Dim 1", y="Dim 2", hue="label", data=data)

df = pd.read_csv('Data/train.csv')
vectorizer = CountVectorizer()
dim_reduc = TruncatedSVD(n_components=30, n_iter=100)
vectors = vectorizer.fit_transform(df['lyrics'])
print(vectors)
vectors = dim_reduc.fit_transform(vectors)

plot_tsne(vectors, df['label'])

plt.show()

df = pd.read_csv('Data/songs_with_lyrics.csv')
df = df[['track_name',
         'track_artist',
         'valence',
         'danceability',
         'energy',
         'speechiness', 
         'loudness', 
         'tempo',
         'lyrics']]

correlation_list = ['valence',
                    'danceability',
                    'energy',
                    'speechiness', 
                    'loudness', 
                    'tempo']

correlation_df: pd.DataFrame = df[correlation_list]

out = pd.cut(df['valence'], 3, labels=['negative', 'neutral', 'positive'])
df['label'] = out

correlation_df = correlation_df.corr()['valence'][1:]

for column in correlation_list[1:]:
    plt.cla()
    plot_histograms(df, column)
    plt.legend()
    plt.show()

df = pd.read_csv('Data/data.csv')
df['label'].value_counts().plot(kind='bar')
word_counts = []
for lyric in df['lyrics']:
    word_counts.append(len(lyric.split()))

print(df.loc[word_counts.index(max(word_counts))])
print(max(word_counts))
print(len(word_counts))
print(sum(list(map(lambda x: 1 if x < 10  else 0, word_counts))))

scatter_df = pd.DataFrame({'label': df['label'], 
                           'valence': df['valence'], 
                           'word count': word_counts})

sns.scatterplot(y='valence', x='word count', hue='label', data=scatter_df)
model_name = "prajjwal1/bert-tiny"
model_name = "model"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 512)
model = AutoModel.from_pretrained(model_name)
extractor = pipeline("feature-extraction", batch_size=512, model=model, tokenizer=tokenizer)
vectors = extract_embeddings(df['lyrics'].tolist())
print(len(vectors[0]))
plot_tsne(vectors, df['label'])

plt.show()