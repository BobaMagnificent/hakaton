import pandas as pd
import os
import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




stops = set(stopwords.words('english'))
stop_words = stopwords.words("english")
morph = MorphAnalyzer()


def text_proc(text):
    t = " ".join(re.findall(r"[a-zA-Z]+", text)).lower().split()
    lemmatized_text = []
    for word in t:
        if word not in stops:
            lemmatized_word = morph.parse(word)[0].normal_form
            lemmatized_text.append(lemmatized_word)
    lemmatized_text2 = " ".join(lemmatized_text)
    return lemmatized_text2


listio = []


os.chdir(r'C:\hacathon\raw_texts')
for id, i in enumerate(os.listdir()):
    with open(i, encoding='ISO-8859-1') as file:
        dictio = {}
        text = file.read()
        dictio["title"] = i.split("_")[0]
        dictio["text"] = text_proc(text)
        listio.append(dictio)
        if id > 45:
            break

big_data = pd.DataFrame(listio)




tfidf = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 3), min_df=3)
tfidf.fit(big_data["text"])
tfidf_matrix = tfidf.transform(big_data["text"])


def get_top_tf_idf_words(tfidf_vector, feature_names, top_n):
    sorted_nzs = np.argsort(tfidf_vector.data)[:-(top_n+1):-1]
    return feature_names[tfidf_vector.indices[sorted_nzs]]


feature_names = np.array(tfidf.get_feature_names())
keywords = []
for i in range(big_data.shape[0]):
    article_vector = tfidf_matrix[i, :]
    words = get_top_tf_idf_words(article_vector, feature_names, 30)
    keywords.append((big_data["title"][i], words))




texts = []
for i, row in big_data.iterrows():
    texts.append(row["title"])


vectorizer = TfidfVectorizer(stop_words={'english'})
X = vectorizer.fit_transform(texts)

Sum_of_squared_distances = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
k_true = 9
model = KMeans(n_clusters=k_true, init='k-means++', max_iter=200, n_init=10)
model.fit(X)

labels = model.labels_
clusters = pd.DataFrame(list(zip(texts, labels)), columns=['title', 'cluster'])
# print(clusters.sort_values(by=['cluster']))

for i in range(k_true):
    print(clusters[clusters['cluster'] == i])

d = {}
for i, row in clusters.iterrows():
    d[row["cluster"]] = []
for k, v in d.items():
    for i, row in clusters.iterrows():
        if k == row["cluster"]:
            v.append(row["title"])

os.chdir('C:\hacathon')
csv = pd.read_csv('movie_meta_data.csv')

sd = dict()
for i, row in csv.iterrows():
    title = row['title']
    rating = row['imdb user rating']
    sd[title] = rating


for i in d:
    for movie in d[i]:
        if movie in sd:
            movie = sd[movie]

for v in d.values():
    al = list()
    for k in sd.keys():
        for name in v:
            if name == k:
                al.append(int(sd[k]))
    v.append(al)
for i in d:
    print(i, sum(d[i][-1])/len(d[i][-1]))

