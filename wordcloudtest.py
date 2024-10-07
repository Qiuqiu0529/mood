import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('processed_text.csv')
label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

category_text = df.groupby('label')['text'].apply(' '.join).to_dict()
vectorizer = TfidfVectorizer(ngram_range=(1,2),analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
tfidf_matrix = vectorizer.fit_transform(category_text.values())
feature_names = vectorizer.get_feature_names_out()

filtered_category_word_freq = defaultdict(dict)
for idx, (label, _) in enumerate(category_text.items()):
    tfidf_scores = tfidf_matrix[idx].toarray()[0]
    for word, score in zip(feature_names, tfidf_scores):
        if score > 0.1:  # 示例阈值，需根据实际情况调整
            filtered_category_word_freq[label][word] = score

