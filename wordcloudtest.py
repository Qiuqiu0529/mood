import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('processed_text.csv')
label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

category_text = df.groupby('label')['text'].apply(' '.join).to_dict()
vectorizer = TfidfVectorizer(ngram_range=(1,2),analyzer='word',token_pattern=u"(?u)\\b\\w+\\b",max_features=20000)
tfidf_matrix = vectorizer.fit_transform(category_text.values())
feature_names = vectorizer.get_feature_names_out()

filtered_category_word_freq = defaultdict(dict)
for idx, label in enumerate(category_text.keys()):
    tfidf_scores = tfidf_matrix[idx].toarray()[0]
    for word, score in zip(feature_names, tfidf_scores):
        if score > 0.1:
            filtered_category_word_freq[label][word] = score



for label, freq_dict in filtered_category_word_freq.items():
    if not freq_dict:
        print(f"This gourp '{label_mapping[label]}' does not have enough words to generate a word cloud.")
        continue

    wordcloud = WordCloud(
        background_color='white',
        width=800,
        height=600,
        max_words=200,
        max_font_size=100,
        random_state=42,
        scale=2,
        colormap='viridis'
    ).generate_from_frequencies(freq_dict)


    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{label_mapping[label]}", fontsize=20)
    plt.show()