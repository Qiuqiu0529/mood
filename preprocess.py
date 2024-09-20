import re
import pandas as pd
import time
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.strip()
    text = text.lower()
    return text

filename = 'text.csv'
df = pd.read_csv(filename)
print(f"original:{len(df)}")

start_time = time.time()

df['text'] = df['text'].apply(clean_text)

end_time = time.time()
execution_time = end_time - start_time
print(f"clean text{execution_time}s")

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')

with open('stopwords-en.txt', 'r',encoding='utf-8') as f:
    stop_words = set(f.read().splitlines())
stop_words.update({'feel', 'go', 'one','make', 'know', 'im', 'think','time'})

#print(stop_words)

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = " ".join(filtered_words)
    return text

df['text'] = df['text'].apply(remove_stopwords)

end_time = time.time()
execution_time = end_time - start_time
print(f"remove stopwords{execution_time}s")

stemmer = PorterStemmer()
df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
df['text'] = df['text'].apply(remove_stopwords)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['text'].str.len() >= 15]
df.reset_index(drop=True, inplace=True)

df.sample(10)
print(f"processed:{len(df)}")
label_counts = df['label'].value_counts()
print(label_counts)

plt.bar(label_counts.index, label_counts.values)
plt.title('Label Counts')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4, 5], ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
plt.show()

df.to_csv('processed_text.csv', index=False)

