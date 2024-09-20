import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time

df = pd.read_csv('processed_text.csv')
label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

start_time = time.time()
grouped_text = df.groupby('label')
end_time = time.time()
execution_time = end_time - start_time
print(f"group text{execution_time}s")

for label, group in grouped_text:
    text1 = ' '.join(group['text'].tolist())
    wordcloud = WordCloud(background_color='white').generate(text1)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{label_mapping[label]}')
    plt.show()