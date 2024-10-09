import nltk
from nltk.stem import WordNetLemmatizer
from  train_util import get_train_time_from_log


print(get_train_time_from_log('Softmax','Bag-of-Words'))

# with open('stopwords-en.txt', 'r',encoding='utf-8') as f:
#     stop_words = set(f.read().splitlines())
# stop_words.update({'feel', 'go', 'one','make', 'know', 'im', 'think','time'})
#
# print(stop_words)
# text = "enjoy slouch relax unwind frankli week uni expo start feel bit listless"
# words = text.split()
# filtered_words = [word for word in words if word.lower() not in stop_words]
# filtered_text = " ".join(filtered_words)
#
# print(filtered_text)
