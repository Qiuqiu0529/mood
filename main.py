import pandas as pd

filename = 'text.csv'
data = pd.read_csv(filename)

print(data.head())
category_counts = data['label'].value_counts()
print(category_counts)
