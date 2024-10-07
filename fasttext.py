import fasttext
from itertools import product

from train_util import draw_confusion_matrix_hitmap,split_data

X_train, X_val, X_test, y_train, y_val, y_test = split_data()

def convert_to_fasttext_format(X, y, filepath):
    with open(filepath, 'w') as f:
        for text, label in zip(X, y):
            f.write(f"__label__{label} {text}\n")

convert_to_fasttext_format(X_train, y_train, 'train_fasttext.txt')
convert_to_fasttext_format(X_val, y_val, 'val_fasttext.txt')
convert_to_fasttext_format(X_test, y_test, 'test_fasttext.txt')

params = {
    'lr': [0.1, 0.5, 0.8],
    'epoch': [20, ],#30, 50],
    'wordNgrams': [1,],# 2, 3],
    'dim': [50,],# 100, 200],
    'loss': ['softmax', 'ova'],
}
best_acc = 0
best_params = None

for param_combination in product(*params.values()):
    param_dict = dict(zip(params.keys(), param_combination))
    print(f"Training FastText with params: {param_dict}")
    model = fasttext.train_supervised(input='train_fasttext.txt', **param_dict)
    val_acc = model.test('val_fasttext.txt')[1]
    print(f"Validation Accuracy: {val_acc}")
    if val_acc > best_acc:
        best_acc = val_acc
        best_params = param_dict

print(f"Best Validation Accuracy: {best_acc}")
print(f"Best Parameters: {best_params}")

best_model = fasttext.train_supervised(input='train_fasttext.txt', **best_params)
test_result = best_model.test('test_fasttext.txt')
print(f"Test Accuracy: {test_result[1]}")