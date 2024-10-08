import time

import fasttext
from itertools import product
from train_util import draw_confusion_matrix_hitmap,split_data,log_metrics,default_save_path,label_mapping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

X_train, X_val, X_test, y_train, y_val, y_test = split_data()

def convert_to_fasttext_format(X, y, filepath):
    with open(filepath, 'w') as f:
        for text, label in zip(X, y):
            f.write(f"__label__{label} {text}\n")

convert_to_fasttext_format(X_train, y_train, 'fasttext_train.txt')
convert_to_fasttext_format(X_val, y_val, 'fasttext_val.txt')
convert_to_fasttext_format(X_test, y_test, 'fasttext_test.txt')

params = {
    'lr': [0.1, 0.5, 0.8],
    'epoch': [20,30, 50],
    'wordNgrams': [1, 2, 3],
    'dim': [50,  100, 200],
    'loss': ['softmax', 'ova'],
}
best_acc = 0
best_params = None



for param_combination in product(*params.values()):
    param_dict = dict(zip(params.keys(), param_combination))
    print(f"Training FastText with params: {param_dict}")
    starttime = time.time()
    model = fasttext.train_supervised(input='fasttext_train.txt', **param_dict)
    traintime = time.time() - starttime
    val_acc = model.test('fasttext_val.txt')[1]
    print(f"Validation Accuracy: {val_acc}")
    if val_acc > best_acc:
        best_acc = val_acc
        best_params = param_dict

    log_metrics(
        model_name='FastText',
        feature_type='Word Embeddings, Character n-grams',
        acc=val_acc,
        test_acc=None,
        class_report='N/A',
        train_time=traintime,  # 这里可以添加更精确的训练时间记录
        params=param_dict,
        cm_path=None,
        csv_file='training_log_fasttext.csv'
    )

print(f"Best Validation Accuracy: {best_acc}")
print(f"Best Parameters: {best_params}")

starttime=time.time()
best_model = fasttext.train_supervised(input='fasttext_train.txt', **best_params)
traintime=time.time()-starttime

test_result = best_model.test('fasttext_test.txt')
test_acc = test_result[1]
print(f"Test Accuracy: {test_acc}")

test_pred = []
true_labels = []
with open('fasttext_test.txt', 'r') as f:
    for line in f:
        split_line = line.split(' ', 1)
        true_label = int(split_line[0].replace('__label__', '').strip())
        text = split_line[1].strip()
        pred_label = best_model.predict(text)[0][0].replace('__label__', '')
        test_pred.append(int(pred_label))
        true_labels.append(true_label)

test_class_report = classification_report(true_labels, test_pred, target_names=list(label_mapping.values()))
print(f"Test Classification Report:\n{test_class_report}")

test_cm_path = f"{default_save_path}/FastText_Test_Confusion_Matrix.png"
draw_confusion_matrix_hitmap(true_labels, test_pred, title='FastText Test Set', save_path=test_cm_path)

log_metrics(
    model_name='FastText',
    feature_type='Word Embeddings, Character n-grams',
    acc=best_acc,
    class_report=test_class_report,
    train_time=traintime,
    params=best_params,
    test_acc=test_acc,
    cm_path=test_cm_path,
    csv_file='training_log_fasttext.csv'
)

log_metrics(
    model_name='FastText',
    feature_type='Word Embeddings, Character n-grams',
    acc=best_acc,
    class_report=test_class_report,
    train_time=traintime,
    params=best_params,
    test_acc=test_acc,
    cm_path=test_cm_path,
)
