import time

import fasttext
import numpy as np
from itertools import product
from train_util import draw_confusion_matrix_hitmap,split_data,log_metrics,default_save_path,label_mapping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,log_loss

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
        val_acc=val_acc,
        class_report='N/A',
        train_time=traintime,  # 这里可以添加更精确的训练时间记录
        params=param_dict,
        csv_file='training_log_fasttext.csv'
    )

print(f"Best Validation Accuracy: {best_acc}")
print(f"Best Parameters: {best_params}")

starttime=time.time()
best_model = fasttext.train_supervised(input='fasttext_train.txt', **best_params)
traintime=time.time()-starttime

def construct_prob_matrix(model, X, label_mapping):
    prob_matrix = []
    for text in X:
        labels, confidences = model.predict(text, k=len(label_mapping))

        prob_vector = np.zeros(len(label_mapping))

        for label, confidence in zip(labels, confidences):
            label_index = int(label.replace('__label__', ''))
            prob_vector[label_index] = confidence
        prob_matrix.append(prob_vector)
    return np.array(prob_matrix)


train_prob_matrix = construct_prob_matrix(best_model, X_train, label_mapping)
train_loss = log_loss(y_train, train_prob_matrix)
train_acc = accuracy_score(y_train, np.argmax(train_prob_matrix, axis=1))

val_prob_matrix = construct_prob_matrix(best_model, X_val, label_mapping)
val_loss = log_loss(y_val, val_prob_matrix)

test_result = best_model.test('fasttext_test.txt')
test_acc = test_result[1]
print(f"Test Accuracy: {test_acc}")

test_prob_matrix = construct_prob_matrix(best_model, X_test, label_mapping)
test_log_loss_value = log_loss(y_test, test_prob_matrix)

test_pred = np.argmax(test_prob_matrix, axis=1)
test_class_report = classification_report(y_test, test_pred, target_names=list(label_mapping.values()))
print(f"Test Classification Report:\n{test_class_report}")


test_cm_path = f"{default_save_path}/FastText_Test_Confusion_Matrix.png"
draw_confusion_matrix_hitmap(y_test, test_pred, title='FastText Test Set', save_path=test_cm_path)


log_metrics(
    model_name='FastText',
    feature_type='Word Embeddings, Character n-grams',
    val_acc=best_acc,
    train_acc=train_acc,
    train_loss=train_loss,
    val_loss=val_loss,
    test_acc=test_acc,
    test_log_loss=test_log_loss_value,
    class_report=test_class_report,
    train_time=traintime,
    params=best_params,
    cm_path=test_cm_path,
    csv_file='training_log_fasttext.csv'
)


log_metrics(
    model_name='FastText',
    feature_type='Word Embeddings, Character n-grams',
    val_acc=best_acc,
    train_acc=train_acc,
    train_loss=train_loss,
    val_loss=val_loss,
    test_acc=test_acc,
    test_log_loss=test_log_loss_value,
    class_report=test_class_report,
    train_time=traintime,
    params=best_params,
    cm_path=test_cm_path,
)
