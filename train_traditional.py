import time
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import csv

import train_util

X_train, X_val, X_test, y_train, y_val, y_test = train_util.split_data()


#tradition

#TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2),analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
X_train_tf = vectorizer.fit_transform(X_train)
X_val_tf = vectorizer.transform(X_val)
X_test_tf = vectorizer.transform(X_test)

f_classif(X_train_tf,y_train)
selector = SelectKBest(f_classif, k=min(20000,X_train_tf.shape[1]))
selector.fit(X_train_tf, y_train)
selected_features = selector.get_support(indices=True)
print("Selected Features:", selected_features)
X_train_tf = selector.transform(X_train_tf)
X_val_tf=selector.transform(X_val_tf)
X_test_tf = selector.transform(X_test_tf)

#Bag of Words
vectorizer_bag=CountVectorizer(ngram_range=(1, 2),analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
X_train_bag = vectorizer_bag.fit_transform(X_train)
X_val_bag = vectorizer_bag.transform(X_val)
X_test_bag = vectorizer_bag.transform(X_test)
# print(X_train_bag)

f_classif(X_train_bag,y_train)
selector1 = SelectKBest(f_classif, k=min(20000,X_train_bag.shape[1]))
selector1.fit(X_train_bag, y_train)
selected_features_bag = selector1.get_support(indices=True)
print("Selected Features:", selected_features_bag)
X_train_bag = selector1.transform(X_train_bag)
X_val_bag = selector1.transform(X_val_bag)
X_test_bag = selector1.transform(X_test_bag)



#old version
# def softmax():
#     title = " Softmax "
#     softmax = LogisticRegression(multinomial='multinomial', solver='lbfgs', max_iter=30)
#     #TF-IDF
#     softmax.fit(X_train_tf, y_train)
#     y_pred_tf = softmax.predict(X_test_tf)
#     accuracy_tf = accuracy_score(y_test, y_pred_tf)
#     report_tf=classification_report(y_test, y_pred_tf)
#     log_metrics(title,"TF-IDF",accuracy_tf,report_tf)
#     print(accuracy_tf)
#     print(report_tf)
#     draw_confusion_matrix_hitmap(y_test, y_pred_tf,title+"TF-IDF")
#
#     #Bag of Words
#     softmax = LogisticRegression(multinomial='multinomial', solver='lbfgs', max_iter=30)
#     softmax.fit(X_train_bag, y_train)
#     y_pred_bag = softmax.predict(X_test_bag)
#     accuracy_bag = accuracy_score(y_test, y_pred_bag)
#     report_bag=classification_report(y_test, y_pred_bag)
#     log_metrics(title,"Bag of Words",accuracy_bag,report_bag)
#     print(accuracy_bag)
#     print(report_bag)
#     draw_confusion_matrix_hitmap(y_test, y_pred_bag, title+"Bag of Words")
#     draw_accuracy_comparison(accuracy_tf, accuracy_bag,title)

# start_time = time.time()
# softmax()
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Softmax{execution_time}s")

#new
models = {
    'Softmax': {
        'model': LogisticRegression( solver='saga', max_iter=500),# default multi_class='multinomial',
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2']
        }
    },
    'Naive Bayes': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1, 0.5, 1.0]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    }
    # 'SVM': {
    #     'model': SVC(),
    #     'params': {
    #         'C': [0.1, 1, 10],
    #         'kernel': ['linear', 'rbf'],
    #         'gamma': ['scale', 'auto']
    #     }
    # } too slow
}


def train_and_evaluate_model(model_name, model, params, X_train, y_train, X_val, y_val, feature_type):
    print(f"Training {model_name} with {feature_type} features...")
    start_time = time.time()
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Best parameters for {model_name}: {grid.best_params_}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy for {model_name} with {feature_type} features: {acc}")
    class_report = classification_report(y_val, y_pred)
    print(f"Classification Report:\n{class_report}")

    # Draw confusion matrix
    title = f"{model_name} ({feature_type})"
    cm_path = f"{train_util.default_save_path}/{model_name}_{feature_type}_val_confusion_matrix.png"
    train_util.draw_confusion_matrix_hitmap(y_val, y_pred, title, cm_path)


    # Log metrics
    train_util.log_metrics(model_name, feature_type, acc, class_report, train_time, params=grid.best_params_,
                           cm_path=cm_path)
    return acc

acc_tf = {}
acc_bag = {}

for model_name, model_info in models.items():
    acc = train_and_evaluate_model(model_name, model_info['model'], model_info['params'],
                                   X_train_tf, y_train, X_val_tf, y_val, 'TF-IDF')
    acc_tf[model_name] = acc


for model_name, model_info in models.items():
    acc = train_and_evaluate_model(model_name, model_info['model'], model_info['params'],
                                   X_train_bag, y_train, X_val_bag, y_val, 'Bag-of-Words')
    acc_bag[model_name] = acc

train_util.draw_accuracy_comparison(acc_tf, 'TF-IDF',f"{train_util.default_save_path}/TF-IDF_accuracy_comparison.png")
train_util.draw_accuracy_comparison(acc_bag, 'Bag-of-Words',f"{train_util.default_save_path}/Bag-of-Words_accuracy_comparison.png")

best_model_name_tf = max(acc_tf, key=acc_tf.get)
best_model_name_bag = max(acc_bag, key=acc_bag.get)

print(f"Best model using TF-IDF features: {best_model_name_tf} with accuracy {acc_tf[best_model_name_tf]}")
print(f"Best model using Bag-of-Words features: {best_model_name_bag} with accuracy {acc_bag[best_model_name_bag]}")

if acc_tf[best_model_name_tf] >= acc_bag[best_model_name_bag]:
    feature_type = 'TF-IDF'
    best_model_info = models[best_model_name_tf]
    X_train_final = X_train_tf
    X_test_final = X_test_tf
    best_model_name = best_model_name_tf
else:
    feature_type = 'Bag-of-Words'
    best_model_info = models[best_model_name_bag]
    X_train_final = X_train_bag
    X_test_final = X_test_bag
    best_model_name = best_model_name_bag

print(f"Evaluating the best model {best_model_name} with {feature_type} features on the test set...")
grid = GridSearchCV(best_model_info['model'], best_model_info['params'], cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_final, y_train)
best_model = grid.best_estimator_
y_pred_test = best_model.predict(X_test_final)
test_acc = accuracy_score(y_test, y_pred_test)
test_class_report = classification_report(y_test, y_pred_test)
print(f"Test Accuracy for {best_model_name} with {feature_type} features: {test_acc}")
print(f"Test Classification Report:\n{test_class_report}")

test_cm_path = f"{train_util.default_save_path}/{best_model_name}_{feature_type}_test_confusion_matrix.png"
train_util.draw_confusion_matrix_hitmap(y_test, y_pred_test, title=f"{best_model_name} ({feature_type}) Test Set", save_path=test_cm_path)

train_util.log_metrics(best_model_name, f"{feature_type} Test", test_acc, test_class_report, train_time=0, params=grid.best_params_, test_acc=test_acc, cm_path=test_cm_path)