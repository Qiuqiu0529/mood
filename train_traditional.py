import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import csv

df = pd.read_csv('processed_text.csv')
label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
print(len(df))
features = df['text']
labels = df['label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
print("train:", len(X_train))
print("test:", len(X_test))


#TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2),analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)
f_classif(X_train_tf,y_train)
selector = SelectKBest(f_classif, k=min(20000,X_train_tf.shape[1]))
selector.fit(X_train_tf, y_train)
selected_features = selector.get_support(indices=True)
print("Selected Features:", selected_features)
X_train_tf = selector.transform(X_train_tf)
X_test_tf = selector.transform(X_test_tf)

#Bag of Words
vectorizer_bag=CountVectorizer()
X_train_bag = vectorizer_bag.fit_transform(X_train)
X_test_bag = vectorizer_bag.transform(X_test)
print(X_train_bag)
f_classif(X_train_bag,y_train)
selector1 = SelectKBest(f_classif, k=min(20000,X_train_bag.shape[1]))
selector1.fit(X_train_bag, y_train)
selected_features_bag = selector1.get_support(indices=True)
print("Selected Features:", selected_features_bag)
X_train_bag = selector1.transform(X_train_bag)
X_test_bag = selector1.transform(X_test_bag)

def draw_accuracy_comparison(accuracy_tf, accuracy_bag,title):
    labels = ['TF-IDF', 'Bag-of-Words']
    accuracies = [accuracy_tf, accuracy_bag]
    plt.bar(labels, accuracies)
    plt.xlabel('Feature Representation')
    plt.ylabel('Accuracy')
    plt.title(title+'Accuracy Comparison')
    plt.show()

def draw_confusion_matrix_hitmap(y_test, y_pred,title):
    cm_tf = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_tf, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.values(),
                yticklabels=label_mapping.values())
    plt.title(title+" Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


csv_file = 'training_log.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Model", "Feature_Type", "Accuracy", "Classification_Report"])

def log_metrics(model_name, feature_type, acc,class_report ):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_time, model_name, feature_type, acc, class_report])


def softmax():
    title = " Softmax "
    softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=30)
    #TF-IDF
    softmax.fit(X_train_tf, y_train)
    y_pred_tf = softmax.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    report_tf=classification_report(y_test, y_pred_tf)
    log_metrics(title,"TF-IDF",accuracy_tf,report_tf)
    print(accuracy_tf)
    print(report_tf)
    draw_confusion_matrix_hitmap(y_test, y_pred_tf,title+"TF-IDF")

    #Bag of Words
    softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=30)
    softmax.fit(X_train_bag, y_train)
    y_pred_bag = softmax.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    report_bag=classification_report(y_test, y_pred_bag)
    log_metrics(title,"Bag of Words",accuracy_bag,report_bag)
    print(accuracy_bag)
    print(report_bag)
    draw_confusion_matrix_hitmap(y_test, y_pred_bag, title+"Bag of Words")

    draw_accuracy_comparison(accuracy_tf, accuracy_bag,title)

start_time = time.time()
softmax()
end_time = time.time()
execution_time = end_time - start_time
print(f"Softmax{execution_time}s")

def KNN():
    title=" KNN "
    knn = KNeighborsClassifier()
    knn.fit(X_train_tf, y_train)
    y_pred_tf = knn.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))
    draw_confusion_matrix_hitmap(y_test, y_pred_tf, title+"TF-IDF")

    knn.fit(X_train_bag, y_train)
    y_pred_bag = knn.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))
    draw_confusion_matrix_hitmap(y_test, y_pred_bag, title+"Bag of Words")

    draw_accuracy_comparison(accuracy_tf, accuracy_bag,title)

# KNN()

def decision_tree():
    title = " Decision Tree "

    dt = DecisionTreeClassifier()

    dt.fit(X_train_tf, y_train)
    y_pred_tf = dt.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))
    draw_confusion_matrix_hitmap(y_test, y_pred_tf, title+"TF-IDF")

    dt.fit(X_train_bag, y_train)
    y_pred_bag = dt.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))
    draw_confusion_matrix_hitmap(y_test, y_pred_bag, title+"Bag of Words")

    draw_accuracy_comparison(accuracy_tf, accuracy_bag,title)

# decision_tree()

def native_bayes():
    title = " Native Bayes "
    nb = MultinomialNB()
    nb.fit(X_train_tf, y_train)
    y_pred_tf = nb.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))
    draw_confusion_matrix_hitmap(y_test, y_pred_tf, title + "TF-IDF")

    nb.fit(X_train_bag, y_train)
    y_pred_bag = nb.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))
    draw_confusion_matrix_hitmap(y_test, y_pred_bag, title + "Bag of Words")

    draw_accuracy_comparison(accuracy_tf, accuracy_bag,title)

# native_bayes()

def SVM():
    title = " SVM "
    clf = svm.SVC(kernel='linear')

    clf.fit(X_train_tf, y_train)
    y_pred_tf = clf.predict(X_test_tf)
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    print(classification_report(y_test, y_pred_tf))
    draw_confusion_matrix_hitmap(y_test, y_pred_tf, title + "TF-IDF")

    clf.fit(X_train_bag, y_train)
    y_pred_bag = clf.predict(X_test_bag)
    accuracy_bag = accuracy_score(y_test, y_pred_bag)
    print(classification_report(y_test, y_pred_bag))
    draw_confusion_matrix_hitmap(y_test, y_pred_bag, title + "Bag of Words")

    draw_accuracy_comparison(accuracy_tf, accuracy_bag,title)

# SVM()