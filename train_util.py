import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import csv

label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

def draw_accuracy_comparison(accuracy_tf, accuracy_bag,title):
    labels = ['TF-IDF', 'Bag-of-Words']
    accuracies = [accuracy_tf, accuracy_bag]
    plt.bar(labels, accuracies)
    plt.xlabel('Feature Representation')
    plt.ylabel('Accuracy')
    plt.title(title+'Accuracy Comparison')
    plt.show()

def draw_accuracy_comparison(accuracies, title):
    labels = list(accuracies.keys())
    values = list(accuracies.values())
    plt.bar(labels, values)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(title + ' Accuracy Comparison')
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

def log_metrics(model_name, feature_type, acc, class_report):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    header = ['Time', 'Model', 'Feature Type', 'Accuracy', 'Classification Report']
    data = [current_time, model_name, feature_type, acc, class_report]
    try:
        with open(csv_file, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(data)
    except FileExistsError:
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(len(df))
    features = df['text']
    labels = df['label']
    return df, features, labels

def split_data():
    df, features, labels=load_data('processed_text.csv')
    data_by_label = df.groupby('label')
    X_balanced = []
    y_balanced = []
    min_count = labels.value_counts().min()

    for label, group in data_by_label:
        if len(group) > min_count:
            group = group.sample(n=min_count, random_state=0)

        X_balanced.extend(group['text'].tolist())
        y_balanced.extend(group['label'].tolist())

    X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
    print("train:", len(X_train))
    print("Train set class distribution:", Counter(y_train))
    print("val:", len(X_val))
    print("Validation set class distribution:", Counter(y_val))
    print("test:", len(X_test))
    print("Test set class distribution:", Counter(y_test))
    return X_train, X_val, X_test, y_train, y_val, y_test
