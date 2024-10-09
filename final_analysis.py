import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import log_loss

csv_save_path = 'confusion_matrix.csv'
confusion_data = pd.read_csv(csv_save_path, header=None, index_col=None)

log_file_path = 'training_log_new.csv'
train_log = pd.read_csv(log_file_path)
print(len(train_log))
confusion_data_cleaned = confusion_data[confusion_data[0] != 0].reset_index(drop=True)
cleaned_matrices = []
current_matrix = None
label_mapping = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


for index, row in confusion_data_cleaned.iterrows():
    if isinstance(row[1], str) and "Model:" in row[1]:
        if current_matrix is not None:
            cleaned_matrices.append(current_matrix)

        current_matrix = pd.DataFrame(columns=label_mapping)
    elif isinstance(row[0], str) and row[0] in label_mapping and current_matrix is not None:
        current_matrix.loc[row[0]] = row[1:].values


if current_matrix is not None:
    cleaned_matrices.append(current_matrix)

if len(cleaned_matrices) > 0:
    print(f"Total confusion matrices identified: {len(cleaned_matrices)}")
else:
    raise ValueError("No valid confusion matrices found in the CSV file.")

confusion_matrix_df = cleaned_matrices[-1].astype(int)
print(confusion_matrix_df)

last_log = train_log.iloc[-1]
classification_report = last_log['Classification Report']
print(classification_report )
class_report = []
if pd.notna(classification_report):
    lines = classification_report.split('\n')
    for line in lines:
        split_line = line.split()
        if len(split_line) >= 5 and split_line[0].isdigit():
            class_report.append({
                'Class': split_line[0],
                'Precision': float(split_line[1]),
                'Recall': float(split_line[2]),
                'F1-Score': float(split_line[3]),
                'Support': int(split_line[4])
            })

print(len(class_report))
class_report_df = pd.DataFrame(class_report)
print(class_report_df)

misclassification_analysis = []
for label in confusion_matrix_df.index:
    TP = confusion_matrix_df.loc[label, label]  # True Positives
    FN = confusion_matrix_df.loc[label, :].sum() - TP  # False Negatives
    FP = confusion_matrix_df.loc[:, label].sum() - TP  # False Positives

    misclassified_counts = confusion_matrix_df.loc[label, :].drop(index=label)
    most_misclassified_label = misclassified_counts.idxmax()
    most_misclassified_count = misclassified_counts.max()

    label_index = label_mapping.index(label)

    misclassification_analysis.append({
        "Class": str(label_index),
        "True Positive (TP)": TP,
        "False Positive (FP)": FP,
        "False Negative (FN)": FN,
        "Most Misclassified As": most_misclassified_label,
        "Misclassified Count": most_misclassified_count
    })

misclassification_df = pd.DataFrame(misclassification_analysis)
print(misclassification_df)
final_analysis = pd.merge(class_report_df, misclassification_df, on='Class')
print("Final Analysis DataFrame:")
print(final_analysis)
final_analysis['Class'] = final_analysis['Class'].map(lambda x: label_mapping[int(x)])


plt.figure(figsize=(8, 5))
sns.barplot(x='Precision', y='Class', data=final_analysis, palette='viridis')
plt.title('Precision for Each Class')
plt.xlabel('Precision')
plt.ylabel('Class')
for index, value in enumerate(final_analysis['Precision']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', color='black')
plt.tight_layout()
plt.savefig("pic/class_precision_comparison.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='Recall', y='Class', data=final_analysis, palette='magma')
plt.title('Recall for Each Class')
plt.xlabel('Recall')
plt.ylabel('Class')
for index, value in enumerate(final_analysis['Recall']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', color='black')
plt.tight_layout()
plt.savefig("pic/class_recall_comparison.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='F1-Score', y='Class', data=final_analysis, palette='coolwarm')
plt.title('F1-Score for Each Class')
plt.xlabel('F1-Score')
plt.ylabel('Class')
for index, value in enumerate(final_analysis['F1-Score']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', color='black')
plt.tight_layout()
plt.savefig("pic/class_f1score_comparison.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='Misclassified Count', y='Class', data=final_analysis, palette='Set2')
plt.title('Misclassification Count for Each Class')
plt.xlabel('Misclassified Count')
plt.ylabel('Class')
for index, value in enumerate(final_analysis['Misclassified Count']):
    plt.text(value, index, f'{value:.0f}', va='center', ha='left', color='black')
plt.tight_layout()
plt.savefig("pic/class_misclassification_count.png")
plt.show()

most_misclassified_class = final_analysis.loc[final_analysis['Misclassified Count'].idxmax()]
print(f"Class most frequently misclassified: {most_misclassified_class['Class']} "
      f"misclassified as {most_misclassified_class['Most Misclassified As']} "
      f"with {most_misclassified_class['Misclassified Count']} instances.")

final_analysis_second = final_analysis.drop(final_analysis['Misclassified Count'].idxmax())
second_most_misclassified_class = final_analysis_second.loc[final_analysis_second['Misclassified Count'].idxmax()]
print(f"Class second most frequently misclassified: {second_most_misclassified_class['Class']} "
      f"misclassified as {second_most_misclassified_class['Most Misclassified As']} "
      f"with {second_most_misclassified_class['Misclassified Count']} instances.")

final_analysis_third = final_analysis_second.drop(final_analysis_second['Misclassified Count'].idxmax())
third_most_misclassified_class = final_analysis_third.loc[final_analysis_third['Misclassified Count'].idxmax()]
print(f"Class third most frequently misclassified: {third_most_misclassified_class['Class']} "
      f"misclassified as {third_most_misclassified_class['Most Misclassified As']} "
      f"with {third_most_misclassified_class['Misclassified Count']} instances.")


max_accuracy_class = final_analysis.loc[final_analysis['Recall'].idxmax()]
min_accuracy_class = final_analysis.loc[final_analysis['Recall'].idxmin()]

print(f"Highest Accuracy Class: {max_accuracy_class['Class']} with Recall: {max_accuracy_class['Recall']:.2f}")
print(f"Lowest Accuracy Class: {min_accuracy_class['Class']} with Recall: {min_accuracy_class['Recall']:.2f}")


final_analysis.to_csv("final_last_model_classification_analysis.csv", index=False)
print("Final classification analysis saved to 'final_last_model_classification_analysis.csv'.")