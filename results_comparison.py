import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# filename = 'text.csv'
# data = pd.read_csv(filename)
#
# print(data.head())
# category_counts = data['label'].value_counts()
# print(category_counts)

log_file_path = 'training_log_new.csv'
df = pd.read_csv(log_file_path)
print(df.head())


df['Method'] = df['Model'] + ' (' + df['Feature Type'] + ')'
missing_test_accuracy = df['Test Accuracy'].isna()
method_repeated = df.duplicated(subset='Method', keep=False)


df_filtered = df[~((missing_test_accuracy) & method_repeated)]

df_sorted = df_filtered[['Method', 'Val Accuracy', 'Test Accuracy', 'Train Accuracy', 'Train Log Loss',
                         'Val Log Loss', 'Test Log Loss', 'Train Time (s)', 'Best Params']].sort_values(by='Val Accuracy', ascending=False)

df_sorted['Method'] = df_sorted['Method'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=20)))

plt.figure(figsize=(9, 6))
sns.barplot(x='Val Accuracy', y='Method', data=df_sorted, palette='viridis')
plt.title('Validation Accuracy Comparison Across Methods (Sorted)')
plt.xlabel('Validation Accuracy')
plt.ylabel('Method')

for index, value in enumerate(df_sorted['Val Accuracy']):
    plt.text(value, index, f'{value:.4f}', va='center', ha='left', color='black')


output_path = "pic/val_accuracy_comparison.png"
plt.tight_layout(pad=2, h_pad=1)
plt.savefig(output_path)
#plt.show()
print(f"Validation Accuracy Comparison plot saved as {output_path}.")


top2_methods = df_sorted.head(2)

print("Top 2 Methods Based on Validation Accuracy:\n")
for index, row in top2_methods.iterrows():
    print(f"Method: {row['Method']} ")
    print(f"Validation Accuracy: {row['Val Accuracy']:.4f}")
    print(f"Test Accuracy: {row['Test Accuracy']:.4f}")
    print(f"Train Accuracy: {row['Train Accuracy']:.4f}")
    print(f"Train Loss: {row['Train Log Loss']:.4f}")
    print(f"Validation Loss: {row['Val Log Loss']:.4f}")
    print(f"Test Log Loss: {row['Test Log Loss']:.4f}")
    print(f"Train Time (s): {row['Train Time (s)']:.2f}")
    print("=" * 50)

method_1 = top2_methods.iloc[0]
method_2 = top2_methods.iloc[1]

final_choice = ""
if abs(method_1['Val Accuracy'] - method_2['Val Accuracy']) < 0.02:
    if method_1['Train Time (s)'] > 5 * method_2['Train Time (s)']:
        final_choice = f"Selected {method_2['Method']}  due to shorter training time."
    else:
        final_choice = f"Selected {method_1['Method']} due to better accuracy."
else:
    final_choice = f"Selected {method_1['Method']} due to significantly higher accuracy."

print("\nFinal Model Choice:")
print(final_choice)

labels = top2_methods['Method']
val_accuracies = top2_methods['Val Accuracy']
test_accuracies = top2_methods['Test Accuracy']
train_times = top2_methods['Train Time (s)']


x = range(len(labels))
fig, ax1 = plt.subplots(figsize=(8, 5))
bar_width = 0.3
ax1.bar(x, val_accuracies, width=bar_width, label='Validation Accuracy', align='center', color='blue')
ax1.bar([p + bar_width for p in x], test_accuracies, width=bar_width, label='Test Accuracy', align='center', color='orange')


ax1.set_xlabel('Methods')
ax1.set_ylabel('Accuracy')
ax1.set_title('Top 2 Methods: Validation and Test Accuracy Comparison')
ax1.set_xticks([p + bar_width/2 for p in x])
ax1.set_xticklabels(labels, ha='right')

ax2 = ax1.twinx()
ax2.plot([p + bar_width/2 for p in x], train_times, label='Train Time (s)', color='green', marker='o', linestyle='--')
ax2.set_ylabel('Training Time (s)')

for i, v in enumerate(val_accuracies):
    ax1.text(i-0.07, v + 0.01, f"{v:.4f}", color='blue', fontweight='bold')
for i, v in enumerate(test_accuracies):
    ax1.text(i + bar_width, v + 0.01, f"{v:.4f}", color='orange', fontweight='bold')
for i, v in enumerate(train_times):
    ax2.text(i + bar_width/2, v + 3, f"{v:.1f}", color='green', fontweight='bold')

ax1.legend(loc='upper left', bbox_to_anchor=(0.79, -0.08), borderaxespad=0)
ax2.legend(loc='upper left', bbox_to_anchor=(0.79, -0.2), borderaxespad=0)

plt.tight_layout(rect=[0, 0, 0.8, 1])

output_path_top2 = "pic/top2_methods_comparison.png"
plt.tight_layout()
plt.savefig(output_path_top2)
plt.show()
print(f"Top 2 Methods Comparison plot saved as {output_path_top2}.")



