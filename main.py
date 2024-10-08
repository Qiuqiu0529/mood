import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
df_unique = df.iloc[:-1, :]

df_sorted = df_unique[['Method', 'Val Accuracy']].sort_values(by='Val Accuracy', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Val Accuracy', y='Method', data=df_sorted, palette='viridis')
plt.title('Validation Accuracy Comparison Across Methods (Sorted)')
plt.xlabel('Validation Accuracy')
plt.ylabel('Method')

for index, value in enumerate(df_sorted['Val Accuracy']):
    plt.text(value, index, f'{value:.4f}', va='center', ha='left', color='black')

output_path = "pic/val_accuracy_comparison.png"
plt.tight_layout()
plt.savefig(output_path)
plt.show()
print(f"Validation Accuracy Comparison plot saved as {output_path}.")


