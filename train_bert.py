import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from train_util import draw_confusion_matrix_hitmap,split_data,label_mapping,log_metrics

print(torch.__version__)
print(torch.version.cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


X_train, X_val, X_test, y_train, y_val, y_test = split_data()

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)


model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=64)


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))
model.to(device)

training_args = TrainingArguments(
    output_dir='./distilbert_results',
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./distilbert_logs',
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    gradient_accumulation_steps=2,
    resume_from_checkpoint=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    predictions = torch.argmax(logits, dim=-1)
    accuracy = accuracy_score(labels.numpy(), predictions.numpy())
    return {'accuracy': accuracy}

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Starting Training...")
starttime=time.time()
train_output = trainer.train()
traintime=time.time()-starttime

print("Evaluating on Validation Set...")
val_metrics = trainer.evaluate()
val_accuracy = val_metrics['eval_accuracy']

train_losses = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
eval_accuracies = [x['eval_accuracy'] for x in trainer.state.log_history if 'eval_accuracy' in x]

train_steps = range(len(train_losses))
eval_steps = [i for i, x in enumerate(trainer.state.log_history) if 'eval_accuracy' in x]

fig, ax = plt.subplots()
ax.plot(train_steps, train_losses, label='Training Loss')
ax.plot(eval_steps, eval_accuracies, label='Validation Accuracy')
ax.set_xlabel('Steps')
ax.set_ylabel('Loss/Accuracy')
ax.set_title('Training and Validation Accuracy Curve')
ax.legend()
plt.savefig('distilbert_accuracy_curve.png')
plt.close()

print("Evaluating on Test Set...")
predictions, labels, _ = trainer.predict(test_dataset)
predictions = torch.argmax(torch.tensor(predictions), dim=1)
test_accuracy = accuracy_score(labels, predictions)
test_class_report = classification_report(labels, predictions, target_names=list(label_mapping.values()))
print(f"Test Accuracy: {test_accuracy}")
print(f"Classification Report:\n{test_class_report}")

model.save_pretrained("./best_distilbert_model")
tokenizer.save_pretrained("./best_distilbert_model")
print("Model and Tokenizer saved to './best_distilbert_model'")

log_metrics('DistilBERT', 'Test Set', val_accuracy, test_class_report, train_time=0, params=None, test_acc=test_accuracy)

test_cm_path = "./pic/distilbert_test_confusion_matrix.png"
draw_confusion_matrix_hitmap(labels, predictions, title='DistilBERT Test Set', save_path=test_cm_path)