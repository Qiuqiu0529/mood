import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, classification_report,log_loss
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

train_outputs = trainer.predict(train_dataset)
train_logits, train_labels = train_outputs.predictions, train_outputs.label_ids
train_probabilities = torch.softmax(torch.tensor(train_logits), dim=-1).numpy()
train_loss = log_loss(train_labels, train_probabilities)
train_acc = accuracy_score(train_labels, np.argmax(train_probabilities, axis=1))


print("Evaluating on Validation Set...")
val_metrics = trainer.evaluate()
val_accuracy = val_metrics['eval_accuracy']

val_outputs = trainer.predict(val_dataset)
val_logits, val_labels = val_outputs.predictions, val_outputs.label_ids
val_probabilities = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
val_loss = log_loss(val_labels, val_probabilities)

train_losses = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
eval_accuracies = [x['eval_accuracy'] for x in trainer.state.log_history if 'eval_accuracy' in x]


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
plt.savefig('pic/distilbert_accuracy_curve.png')
plt.close()

print("Evaluating on Test Set...")
test_outputs = trainer.predict(test_dataset)
test_logits, test_labels = test_outputs.predictions, test_outputs.label_ids
test_probabilities = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
test_loss = log_loss(test_labels, test_probabilities)
test_predictions = np.argmax(test_probabilities, axis=1)

test_accuracy = accuracy_score(test_labels, test_predictions)
test_class_report = classification_report(test_labels, test_predictions, target_names=list(label_mapping.values()))


model.save_pretrained("./best_distilbert_model")
tokenizer.save_pretrained("./best_distilbert_model")
print("Model and Tokenizer saved to './best_distilbert_model'")


test_cm_path = "./pic/distilbert_test_confusion_matrix.png"
draw_confusion_matrix_hitmap(test_labels, test_predictions, title='DistilBERT Test Set', save_path=test_cm_path)

log_metrics(
    model_name='DistilBERT',
    feature_type='Contextualized Word Embeddings, [CLS] Token Output',
    val_acc=val_accuracy,
    train_acc=train_acc,
    train_loss=train_loss,
    val_loss=val_loss,
    test_acc=test_accuracy,
    test_log_loss=test_loss,
    class_report=test_class_report,
    train_time=traintime,
    params=str(training_args.to_dict()),
    cm_path=test_cm_path,
)