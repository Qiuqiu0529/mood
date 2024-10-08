import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from train_util import draw_confusion_matrix_hitmap,split_data,label_mapping

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
    evaluation_strategy="epoch",
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
trainer.train()
print("Evaluating on Validation Set...")
trainer.evaluate()
print("Evaluating on Test Set...")

predictions, labels, _ = trainer.predict(test_dataset)
predictions = torch.argmax(torch.tensor(predictions), dim=1)
accuracy = accuracy_score(labels, predictions)
print(f"Test Set Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(labels, predictions, target_names=list(label_mapping.values())))

model.save_pretrained("./best_distilbert_model")
tokenizer.save_pretrained("./best_distilbert_model")
print("Model and Tokenizer saved to './best_distilbert_model'")