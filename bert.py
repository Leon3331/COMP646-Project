import pandas as pd

# Load the dataset to see the first few rows and its structure
file_path = './Top_20_Sampled_Poetry_Dataset.csv'
poetry_data = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split

# Preprocess the dataset: remove unnecessary whitespace
poetry_data['Poem'] = poetry_data['Poem'].str.strip()

# Splitting the dataset into training and validation sets
train_df, val_df = train_test_split(poetry_data, test_size=0.2, random_state=42)

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode the text data for batch processing
def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

# Update the PoetryDataset class to process batch inputs
class PoetryDataset(Dataset):
    def __init__(self, poems, labels):
        self.poems = poems
        self.labels = labels

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, idx):
        poem_encoded = tokenizer(self.poems[idx], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        item = {key: val.squeeze() for key, val in poem_encoded.items()}
        label = torch.tensor(self.labels[idx])
        return item, label

# Label encoding
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['Random_Tag'])
val_labels = label_encoder.transform(val_df['Random_Tag'])

# Creating datasets
train_dataset = PoetryDataset(train_df['Poem'].tolist(), train_labels)
val_dataset = PoetryDataset(val_df['Poem'].tolist(), val_labels)


# Creating data loaders with correct batch handling
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=None)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=None)

import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import collections

# Set up GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model.to(device)
# Start training loop
epochs = 10
# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for inputs, labels in data_loader:
        input_ids = inputs['input_ids'].squeeze().to(device)
        attention_mask = inputs['attention_mask'].squeeze().to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        predictions = outputs.logits.argmax(dim=1)
        correct_predictions += torch.sum(predictions == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# Validation function
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            input_ids = inputs['input_ids'].squeeze().to(device)
            attention_mask = inputs['attention_mask'].squeeze().to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=1)
            correct_predictions += torch.sum(predictions == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# Initialize lists to save history
history = collections.defaultdict(list)
best_accuracy = 0


for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, len(train_df))
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, len(val_df))
    print(f'Val   loss {val_loss} accuracy {val_acc}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

# Plotting training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='train loss', ls='--')
plt.plot(history['val_loss'], label='validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
