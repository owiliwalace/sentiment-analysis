import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
csv_file_path = 'C:/Users/User/Desktop/Tweets.csv'
df = pd.read_csv(csv_file_path)

#november 26th

for column in ['textID', 'text', 'selected_text', 'sentiment']:
    print(f"Length of {column}: {len(df[column])}")

df['text'] = df['text'].astype(str)
train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['sentiment'])
test_labels = label_encoder.transform(test_data['sentiment'])

train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)

test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_labels)
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['sentiment'].unique()))

# Training Loop (Placeholder, replace with actual training





optimizer = Adam(model.parameters(), lr=2e-5)
criterion = CrossEntropyLoss()

# Example training loop (replace with your actual training code)
num_epochs = 5
batch_size = 2
training_losses = []




for epoch in range(num_epochs):
    model.train()
    for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
        inputs, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
 evaluation code)
model.eval()
predictions = []
true_labels = []
#september 13th
with torch.no_grad():
    for batch in DataLoader(test_dataset, batch_size=batch_size):
        inputs, attention_mask, labels = batch
        outputs = model(inputs, attention_mask=attention_mask)
        predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

bert_f1 = f1_score(true_labels, predictions, average='weighted')
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Iterations ')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f"Bert F1 Score: {bert_f1}")

