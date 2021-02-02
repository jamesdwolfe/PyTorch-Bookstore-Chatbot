import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from toolkit import tokenize, stem, bag_of_words
from model import NeuralNet

OUTPUT_FILE_NAME = "trained_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_words = []
tags = []
xy = []
x_train = []
y_train = []

with open('messages.json', 'r') as file:
    messages = json.load(file)

for message in messages['messages']:
    tag = message['tag']
    tags.append(tag)
    for pattern in message['patterns']:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))

# Stem -> Check if digit or alpha -> Unique Set -> Sort
all_words = sorted(set([stem(word) for word in all_words if word.isalpha() or word.isdigit()]))
tags = sorted(set(tags))

for pattern_array, tag in xy:
    bag = bag_of_words(pattern_array, all_words)
    x_train.append(bag)
    y_train.append(tags.index(tag))

x_train = np.array(x_train)
y_train = np.array(y_train)

batch_size = 8
num_workers = 0
input_size = len(all_words)
hidden_size = 16
output_size = len(tags)
learning_rate = 0.001
epochs = 500


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = NeuralNet(input_size, hidden_size, output_size).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(DEVICE)
        labels = labels.to(dtype=torch.long).to(DEVICE)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'{epoch+1},{loss.item():.10f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, OUTPUT_FILE_NAME)
print(f'Training complete. Model saved to {OUTPUT_FILE_NAME}.')
