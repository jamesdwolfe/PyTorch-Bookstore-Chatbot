import random
import json
import torch
from model import NeuralNet
from toolkit import bag_of_words, tokenize

INPUT_FILE_NAME = "trained_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOT_NAME = "Bob"

with open('messages.json', 'r') as file:
    messages = json.load(file)

data = torch.load(INPUT_FILE_NAME)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print(f"{BOT_NAME}: Let's talk! Type 'quit' to exit.")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    model_input = bag_of_words(sentence, all_words)
    model_input = model_input.reshape(1, model_input.shape[0])
    model_input = torch.from_numpy(model_input).to(device)

    output = model(model_input)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    prob = torch.softmax(output, dim=1)[0][predicted.item()]

    if prob.item() > 0.8:
        for intent in messages["messages"]:
            if tag == intent["tag"]:
                print(f'{BOT_NAME}: {random.choice(intent["responses"])}')
    else:
        print(f'{BOT_NAME}: I do not understand what you said.')
