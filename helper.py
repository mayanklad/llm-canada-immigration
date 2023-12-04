# Importing libraries
import re
import warnings

import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from nltk.translate.bleu_score import sentence_bleu

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

# Class for initializing training data
class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = pd.read_csv(path, encoding='unicode_escape')
        self.X = []

        for idx, row in self.data.iterrows():
            question = str(row['question'])
            answer = str(row['answer'])
            self.X.append('<startofstring> ' + question + ' <bot>: ' + answer + ' <endofstring')

        self.X_encoded = tokenizer(self.X, max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Calculate BLEU score
def cal_epoch_bleu(model, tokenizer, device):
    df_eval = pd.read_csv('eval_chatbot.csv', encoding='unicode_escape')
    bleu_scores = []
    
    for index, row in df_eval.iterrows():        
        output = infer(row['question'], model, tokenizer, device)

        bleu_score = sentence_bleu([row['answer']], output)
        bleu_scores.append(bleu_score)
    
    return bleu_scores

# Function to train model
def train(chatData, model, tokenizer, optimizer, device, model_path, epochs=100):
    loss_values = []  # To store loss values
    bleu_scores = []  # To store BLEU scores

    for i in tqdm(range(epochs)):
        total_loss = 0

        for data in chatData:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = output.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate BLEU scores for the epoch
        epoch_bleu_scores = cal_epoch_bleu(model, tokenizer, device)
        print(epoch_bleu_scores)
        average_epoch_bleu_score = sum(epoch_bleu_scores) / len(epoch_bleu_scores)
        bleu_scores.append(average_epoch_bleu_score)

        # Log and print the loss for the epoch
        print(f"Epoch {i + 1}/{epochs} - Loss: {total_loss:.4f} - BLEU Score: {average_epoch_bleu_score:.4f}")

        # Append the loss values to the list for plotting
        loss_values.append(total_loss)

        # Save the model and its configuration after each epoch
        model.save_pretrained(model_path)  # Save both weights and configuration
        tokenizer.save_pretrained(model_path)  # Save tokenizer's configuration

    # Plot loss and BLEU scores
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, bleu_scores, label='BLEU Score', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title('BLEU Scores')

    plt.tight_layout()
    plt.show()

    print("Training completed.")

# Function for model inference
def infer(inp, model, tokenizer, device):
    inp = '<startofstring> ' + inp + ' <bot>: '
    inp = tokenizer(inp, return_tensors='pt')
    X = inp['input_ids'].to(device)
    a = inp['attention_mask'].to(device)

    output = model.generate(X, attention_mask=a, max_length=50)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    output = re.split('<bot>:', output)[-1].strip()
    output = re.split('<end', output, 1)[0].strip()

    return output
