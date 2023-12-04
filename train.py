# Importing libraries
import os
import warnings

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from helper import ChatData
from helper import train

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

def main():
    # Path to save the trained model
    MODEL_PATH = 'chatbot_model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<pad>',
                                'bos_token': '<startofstring>',
                                'eos_token': '<endofstring>'})
    tokenizer.add_tokens(['<bot>:'])

    # Load or initialize the model
    if os.path.exists(MODEL_PATH):
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, local_files_only=True)
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    chatData = ChatData('sample_data.csv', tokenizer)
    chatData = DataLoader(chatData, batch_size=4)
    model.train()

    optimizer = Adam(model.parameters(), lr=1e-3)

    print("Training ...")
    train(chatData, model, tokenizer, optimizer, device, MODEL_PATH)

if __name__ == "__main__":
    main()
