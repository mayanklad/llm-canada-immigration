# Importing libraries
import os
import warnings

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from helper import infer

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

    # Load the model
    if os.path.exists(MODEL_PATH):
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, local_files_only=True)
        model = model.to(device)
        model.eval()

        print('Chat with the chatbot. Type "exit" to end the conversation.')
        while True:
            inp = input('You: ')
            if inp == 'exit':
                break
            response = infer(inp, model, tokenizer, device)
            print(f'Chatbot: {response}')
    else:
        print('Error: Model not found at location', MODEL_PATH)

if __name__ == "__main__":
    main()

