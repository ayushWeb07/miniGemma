# import packages
import torch
from config import GEMMA3_CONFIG
from model import Gemma3
from train import DEVICE, PIN_MEMORY, best_model_params_path
import tiktoken
import os

# get the model outputs
def get_model_outputs(inputs: torch.Tensor, model: Gemma3, max_new_tokens_generated: int, config: dict):
    
    model.eval()
    
    # inputs -> (B, S)
    
    for _ in range(max_new_tokens_generated):
        
        # get the outputs, probs, preds
        with torch.no_grad():
            logits= model(inputs[:, -config["context_length"]: ]) # (B, S, V)
        
        probs= torch.softmax(logits, dim= -1) # (B, S, V)
        preds= torch.argmax(probs, dim= -1) # (B, S)
        
        # get the final model preds
        final_model_preds= preds[:, -1].unsqueeze(1) # (B, 1)
        
        # append the final model pred -> inputs := (B, S + 1)
        inputs= torch.cat([inputs, final_model_preds], dim= -1)
        
    
    
    return inputs

# load the model with the best weights
model = Gemma3(GEMMA3_CONFIG).to(DEVICE)

if os.path.exists(best_model_params_path):
    model.load_state_dict(torch.load(best_model_params_path, map_location= DEVICE))
    
    
model.eval();


if __name__ == "__main__":
    # actual inference
    tokenizer = tiktoken.get_encoding("gpt2")

    sentence = "Once upon a time there was a pumpkin."

    x = torch.tensor(tokenizer.encode_ordinary(sentence), device= DEVICE).unsqueeze(dim = 0)

    y = get_model_outputs(x, model, 200, GEMMA3_CONFIG)

    print(f"X: {x.shape}\nY: {y.shape}\n")
    print(f"Text (before generation): {sentence}")
    print(f"Text (after generation): {tokenizer.decode(y.squeeze().tolist())}")