# import packages
import torch
from torch import nn, optim
from dataset import get_dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR  
from config import GEMMA3_CONFIG
from model import Gemma3


# model constants
PIN_MEMORY= torch.cuda.is_available()
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train constants

learning_rate = 1e-4 
max_iters = 20000 # no of total epochs
warmup_steps = 1000 # increase LR for these many warmup epochs
min_lr = 5e-4 # min lr to reach while cosine annhealing
wd= 1e-3
batch_size= 32

best_model_params_path= "best_model_params.pth"
console_freq= 500



# model related stuff
model = Gemma3(GEMMA3_CONFIG).to(DEVICE)
scaler = torch.amp.GradScaler(DEVICE.type)  # Handles FP16 scaling

loss_func= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr= learning_rate, weight_decay= wd)

warmup_scheduler = LinearLR(optimizer, total_iters= warmup_steps)

cosine_scheduler = CosineAnnealingLR(optimizer, T_max= max_iters - warmup_steps, eta_min= min_lr)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)



# load dataset

# get the train and valid data loaders
train_loader= get_dataloader(
    split= "train",
    pin_memory= PIN_MEMORY,
    context_size= GEMMA3_CONFIG["context_length"],
    stride= GEMMA3_CONFIG["context_length"],
    batch_size= batch_size
)

valid_loader= get_dataloader(
    split= "validation",
    pin_memory= PIN_MEMORY,
    context_size= GEMMA3_CONFIG["context_length"],
    stride= GEMMA3_CONFIG["context_length"],
    batch_size= batch_size
)


# train the model
def train_and_valid_model(loss_func, optimizer, train_loader: DataLoader, valid_loader: DataLoader, model: Gemma3, n_epochs: int, best_model_params_path: str):
    
    train_losses= []
    valid_losses= []
    best_valid_loss= float("inf")
    
    for epoch in range(n_epochs):
        
        # TRAIN LOOP
        model.train()
        total_train_loss= 0
        
        for (batch_inputs, batch_targets) in tqdm(train_loader, desc= f"\nTraining [{epoch+1}/{n_epochs}]", total= len(train_loader)):
        
            # move to device
            batch_inputs= batch_inputs.to(DEVICE)
            batch_targets= batch_targets.to(DEVICE)
            
            with torch.amp.autocast(DEVICE.type):
                # forward pass
                model_logits= model(batch_inputs) # (b, s, v)
                
                # calc losses
                loss= loss_func(model_logits.to(torch.float32).flatten(0, 1), batch_targets.to(torch.long).view(-1))
            
            # update params
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            # update the storers
            total_train_loss+= loss.item()            
        
        # get the avg losses
        avg_train_loss= total_train_loss / len(train_loader)
        
        # update the storers
        train_losses.append(avg_train_loss)
        
        if (epoch + 1) % console_freq == 0 or (epoch + 1) == max_iters:
          print(f"Epoch [{epoch + 1}/{n_epochs}] (TRAIN):  loss -> {avg_train_loss:.2f}\n")
        
        
        
        
        # VALIDATION LOOP
        
        model.eval()
    
        total_valid_loss= 0
    
        for (batch_inputs, batch_targets) in tqdm(valid_loader, desc= f"\nValidation [{epoch+1}/{n_epochs}]", total= len(valid_loader)):
        
            # move to device
            batch_inputs= batch_inputs.to(DEVICE)
            batch_targets= batch_targets.to(DEVICE)
            
            # forward pass
            with torch.no_grad():
                with torch.amp.autocast(DEVICE.type):

                    
                    # forward pass
                    model_logits= model(batch_inputs) # (b, s, v)
                    
                    # calc losses
                    loss= loss_func(model_logits.to(torch.float32).flatten(0, 1), batch_targets.to(torch.long).view(-1))
            
            
            # update the storers
            total_valid_loss+= loss.item()            
        
        # get the avg losses
        avg_valid_loss= total_valid_loss / len(valid_loader)
    
        # update the storers
        valid_losses.append(avg_valid_loss)
        
        if (epoch + 1) % console_freq == 0 or (epoch + 1) == max_iters:
          print(f"Epoch [{epoch + 1}/{n_epochs}] (VALID):  loss -> {avg_valid_loss:.2f}\n\n")
        
        
        # save weights if this is better weights
        if(avg_valid_loss < best_valid_loss):
            best_valid_loss= avg_valid_loss
            torch.save(model.state_dict(), best_model_params_path)
    
        
    
    return train_losses, valid_losses
        
        
        
def plot_train_valid_loss_graph(train_losses: list, valid_losses: list):
    epochs = list(range(1, len(train_losses) + 1))

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(epochs, valid_losses, marker='s', label="Valid Loss")

    plt.title("Training vs Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    
    
if __name__ == "__main__":
    
    
    print(f"\n~ No. of train batches: {len(train_loader)}")
    print(f"~ No. of valid batches: {len(valid_loader)}")
    
    print(f"\n\n--- TRAINING Gemma3 ---\n\n")
    
    # train the model
    train_losses, valid_losses= train_and_valid_model(
        loss_func= loss_func,
        optimizer= optimizer,
        train_loader= train_loader,
        valid_loader= valid_loader,
        model= model,
        n_epochs= max_iters,
        best_model_params_path= best_model_params_path
    )
    
    
    # display the train and valid losses
    print(f"\n\n--- TRAIN RESULTS: ---\n\n")
    print(f"~ Train losses: {train_losses}")
    print(f"~ Valid losses: {valid_losses}")
    
    print(f"\n")
    plot_train_valid_loss_graph(train_losses, valid_losses)
