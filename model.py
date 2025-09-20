# import packages
import torch
from torch import nn
import math
from torch.nn import functional as F
from config import GEMMA3_CONFIG


# constants
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")



# RoPE: rotary positional embeddings

# 1) compute sin_thetas & cos_thetas
def compute_rope_params(head_dim: int, base_theta: int, context_length: int, dtype: torch.dtype):
    
    # calc the 2_i vector
    even_indices= torch.arange(0, head_dim, 2, dtype= dtype) # (head_dim / 2)
    
    # calc the inv freq first (omega_i)
    inv_freq= 1 / (base_theta ** (even_indices / head_dim)  ) # (head_dim / 2)
    
    # calc the positions vector
    pos= torch.arange(0, context_length, 1, dtype= dtype) # (s)
    
    # calc the angles or thetas
    thetas= pos.unsqueeze(1) * inv_freq.unsqueeze(0) # (s, 1) * (1, head_dim / 2) = (s, head_dim / 2)
    
    # duplicate to match the head_dim dimension 
    thetas = thetas.repeat_interleave(2, dim=1) # (s, head_dim)

    
    # calc the sin and cos angles
    sin_thetas= torch.sin(thetas) # (s, head_dim)
    cos_thetas= torch.cos(thetas) # (s, head_dim)
    
    return sin_thetas, cos_thetas


# 2) apply rope to the input tensor 
def apply_rope(x: torch.Tensor, sin_thetas: torch.Tensor, cos_thetas: torch.Tensor ):
    
    # x -> (b, n_h, s, d_h)
    # sin_thetas -> (s, d_h)
    # cos_thetas -> (s, d_h)
    
    b, n_h, s, d_h= x.shape
    
    # segerate x into 2 halves
    x1= x[..., :d_h//2] # (b, n_h, s, d_h/2)
    x2= x[..., d_h//2:] # (b, n_h, s, d_h/2)
    
    # expand sin/cos to match x dimensions
    sin_thetas= sin_thetas.unsqueeze(0).unsqueeze(0) # (1, 1, s, d_h)
    cos_thetas= cos_thetas.unsqueeze(0).unsqueeze(0) # (1, 1, s, d_h)
        
    # rotate x
    rot_x= torch.cat([-x2, x1], dim= -1) # (b, n_h, s, d_h)
    
    out= (x * cos_thetas) + (rot_x * sin_thetas)
    
    return out.to(dtype= x.dtype)
    


# RMS Normalization
class RMSNorm(nn.Module):
    
    def __init__(self, emb_dim: int, eps: float= 1e-6):
        
        super().__init__()
        
        self.eps= eps
        
        self.scale= nn.Parameter(torch.zeros(emb_dim))        
        
        self.shift= nn.Parameter(torch.zeros(emb_dim))        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S, d_m)
        x_f= x.float()
        
        var_x= x_f.pow(2).mean(dim= -1, keepdim= True) 
        rms= torch.sqrt(var_x + self.eps)
        x_norm= x_f / rms
        
        out= x_norm*(1 + self.scale.float()) + self.shift.float();
        
        return out.to(x.dtype) 
        
        
        
# Feed Forward MLP
class FeedForwardMLP(nn.Module):
    
    def __init__(self, emb_dim: int, hid_dim: int, dtype: torch.dtype):
        
        super().__init__()
        
        self.exp_1= nn.Linear(emb_dim, hid_dim, False, dtype= dtype) 
        self.exp_2= nn.Linear(emb_dim, hid_dim, False, dtype= dtype) 
        
        self.contr= nn.Linear(hid_dim, emb_dim, False, dtype= dtype) 
        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S, emb_dim)
        e1= self.exp_1(x) # (B, S, hid_dim)
        e2= self.exp_2(x) # (B, S, hid_dim)
        
        out= F.gelu(e1, approximate= "tanh") * e2; # (B, S, hid_dim)
        
        out= self.contr(out) # (B, S, emb_dim)
        
        return out
    
    
    
# Multi Query Attention
class MultiQueryAttention(nn.Module):
    
    def __init__(self, d_in: int, n_heads: int, head_dim: int, dtype: torch.dtype):
        
        super().__init__()
        
        self.d_in= d_in
        self.n_heads= n_heads
        self.head_dim= head_dim
        self.d_out= n_heads * head_dim
        
        
        # projection layers
        self.w_q= nn.Linear(self.d_in, self.d_out, bias= False, dtype= dtype)
        self.w_k= nn.Linear(self.d_in, self.head_dim, bias= False, dtype= dtype)
        self.w_v= nn.Linear(self.d_in, self.head_dim, bias= False, dtype= dtype)
        self.w_o= nn.Linear(self.d_out, self.d_in, bias= False, dtype= dtype)
        

        # RMS normalization for q/k
        self.q_norm_layer= RMSNorm(self.head_dim)
        self.k_norm_layer= RMSNorm(self.head_dim)
        
        
        
        self.scale= self.head_dim ** -0.5 # scaling factor for attention
        
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, sin_thetas: torch.Tensor, cos_thetas: torch.Tensor):
        
        b, s, _= x.shape
        
        
        # calc the query, key, value vectors
        q= self.w_q(x) # (b, s, d_out)
        k= self.w_k(x) # (b, s, head_dim)
        v= self.w_v(x) # (b, s, head_dim)
        
        
        # change shape of query, key, value vectors
        q= q.view(b, s, self.n_heads, self.head_dim) # (b, s, n_heads, head_dim)
        k= k.view(b, s, 1, self.head_dim) # (b, s, 1, head_dim)
        v= v.view(b, s, 1, self.head_dim) # (b, s, 1, head_dim)
        
        
        # change shape of query, key, value vectors
        q= q.transpose(1, 2) # (b, n_heads, s, head_dim)
        k= k.transpose(1, 2) # (b, 1, s, head_dim)
        v= v.transpose(1, 2) # (b, 1, s, head_dim)
        
        
        # qk norm
        q= self.q_norm_layer(q) # (b, n_heads, s, head_dim)
        k= self.k_norm_layer(k) # (b, 1, s, head_dim)
        
        
        # apply RoPE
        q = apply_rope(q, sin_thetas, cos_thetas) # (b, n_heads, s, head_dim)
        k = apply_rope(k, sin_thetas, cos_thetas) # (b, 1, s, head_dim)
        
        
        # expand k, v accross all the heads
        k= k.repeat_interleave(self.n_heads, dim= 1) # (b, n_heads, s, head_dim)
        v= v.repeat_interleave(self.n_heads, dim= 1) # (b, n_heads, s, head_dim)
        
        
        # calc the attention score
        atten_scores= q @ k.transpose(-1, -2) # (b, n_heads, s, head_dim) @ (b, n_heads, head_dim, s) -> (b, n_heads, s, s)
        
        atten_scores*= self.scale # scale the attention score by root of d_m
        
        # apply the mask
        atten_scores.masked_fill_(mask, float("-inf"))
        
        # calc the attention weights
        atten_weights= torch.softmax(atten_scores, dim= -1) # (b, n_heads, s, s)
        
        
        # calc the output
        out= atten_weights @ v # (b, n_heads, s, s) @ (b, n_heads, s, head_dim) -> (b, n_heads, s, head_dim)
        
        # change shape of the output: (b, n_heads, s, head_dim) -> (b, s, n_heads, head_dim) -> (B, S, d_out)
        out= out.transpose(1, 2)
        out= out.contiguous().view(b, s, self.d_out)
        
        # pass the output through the final output weight matrix
        out= self.w_o(out) # (B, S, d_in)
        
        return out

    
    
# Transformer Block
class TransformerBlock(nn.Module):
    
    def __init__(self, config: dict):
        
        super().__init__()
        
        self.atten= MultiQueryAttention(
            d_in= config["emb_dim"],
            n_heads= config["n_heads"],
            head_dim= config["head_dim"],
            dtype= config["dtype"]
        )
        
        self.mlp= FeedForwardMLP(
            emb_dim= config["emb_dim"],
            hid_dim= config["hid_dim"], 
            dtype= config["dtype"]
        )
        
        self.pre_atten_norm= RMSNorm(emb_dim= config["emb_dim"])
        self.post_atten_norm= RMSNorm(emb_dim= config["emb_dim"])
        self.pre_mlp_norm= RMSNorm(emb_dim= config["emb_dim"])
        self.post_mlp_norm= RMSNorm(emb_dim= config["emb_dim"])
        
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, sin_thetas: torch.Tensor, cos_thetas: torch.Tensor):
        
        # attention phase
        shortcut= x
        x= self.pre_atten_norm(x)
        x= self.atten(x, mask, sin_thetas, cos_thetas)
        x= self.post_atten_norm(x)
        x= x + shortcut

        # feed-forward phase      
        shortcut= x
        x= self.pre_mlp_norm(x)
        x= self.mlp(x)
        x= self.post_mlp_norm(x)
        x= x + shortcut
        
        return x
    
    
    
# Final Gemma3 Model
class Gemma3(nn.Module):
    
    def __init__(self, config: dict):
        
        super().__init__()
        
        self.vocab_size= config["vocab_size"]   
        self.emb_dim= config["emb_dim"]   
        self.head_dim= config["head_dim"]   
        self.context_length= config["context_length"]   
        self.dtype= config["dtype"]
        self.sliding_window= config["sliding_window"]
        
        
        self.rope_sliding_base= config["rope_local_base"]
        self.rope_full_base= config["rope_base"]
        
        
        self.token_embeds= nn.Embedding(self.vocab_size, self.emb_dim, dtype= self.dtype)
        self.blocks= nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm= RMSNorm(config["emb_dim"])
        self.out_head= nn.Linear(self.emb_dim, self.vocab_size, bias= False, dtype= self.dtype)
        self.layer_types= config["layer_types"]
        
        
        # compute the sin_thetas & cos_thetas for sliding and full attention
        sin_thetas_sliding, cos_thetas_sliding= compute_rope_params(self.head_dim, self.rope_sliding_base, self.context_length, dtype= torch.float32)
        
        sin_thetas_full, cos_thetas_full= compute_rope_params(self.head_dim, self.rope_full_base, self.context_length, dtype= torch.float32)
        
        
        # register as buffers
        self.register_buffer("sin_thetas_sliding", sin_thetas_sliding, persistent=True)
        self.register_buffer("cos_thetas_sliding", cos_thetas_sliding, persistent=True)
        self.register_buffer("sin_thetas_full", sin_thetas_full, persistent=True)
        self.register_buffer("cos_thetas_full", cos_thetas_full, persistent=True)

           
           
    def _create_masks(self, context_length: int, device):
                
        i= torch.arange(end= context_length, device= device).unsqueeze(1) # (s, 1)
        j= torch.arange(end= context_length, device= device).unsqueeze(0) # (1, s)
        
        # create the mask_full which masks the future tokens
        mask_full= j > i # (s, s)
        mask_past_tokens= (i-j) >= self.sliding_window
        mask_sliding= mask_full | mask_past_tokens # (s, s)
        
        return mask_full, mask_sliding
        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (b, s)
        b, s= x.shape
        
        # get the token embeds
        x= self.token_embeds(x) # (b, s, emb_dim)
        x*= self.emb_dim ** 0.5 # (b, s, emb_dim)
        
        # get the masks
        mask_full, mask_sliding= self._create_masks(s, x.device)
        
        # pass through the transformer blocks
        for i, block in enumerate(self.blocks):
            
            # check if its full or sliding attention
            if(self.layer_types[i] == "sliding_attention"):
                x= block(x, mask_sliding, self.sin_thetas_sliding, self.cos_thetas_sliding)
                
            else:
                x= block(x, mask_full, self.sin_thetas_full, self.cos_thetas_full)
                
        # final norm and output head
        x= self.final_norm(x)
        out= self.out_head(x.to(self.dtype)) # (b, s, vocab_size)
        
        return out
                
                
                
if __name__ == "__main__":
    
    # sample inputs
    x = torch.randint(1, 9, (2, GEMMA3_CONFIG["context_length"]), dtype= torch.int32, device= DEVICE) # (B, S)
    y = torch.randint(1, 9, (2, GEMMA3_CONFIG["context_length"]), dtype= torch.int32, device= DEVICE) # (B, S)
    
    
    model = Gemma3(GEMMA3_CONFIG).to(DEVICE)
    
    logits = model(x)
    
    print("~ Input shape:", x.shape)
    print("~ Output shape:", y.shape)
    print("~ Logits shape:", logits.shape)
    


    # get the loss
    loss_func= nn.CrossEntropyLoss()
    
    ce_loss= loss_func(logits.to(torch.float32).flatten(0, 1), y.to(torch.long).view(-1))
    
    print(f"~ CE Loss: {ce_loss:.2f}")

    # calc total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n~ Total number of parameters: {total_params:,}")