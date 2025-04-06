from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 1024 # Maximum sequence length
    vocab_size: int = 50257 # Number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 #number of layers
    n_head: int = 12 #number of heads
    n_embd: int = 768 #embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads in batches
        self.key = nn.Linear(config.n_embd,config.n_embd)
        self.query = nn.Linear(config.n_embd,config.n_embd)
        self.value = nn.Linear(config.n_embd,config.n_embd)
        self.n_head = config.n_head
        self.embd = config.n_embd 
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C = x.size() #batch_size, sequence_length, embedding dimensionality
        #nh = number of head, hs = head size and C (number of channels) = nh * hs
        # In GPT-2 (124M), nh = 12, hs = 64, hence C = 768 channels in transformer 
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        att = (q @ k.transpose(-2,-1))**(1.0/math.sqrt(k.size(-1))) # where k.size(-1) is the head dimension 
        # shape of attention will be (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        att = F.softmax(att,dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side
        y = self.c_proj(y) 
        #Each head learns to focus on different aspects (e.g., syntax, long-term dependencies, etc.) c_proj gives the model the ability to mix the heads in a trainable way.
        #It adds flexibility and expressiveness.
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)     
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),#weight token embedding
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range (config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias = False)
    
    def forward(self,idx):
        #idx is of shape (B,T)
        B,T = idx.size()
        assert T <= self.config.block_size, f"Sequence length cannot be more than block size"
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
        pos = torch.arange(0,T,dtype=torch.long,device= idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T,n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocal_size)
        return logits
        


config = GPTConfig()
model = GPT(config)
model.eval()
model.to('cuda')
# def count_parameters(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total:,}")
#     print(f"Trainable parameters: {trainable:,}")

# # Usage
# count_parameters(model)
# -----------------------------------------------------------------------------------------------------------
num_return_sequences = 5
max_length = 30

#prefix tokenizer
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello,I'm a language model")
tokens = torch.tensor(tokens,dtype=torch.long) #(B,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) #(5,8)
x = tokens.to('cuda')
#------------------------------------------------------------------------------------------------------------

#generation code Block
torch.manual_seed(2909)
torch.cuda.manual_seed(2909)
while x.size(1)<max_length:
    with torch.no_grad():
        # x = (B,T)
        logits = model(x) #(B,T,vocal_size)
        # taking last token since its prediction from our model
        logits = logits[:,-1,:] #(B,vocab_size)
        probs = F.softmax(logits,dim=-1)
        #taking top-k sampling from the output
        # here k = 50, hence top 50 -> (5,50)
        topk_probs,topk_indices = torch.topk(probs,50,dim=-1) 
        #select the token from the topk probabilities
        ix = torch.multinomial(topk_probs,1) #(B,1)
        #gather the corrresponding indices
        xcol = torch.gather(topk_indices,-1,ix) #(B,1)
        #appending to the sequence
        x = torch.cat((x,xcol),dim=1)

for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)

