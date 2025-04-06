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
        self.c_proj.GPT_SCALE_INIT = 1
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
        att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1))) # where k.size(-1) is the head dimension 
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
        self.c_proj.GPT_SCALE_INIT = 1
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

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init prameters
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # we have 2* since we added residue 2 times (one for attention and another for MLP)
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    def forward(self,idx,targets = None):
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
        loss = None
        if targets is not None:
            logits = logits.view(B*T,-1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss


if torch.cuda.is_available():
    device = "cuda"
    print("Computation is running on gpu")
else:
    device = "cpu"
    print("Computation is running on cpu")

config = GPTConfig()
model = GPT(config)
model.eval()
model.to(device)
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

# Usage
count_parameters(model)
# -----------------------------------------------------------------------------------------------------------
#Loading Data
import tiktoken

class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T
        # at init load tokens from disk and store them in memory
        with open("data/input.txt","r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens,dtype = torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        self.current_position = 0
    def next_batch(self):
        B,T = self.B,self.T
        buffer_text =self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buffer_text[:-1]).view(B,T) #inputs
        y = (buffer_text[1:]).view(B,T) #targets
        self.current_position += B*T
        #if loading the next batch would be out of bounds, reset
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x,y

# -----------------------------------------------------------------------------------------------------------
#Training Loop

train_loader = DataLoaderLite(B=4,T=1000)
optimizer = torch.optim.AdamW(model.parameters(),lr = 3e-4)
for i in range(10):
    x,y =train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    optimizer.zero_grad()
    logits,loss = model(x,y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys
sys.exit(0)
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

