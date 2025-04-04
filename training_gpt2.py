from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import Functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads in batches
        # self.key = nn.Linear(config.n_embd,config.n_embd)
        # self.query = nn.Linear(config.n_embd,config.n_embd)
        # self.value = nn.Linear(config.n_embd,config.n_embd)
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.n_head = config.n_head
        self.embd = config.embd 
        self.c_proj = nn.Linear(config.embd,config.embd)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C = x.size() #batch_size, sequence_length, embedding dimensionality
        #nh = number of head, hs = head size and C (number of channels) = nh * hs
        # In GPT-2 (124M), nh = 12, hs = 64, hence C = 768 channels in transformer 
        # k = self.key(x)
        # q = self.query(x)
        # v = self.value(x)
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
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