import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
block_size = 8
batch_size = 32
max_iters = 3000
eval_intervals = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# --------------

torch.manual_seed(1337)

# Reading input and inspect it
with open('input.txt' , 'r', encoding = 'utf-8') as f:
    text = f.read()
      
chars = sorted(set(text))
vocab_size = len(chars)

# encoding and decoding a string
stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i : ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])


data = torch.tensor(encode(text),dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
       data = train_data if split == 'train' else test_data
       ix = torch.randint(len(data) - block_size, (batch_size,))
       x = torch.stack([data[i:i+block_size] for i in ix])
       y = torch.stack([data[i+1:i+block_size+1] for i in ix])
       x,y = x.to(device), y.to(device)
       return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for k  in range(eval_iters):
            X , Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# train_data[:block_size + 1]

# x = train_data[:block_size]
# y = train_data[1:block_size+1]

# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target is : {target}")

# batch_size = 4
# block_size = 8

# xb,yb = get_batch('train')    
# print('inputs : ')
# print(xb.shape)
# print(xb)
# print('targets : ')
# print(yb.shape)
# print(yb)

# print('----')

# torch.manual_seed(1337)
# for b in range(batch_size):
#     for t in range(block_size):
#           context = xb[b, :t+1]
#           target = yb[b,t]
#           print(f"when input is {context.tolist()} the target is : {target}")
          

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
    

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device = device)) # T C
        x = tok_emb + pos_emb # B T C
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# training the model
optimizer = torch.optim.AdamW(m.parameters(),lr = 1e-3)    

for iter in range(max_iters):
    
    if iter % eval_intervals == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")


    xb,yb, = get_batch('train')

    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))