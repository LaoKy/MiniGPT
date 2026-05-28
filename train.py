# MODEL DUA TREN KIEN TRUC MODEL LLAMA THAY VI GPT-3 DE MUOT TRONG MW HON VA THONG MINH HON

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math, random, os, time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Đang dùng: {DEVICE}')

CFG = {
    'vocab_size':  2000,
    'embed_dim':   352,
    'ffn_dim':     1408,
    'n_layers':    6,
    'n_heads':     4,
    'n_kv_heads':  2,
    'ctx_len':     96,
    'dropout':     0.1,
    'lr':          1e-4,
    'epochs':      20000,
    'batch_size':  32,
    'patience':    800,
    'min_epoch':   1000,
}

exec(open('tokenizer.py').read())
tok = BPETokenizer(vocab_size=CFG['vocab_size'])
tok.load('data/tokenizer.json')
real_vocab = len(tok.vocab)
CFG['vocab_size'] = real_vocab
print(f'Vocab: {real_vocab} tokens')

class RoPE(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())
    def forward(self, x):
        seq = x.shape[2]
        cos = self.cos[:seq].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim * n_kv_heads, bias=False)
        self.wv = nn.Linear(dim, self.head_dim * n_kv_heads, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = self.rope(q); k = self.rope(k)
        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        return self.wo((attn @ v).transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg['embed_dim'])
        self.attn  = GQAttention(cfg['embed_dim'], cfg['n_heads'], cfg['n_kv_heads'], cfg['dropout'])
        self.norm2 = nn.RMSNorm(cfg['embed_dim'])
        self.ffn   = SwiGLU(cfg['embed_dim'], cfg['ffn_dim'], cfg['dropout'])
        self.drop  = nn.Dropout(cfg['dropout'])
    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed  = nn.Embedding(cfg['vocab_size'], cfg['embed_dim'])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.norm   = nn.RMSNorm(cfg['embed_dim'])
        self.drop   = nn.Dropout(cfg['dropout'])
        self.cfg    = cfg
    def forward(self, x, mask=None):
        x = self.drop(self.embed(x))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x @ self.embed.weight.T

def load_dataset(path, tok, ctx_len):
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' not in line: continue
            parts = line.split('|', 1)
            if len(parts) != 2: continue
            q, a = parts[0].strip(), parts[1].strip()
            ids = tok.encode_qa(q, a)
            if len(ids) < 4: continue
            try: sep_pos = ids.index(tok.SEP)
            except ValueError: sep_pos = 0
            if len(ids) > ctx_len + 1: ids = ids[:ctx_len + 1]
            samples.append((ids, sep_pos))
    print(f'Loaded {len(samples)} samples')
    return samples

def make_batch(samples, batch_size, ctx_len, device):
    batch = random.sample(samples, min(batch_size, len(samples)))
    max_len = min(ctx_len, max(len(s[0]) for s in batch))
    X = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
    Y = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
    for i, (ids, sep_pos) in enumerate(batch):
        ids = ids[:max_len + 1]
        T = min(len(ids) - 1, max_len)
        X[i, :T] = torch.tensor(ids[:T], dtype=torch.long)
        for t in range(T):
            if t >= sep_pos: Y[i, t] = ids[t + 1]
    return X, Y

model = MiniGPT(CFG).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f'Model: {total_params/1e6:.2f}M params')

samples = load_dataset('data/dataset.txt', tok, CFG['ctx_len'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-5
)

os.makedirs('data', exist_ok=True)
best_loss = float('inf')
patience_count = 0
start_time = time.time()

print(f'\nBắt đầu train {CFG["epochs"]} epochs...')
print('─' * 50)

for epoch in range(1, CFG['epochs'] + 1):
    model.train()
    X, Y = make_batch(samples, CFG['batch_size'], CFG['ctx_len'], DEVICE)
    T = X.shape[1]
    mask = torch.tril(torch.ones(T, T, device=DEVICE)).unsqueeze(0).unsqueeze(0)
    logits = model(X, mask)
    loss = F.cross_entropy(logits.view(-1, CFG['vocab_size']), Y.view(-1), ignore_index=-100)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    current_loss = loss.item()
    scheduler.step(current_loss)

    if epoch % 100 == 0:
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (CFG['epochs'] - epoch)
        saved = ' [SAVED]' if current_loss < best_loss else ''
        print(f'Epoch {epoch:5d}/{CFG["epochs"]} | Loss: {current_loss:.4f} | '
              f'Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m{saved}')

    if current_loss < best_loss:
        best_loss = current_loss
        torch.save({'model_state': model.state_dict(), 'cfg': CFG,
                    'epoch': epoch, 'loss': best_loss}, 'data/best_model.pt')
        patience_count = 0
    else:
        if epoch >= CFG['min_epoch']: patience_count += 1

    if patience_count >= CFG['patience']:
        print(f'\n Early stopping tại epoch {epoch}')
        break

print(f'\n Train xong! {(time.time()-start_time)/60:.1f} phút | Best loss: {best_loss:.4f}')
print('Model lưu tại: data/best_model.pt')
