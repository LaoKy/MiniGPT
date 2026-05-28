# MiniGPT ~11.9M

**MiniGPT** là một chatbot AI nhỏ gọn chạy hoàn toàn bên trong game **Mini World** thông qua cơ chế Lua scripts. Mô hình được huấn luyện để trò chuyện bằng tiếng Việt, hỗ trợ người chơi về các chủ đề Mini World và đời sống.

---

## Kiến trúc mô hình

MiniGPT sử dụng kiến trúc **Decoder-only Transformer** phong cách **Llama**, bao gồm:

| Thành phần | Giá trị |
|---|---|
| **Loại** | Decoder-only Transformer |
| **Số tham số** | ~11.9M |
| **Embedding dimension** | 352 |
| **FFN dimension** | 1,408 (4× embed_dim) |
| **Số layers** | 6 |
| **Số attention heads** | 4 (query) / 2 (key-value) |
| **Context length** | 96 tokens |
| **Vocab size** | 1,811 |
| **Dropout** | 0.1 |
| **Position encoding** | Rotary Position Embeddings (RoPE) |
| **Normalization** | RMSNorm |
| **FFN activation** | SwiGLU (`silu(w1(x)) * w3(x)`) |
| **Attention** | Grouped-Query Attention (GQA) |
| **Weight tying** | Logits = `x @ embed_weight.T` (không có lm_head riêng) |
| **Bias** | Không sử dụng bias ở bất kỳ Linear layer nào |

### Chi tiết từng block (6 blocks)

```
Embedding(1811, 352)
  ↓
TransformerBlock × 6:
  ├── RMSNorm → GQAttention(4H, 2KV, RoPE) → Residual
  └── RMSNorm → SwiGLU(352 → 1408 → 352) → Residual
  ↓
RMSNorm
  ↓
Weight-tied projection (× embed_weight.T)
```

### Các layer cụ thể trong mỗi TransformerBlock

- **`norm1`**: RMSNorm(352)
- **`attn.wq`**: Linear(352, 352, bias=False) — query projection
- **`attn.wk`**: Linear(352, 176, bias=False) — key projection (2 KV heads × 88 head_dim)
- **`attn.wv`**: Linear(352, 176, bias=False) — value projection
- **`attn.wo`**: Linear(352, 352, bias=False) — output projection
- **`rope`**: Rotary Position Embeddings(88 dim, max_len=512)
- **`norm2`**: RMSNorm(352)
- **`ffn.w1`**: Linear(352, 1408, bias=False) — gate projection
- **`ffn.w2`**: Linear(1408, 352, bias=False) — down projection
- **`ffn.w3`**: Linear(352, 1408, bias=False) — up projection

---

## Cấu trúc thư mục

```
MiniGPT_11.9M/
│
├── README.md                    # Tệp này
├── note.txt                     # Ghi chú từ tác giả
│
├── train.py                     # Huấn luyện mô hình từ đầu
├── tokenizer.py                 # Huấn luyện BPE tokenizer
├── export.py                    # Lượng tử hóa & xuất weights → Lua
├── export_tokenizer.py          # Xuất tokenizer → Lua
├── convert_weights.py           # Chuyển output/ → scripts/
│
├── data/
│   ├── dataset.txt              # 500+ câu hỏi-đáp tiếng Việt (Q | A)
│   ├── tokenizer.json           # Tokenizer đã train (tạo bởi tokenizer.py)
│   └── best_model.pt            # Model checkpoint tốt nhất (tạo bởi train.py)
│
├── output/                      # Đầu ra từ export.py & export_tokenizer.py
│   ├── config.json              # Cấu hình mô hình
│   ├── index.lua                # Index map cho weights
│   ├── tokenizer_vocab.lua      # Tokenizer dạng Lua
│   └── weights_XXX.lua          # 17 file weights (lượng tử hóa int8)
│
└── scripts/                     # Scripts sẵn sàng copy vào Mini World
    ├── mg_main.lua              # Engine inference chính (~869 dòng)
    ├── mg_tick.lua              # Coroutine tick handler
    ├── mg_tok.lua               # Tokenizer Lua
    ├── mg_idx.lua               # Index Lua
    ├── mg_w0.lua ... mg_w16.lua # 17 file weights Lua
```

---

## Quy trình huấn luyện & xuất

1. **`tokenizer.py`** — Train BPE tokenizer trên dataset → `data/tokenizer.json`
2. **`train.py`** — Train mô hình với Cross-entropy loss (chỉ tính trên phần answer) + AdamW + ReduceLROnPlateau + Early Stopping → `data/best_model.pt`
3. **`export_tokenizer.py`** — Chuyển tokenizer → Lua → `output/tokenizer_vocab.lua`
4. **`export.py`** — Lượng tử hóa weights **per-row int8**, chia thành ~17 file + index → `output/`
5. **`convert_weights.py`** — Đổi tên biến Lua global, copy từ `output/` → `scripts/`

### Lượng tử hóa

- **2D weights**: Per-row int8 — mỗi hàng có scale riêng. Lua lưu dạng `{rows, cols, s=[scales], n, d=[int8 data]}`.
- **1D weights**: Per-tensor int8 — một scale cho toàn bộ tensor.
- Dequantize trong Lua: `value = int8_value * scale / 127`

### Dataset

File `data/dataset.txt` chứa 500+ cặp Q&A tiếng Việt, mỗi dòng định dạng:
```
câu hỏi | câu trả lời
```

Mô hình được huấn luyện với **causal mask + ignore_index=-100**: chỉ các token **answer** mới được tính loss, token câu hỏi bị bỏ qua.

---

## Cách chạy

### Train model mới

```bash
python tokenizer.py          # Bước 1: Train tokenizer
python train.py              # Bước 2: Train model (cần GPU nếu có)
python export_tokenizer.py   # Bước 3: Xuất tokenizer
python export.py             # Bước 4: Xuất weights
python convert_weights.py    # Bước 5: Chuyển sang scripts/
```

### Chạy trong Mini World

Copy các tệp từ `scripts/` vào Mini World theo thứ tự:

```
mg_w0 → mg_w1 → ... → mg_w16
mg_tok → mg_idx → mg_main → mg_tick
```

Trong game, gõ lệnh:

```
!ai [câu hỏi của bạn]
```

### File Lua quan trọng

| File | Chức năng |
|---|---|
| `mg_main.lua` | Engine inference: BPE tokenizer, dequantize, matrix-vector multiply, RMSNorm, softmax, RoPE, SiLU, GQA attention, SwiGLU FFN, top-k sampling, vòng lặp sinh |
| `mg_tick.lua` | Xử lý coroutine, tạm dừng/tiếp tục generation theo frame để không lag game |

---

## Hyperparameters huấn luyện

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=200, min_lr=1e-5) |
| Gradient clipping | 1.0 |
| Max epochs | 20,000 |
| Early stopping | Patience=800 (sau min_epoch=1000) |

---

## Ảnh Review trong Mini World

> Người dùng hỏi `!ai hello`

![Review 1](https://i.ibb.co/2190Mhgt/654659521-2158720534894745-5395433327503041230-n.png)

> Người dùng hỏi `!ai bạn có thông minh không`

![Review 2](https://i.ibb.co/35MgbqjC/656164665-953609690515791-1391295739320689249-n.jpg)

> Người dùng hỏi `!ai bạn đang làm gì`

![Review 3](https://i.ibb.co/xKyLRmxM/658224868-953609710515789-180585153797779312-n.jpg)

---

## Liên hệ

Tác giả: [Facebook](https://www.facebook.com/profile.php?id=100076003056977)
