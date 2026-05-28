import torch
import numpy as np
import os
import json

TABLE_LIMIT = 262143
FILE_LIMIT  = TABLE_LIMIT * 3

# QUANTIZATION (Luong tu hoa)

def quantize_tensor(tensor):
    """
    2D tensor -> per-row quantization (moi hang co scale rieng)
    1D tensor -> per-tensor quantization
    Returns: list of chunks, moi chunk la dict
    """
    t = tensor.float().numpy()

    if t.ndim == 1:
        # Per-tensor
        mv = max(abs(float(t.max())), abs(float(t.min()))) + 1e-8
        q  = np.clip(t * 127.0 / mv, -127, 127).astype(np.int8)
        return [{
            'data':    q,
            'scale':   mv,
            'per_row': False,
            'rows':    None,
            'cols':    None,
            'scales':  None,
        }]

    # Per-row
    R, C = t.shape
    scales = np.zeros(R, dtype=np.float64)
    qdata  = np.zeros((R, C), dtype=np.int8)
    for i in range(R):
        mv = max(abs(float(t[i].max())), abs(float(t[i].min()))) + 1e-8
        scales[i] = mv
        qdata[i]  = np.clip(t[i] * 127.0 / mv, -127, 127).astype(np.int8)

    # Chia thanh cac chunk khong vuot qua TABLE_LIMIT phan tu
    max_rows_per_chunk = max(1, TABLE_LIMIT // C)
    chunks = []
    r = 0
    while r < R:
        er = min(r + max_rows_per_chunk, R)
        chunks.append({
            'data':    qdata[r:er].flatten(),
            'scales':  scales[r:er].copy(),
            'scale':   None,
            'per_row': True,
            'rows':    er - r,
            'cols':    C,
        })
        r = er
    return chunks


def write_lua_chunk(f, table_var, name, chunk):
    f.write(f'{table_var}["{name}"] = {{\n')
    if chunk['per_row']:
        f.write(f'  rows = {chunk["rows"]},\n')
        f.write(f'  cols = {chunk["cols"]},\n')
        scales_str = ','.join(f'{s:.8f}' for s in chunk['scales'])
        f.write(f'  s = {{{scales_str}}},\n')
    else:
        f.write(f'  scale = {chunk["scale"]:.10f},\n')
    data = chunk['data']
    f.write(f'  n = {len(data)},\n')
    f.write(f'  d = {{')
    f.write(','.join(str(int(v)) for v in data))
    f.write('}\n}\n')

# EXPORT

def export(model_path='data/best_model.pt', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dang load model tu {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state = checkpoint['model_state']
    cfg   = checkpoint['cfg']

    print("\n=== WEIGHTS ===")
    for name in sorted(state.keys()):
        print(f"  {name}: {state[name].shape}")
    print("===============\n")

    with open(f'{output_dir}/config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    print(f"Config: {cfg}\n")

    all_chunks   = []
    total_params = 0
    shape_map    = {}

    for tensor_name, tensor in state.items():
        safe_name = tensor_name.replace('.', '_')
        shape     = list(tensor.shape)
        total_params += tensor.numel()
        shape_map[safe_name] = shape

        chunks = quantize_tensor(tensor)

        if len(chunks) == 1:
            all_chunks.append({
                'name':   safe_name,
                'chunk':  chunks[0],
                'parts':  1,
                'part':   0,
            })
        else:
            for i, ch in enumerate(chunks):
                all_chunks.append({
                    'name':  f'{safe_name}_p{i}',
                    'chunk': ch,
                    'parts': len(chunks),
                    'part':  i,
                })

    print(f"Tong params: {total_params:,} | Tong chunks: {len(all_chunks)}")

    # Gop chunks (tam thoi de moi file 3 bang de khong bi lag khi save vao MW)
    def flush_file(f_idx, table_groups):
        path = f'{output_dir}/weights_{f_idx:03d}.lua'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'-- MiniGPT weights file {f_idx}\n')
            f.write(f'-- {len(table_groups)} bang\n\n')
            for t_idx, tbl in enumerate(table_groups):
                var = f'MGPT_W{f_idx}_{t_idx}'
                f.write(f'{var} = {{}}\n')
                for item in tbl:
                    write_lua_chunk(f, var, item['name'], item['chunk'])
                f.write('\n')
        kb = os.path.getsize(path) / 1024
        print(f"  weights_{f_idx:03d}.lua — {kb:.1f} KB ({len(table_groups)} bang)")

    current_tables = []
    current_table  = []
    cur_table_size = 0
    f_idx  = 0
    f_map  = {}

    for item in all_chunks:
        size = len(item['chunk']['data'])

        if cur_table_size + size > TABLE_LIMIT and current_table:
            current_tables.append(current_table)
            current_table  = []
            cur_table_size = 0

        if len(current_tables) >= 3:
            flush_file(f_idx, current_tables)
            for t_i, tbl in enumerate(current_tables):
                for it in tbl:
                    f_map[it['name']] = (f_idx, t_i)
            f_idx += 1
            current_tables = []

        current_table.append(item)
        cur_table_size += size

    if current_table:
        current_tables.append(current_table)
    if current_tables:
        flush_file(f_idx, current_tables)
        for t_i, tbl in enumerate(current_tables):
            for it in tbl:
                f_map[it['name']] = (f_idx, t_i)
        f_idx += 1

    _write_index(output_dir, f_map, shape_map, cfg, f_idx)
    print(f"\nXuat xong {f_idx} file weights vao '{output_dir}/'")


def _write_index(output_dir, f_map, shape_map, cfg, total_files):
    path = f'{output_dir}/index.lua'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('-- MiniGPT Index\n-- Tu dong tao boi export.py\n\n')
        f.write('local IDX = {}\n\n')

        f.write('IDX.files = {}\n')
        for name, (fidx, tidx) in f_map.items():
            f.write(f'IDX.files["{name}"] = {fidx}\n')

        f.write('\nIDX.tables = {}\n')
        for name, (fidx, tidx) in f_map.items():
            f.write(f'IDX.tables["{name}"] = {tidx}\n')

        f.write('\nIDX.shapes = {}\n')
        for name, shape in shape_map.items():
            f.write(f'IDX.shapes["{name}"] = {{{",".join(str(s) for s in shape)}}}\n')

        f.write(f'\nIDX.cfg = {{\n')
        f.write(f'  vocab_size  = {cfg["vocab_size"]},\n')
        f.write(f'  embed_dim   = {cfg["embed_dim"]},\n')
        f.write(f'  ffn_dim     = {cfg["ffn_dim"]},\n')
        f.write(f'  n_layers    = {cfg["n_layers"]},\n')
        f.write(f'  n_heads     = {cfg["n_heads"]},\n')
        f.write(f'  n_kv_heads  = {cfg.get("n_kv_heads", 2)},\n')
        f.write(f'  ctx_len     = {cfg["ctx_len"]},\n')
        f.write(f'  total_files = {total_files},\n')
        f.write(f'  sep_id      = 4,\n')
        f.write('}\n\nreturn IDX\n')

    print(f"  index.lua — OK")


if __name__ == '__main__':
    export()
