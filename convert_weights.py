# convert_weights.py - Chuyen output/ -> scripts/ cho Mini World
import os

def convert_weights_file(input_path, output_path, file_idx):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('local MGPT_', 'MGPT_')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'-- Script: MGPT_W{file_idx} (multi-table)\n\n')
        f.write(content)
    kb = os.path.getsize(output_path) / 1024
    print(f"  {output_path} ({kb:.1f} KB)")

def convert_tokenizer(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('local T = {}', 'MGPT_TOK = {}')
    for key in ['vocab','inv_vocab','merges','PAD','UNK','BOS','EOS','SEP']:
        content = content.replace(f'T.{key}', f'MGPT_TOK.{key}')
    content = content.replace('return T', '')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('-- Script: MGPT_TOK\n\n')
        f.write(content)
    kb = os.path.getsize(output_path) / 1024
    print(f"  {output_path} ({kb:.1f} KB)")

def convert_index(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('local IDX = {}', 'MGPT_IDX = {}')
    content = content.replace('IDX.', 'MGPT_IDX.')
    content = content.replace('IDX[', 'MGPT_IDX[')
    content = content.replace('return IDX', '')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('-- Script: MGPT_IDX\n\n')
        f.write(content)
    kb = os.path.getsize(output_path) / 1024
    print(f"  {output_path} ({kb:.1f} KB)")

if __name__ == '__main__':
    os.makedirs('scripts', exist_ok=True)

    weight_files = sorted([
        f for f in os.listdir('output')
        if f.startswith('weights_') and f.endswith('.lua')
    ])
    print(f"Tim thay {len(weight_files)} file weights")

    for i, fname in enumerate(weight_files):
        convert_weights_file(f'output/{fname}', f'scripts/mg_w{i}.lua', i)

    convert_tokenizer('output/tokenizer_vocab.lua', 'scripts/mg_tok.lua')
    convert_index('output/index.lua', 'scripts/mg_idx.lua')

    print(f"\nXong! Tao ra {len(weight_files)} weight files + mg_tok + mg_idx")
    print("\nThu tu load trong Mini World:")
    for i in range(len(weight_files)):
        print(f"  mg_w{i}")
    print("  mg_tok\n  mg_idx\n  mg_main\n  mg_tick")
