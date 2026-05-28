import json, os

def export_tokenizer(tok_path='data/tokenizer.json', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    with open(tok_path, encoding='utf-8') as f:
        data = json.load(f)

    vocab   = data['vocab']
    merges  = data['merges']
    special = data.get('special_tokens', {'PAD':0,'UNK':1,'BOS':2,'EOS':3,'SEP':4})

    with open(f'{output_dir}/tokenizer_vocab.lua', 'w', encoding='utf-8') as out:
        out.write('-- MiniGPT Tokenizer\nlocal T = {}\n\n')
        out.write('-- Special tokens\n')
        for k in ['PAD','UNK','BOS','EOS','SEP']:
            out.write(f'T.{k} = {special.get(k, ["PAD","UNK","BOS","EOS","SEP"].index(k))}\n')
        out.write('\n')

        out.write('T.vocab = {\n')
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            safe = token.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n')
            out.write(f'  ["{safe}"] = {idx},\n')
        out.write('}\n\n')

        out.write('T.inv_vocab = {}\n')
        out.write('for tok, id in pairs(T.vocab) do T.inv_vocab[id] = tok end\n\n')

        out.write('T.merges = {}\n')
        for key_str, rank in sorted(merges.items(), key=lambda x: x[1]):
            try:
                pair = eval(key_str)
                t1 = str(pair[0]).replace('\\','\\\\').replace('"','\\"')
                t2 = str(pair[1]).replace('\\','\\\\').replace('"','\\"')
                out.write(f'T.merges["{t1}#{t2}"] = {rank}\n')
            except:
                pass

        out.write('\nreturn T\n')

    print(f"Tokenizer xuat xong: {output_dir}/tokenizer_vocab.lua")
    print(f"  Vocab: {len(vocab)} | Merges: {len(merges)}")

if __name__ == '__main__':
    export_tokenizer()
