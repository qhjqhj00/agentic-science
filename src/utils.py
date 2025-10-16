import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_txt(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def save_txt(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)
