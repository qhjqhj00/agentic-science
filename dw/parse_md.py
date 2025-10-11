import re
from typing import Dict, Any

def parse_markdown_sections(md_text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    lines = md_text.rstrip().splitlines()
    if lines and re.match(r"^\s*Generated on", lines[-1]):
        md_text = "\n".join(lines[:-1])
    # 1️⃣ meta
    meta_match = re.search(r'<!--(.*?)-->', md_text, re.DOTALL)
    result["meta"] = {"content": meta_match.group(1).strip() if meta_match else None, "idx": -1}

    # 2️⃣ title
    title_match = re.search(r'(?m)^\s*(.*?)\s*\n=+', md_text)
    result["title"] = {"content": title_match.group(1).strip() if title_match else None, "idx": -1}

    # 3️⃣ author
    author_match = re.search(r'={3,}\n+(.*?)\n+######\s*Abstract', md_text, re.DOTALL)
    result["author"] = {"content": author_match.group(1).strip() if author_match else None, "idx": -1}

    # 4️⃣ abstract
    abs_match = re.search(r'######\s*Abstract\s*(.*?)(?=\n(?:\d+\s+|[A-Za-z])|\Z)', md_text, re.DOTALL)
    result["abstract"] = {"content": abs_match.group(1).strip() if abs_match else None, "idx": -1}

    pattern = re.compile(
        r'(?m)^(?P<header>[^\n]+)\n-+\n(?P<body>.*?)(?=\n[^\n]+\n-+|\Z)',
        re.DOTALL
    )

    for m in pattern.finditer(md_text):
        header_raw = m.group("header").strip()
        body = m.group("body").strip()

        # 提取数字编号
        num_match = re.match(r"^\s*(\d+)\s+(.*)", header_raw)
        if num_match:
            idx = int(num_match.group(1))
            header = num_match.group(2).strip()
        else:
            idx = -1
            header = header_raw

        # 特殊处理 References
        if header.lower() == "references":
            result["References"] = {"content": body, "idx": idx}
        else:
            result[header] = {"content": body, "idx": idx}

    return result

if __name__ == "__main__":
    from src.utils import save_json
    with open("data/markdown/2510.08513.md", "r") as f:
        md_text = f.read()
    result = parse_markdown_sections(md_text)
    save_json(result, "data/2510.08513.json")