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

def unescape_markdown(text: str) -> str:
    """
    Unescape markdown characters that were escaped.
    
    Args:
        text: The text with escaped markdown characters
        
    Returns:
        Text with unescaped characters
    """
    # Dictionary of escaped characters to their unescaped versions
    escape_map = {
        r'\\': '\\',  # Escaped backslash
        r'\-': '-',   # Escaped dash
        r'\_': '_',   # Escaped underscore
        r'\*': '*',   # Escaped asterisk
        r'\[': '[',   # Escaped left bracket
        r'\]': ']',   # Escaped right bracket
        r'\(': '(',   # Escaped left parenthesis
        r'\)': ')',   # Escaped right parenthesis
        r'\{': '{',   # Escaped left brace
        r'\}': '}',   # Escaped right brace
        r'\#': '#',   # Escaped hash
        r'\+': '+',   # Escaped plus
        r'\.': '.',   # Escaped dot
        r'\!': '!',   # Escaped exclamation
        r'\|': '|',   # Escaped pipe
        r'\`': '`',   # Escaped backtick
        r'\~': '~',   # Escaped tilde
        r'\^': '^',   # Escaped caret
        r'\<': '<',   # Escaped less than
        r'\>': '>',   # Escaped greater than
        r'\"': '"',   # Escaped quote
        r"\'": "'",   # Escaped single quote
    }
    # Remove arxiv.org HTML links pattern
    
    result = text

    for escaped, unescaped in escape_map.items():
        result = result.replace(escaped, unescaped)

    result = re.sub(r'\(https://arxiv\.org/html/[^)]*\)', '', result) # Remove arxiv.org HTML links pattern

    return result

def replace_mathml_with_annotation(html: str) -> str:
    # 匹配 <math ...> ... </math>
    pattern = re.compile(r"<math\b.*?>.*?</math>", re.DOTALL)
    # Remove ltx_tocentry elements
    html = re.sub(r'<li[^>]*class="[^"]*ltx_tocentry[^"]*"[^>]*>.*?</li>', '', html, flags=re.DOTALL)
    def repl(match):
        block = match.group(0)
        # 尝试提取 <annotation encoding="application/x-tex">...</annotation>
        m = re.search(r'<annotation[^>]*encoding="application/x-tex"[^>]*>(.*?)</annotation>', block, re.DOTALL)
        if m:
            content = m.group(1).strip()
            return f"${content}$"
        return block  # 如果没找到，保持原样

    return pattern.sub(repl, html)
