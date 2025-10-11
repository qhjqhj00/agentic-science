from bs4 import BeautifulSoup
import pandas as pd
from src.tools import get_multiple_pages
import re
import os
from html_to_markdown import convert_to_markdown
from src.tools import get_one_page, fetch_html_via_api
from src.utils import load_json

api_dict = load_json("config/api_dict.json")

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

async def get_list_page(url):
    response = await fetch_html_via_api(url, api_dict["search_token"], api_dict["web_unlocker_api"])
    fetched_content = response["results"][url]
    soup = BeautifulSoup(fetched_content, features='html.parser')
    content = soup.dl
    date = soup.find('h3')   # TODO: 列表为空的就return none
    try:
        list_ids = content.find_all('a', title = 'Abstract')
        list_title = content.find_all('div', class_ = 'list-title mathjax')
        list_authors = content.find_all('div', class_ = 'list-authors')
        list_subjects = content.find_all('div', class_ = 'list-subjects')
    except Exception as e:
        print(f"Error parsing list page: {e}")
        return []
    list_subject_split = []
    for subjects in list_subjects:
        # print(subjects.text)
        subjects = subjects.text.split(':', maxsplit=1)[1]
        subjects = subjects.replace('\n\n', '')
        subjects = subjects.replace('\n', '')
        subject_split = subjects.split('; ')
        list_subject_split.append(subject_split)

    items = []
    for i, paper in enumerate(zip(list_ids, list_title, list_authors, list_subjects, list_subject_split)):
        _id = paper[0].text.split(":")[1].strip().lstrip()
        _title = paper[1].text.replace("Title:", "").strip().lstrip()
        _authors = paper[2].text
        items.append([_id, _title, _authors, paper[3].text, paper[4]])
    name = ['id', 'title', 'authors', 'subjects', 'subject_split']
    paper = pd.DataFrame(columns=name,data=items)
    return paper.to_dict(orient='records')

async def get_abstracts(papers_list, existing_arxiv_ids: list = []):
    """
    Asynchronously fetch abstracts for all papers in the list and add them to the papers.
    
    Args:
        papers_list: List of paper dictionaries from get_list_page output
    
    Returns:
        List of paper dictionaries with added 'abstract' and 'comment' fields
    """
    
    # Generate URLs for all papers
    urls = [f"https://arxiv.org/abs/{paper['id']}" for paper in papers_list]
    
    # Fetch all pages concurrently
    html_contents = await get_multiple_pages(urls)
    # Process each HTML content and add to papers list
    for i, (paper, html) in enumerate(zip(papers_list, html_contents)):
        if paper['id'] in existing_arxiv_ids:
            continue
        if html is None:
            paper['abstract'] = ""
            paper['comment'] = ""
            continue
            
        soup = BeautifulSoup(html, features='html.parser')
        
        # Extract abstract
        abstract = ""
        for line in soup.find_all("meta"):
            if line.get("property") == "og:description":
                abstract = line.get("content")
                break
        
        # Extract comment
        comment_element = soup.find('td', class_='tablecell comments mathjax')
        if comment_element:
            comment = comment_element.text.replace("Project page at this https URL", "")
        else:
            comment = ""
        
        # Add to paper dictionary
        paper['abstract'] = abstract
        paper['comment'] = comment
    
    return papers_list

async def get_full_content(papers_list, existing_arxiv_ids: list = []):
    """
    Asynchronously fetch full HTML content for all papers in the list and convert to markdown.
    
    Args:
        papers_list: List of paper dictionaries from get_list_page output
    
    Returns:
        List of paper dictionaries with added 'html_path' and 'markdown_path' fields
    """

    
    # Create directories if they don't exist
    os.makedirs("data/html", exist_ok=True)
    os.makedirs("data/markdown", exist_ok=True)
    
    # Generate URLs for all papers
    urls = [f"https://arxiv.org/html/{paper['id']}" for paper in papers_list]
    
    # Fetch all pages concurrently
    response = await fetch_html_via_api(urls, api_dict["search_token"], api_dict["web_unlocker_api"])
    html_contents = response["results"]
    
    # Process each HTML content
    for paper in papers_list:
        arxiv_id = paper['id']
        if arxiv_id in existing_arxiv_ids:
            continue
        html_path = f"data/html/{arxiv_id}.html"
        markdown_path = f"data/markdown/{arxiv_id}.md"
        
        # Get HTML content from the dictionary using the URL as key
        url = f"https://arxiv.org/html/{arxiv_id}"
        html = html_contents.get(url)
        
        if html is None:
            paper['html_path'] = ""
            paper['markdown_path'] = ""
            continue
        
        # Save HTML content
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        # Convert to markdown
        try:
            # Process HTML for better markdown conversion
            processed_html = replace_mathml_with_annotation(html)
            soup = BeautifulSoup(processed_html, "lxml")
            markdown_text = convert_to_markdown(soup)
            markdown_text = unescape_markdown(markdown_text)
            
            # Save markdown content
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            paper['html_path'] = html_path
            paper['markdown_path'] = markdown_path
            
        except Exception as e:
            print(f"Error converting {arxiv_id} to markdown: {e}")
            paper['html_path'] = html_path
            paper['markdown_path'] = ""
    
    return papers_list

def get_existing_arxiv_ids(markdown_dir: str = "data/markdown") -> list:
    """
    Get all arxiv IDs from existing markdown files in the specified directory.
    
    Args:
        markdown_dir: Directory containing markdown files (default: "data/markdown")
    
    Returns:
        List of arxiv IDs extracted from markdown filenames
    """
    import os
    
    if not os.path.exists(markdown_dir):
        return []
    
    arxiv_ids = []
    for filename in os.listdir(markdown_dir):
        if filename.endswith('.md'):
            # Remove .md extension to get arxiv ID
            arxiv_id = filename[:-3]
            arxiv_ids.append(arxiv_id)
    
    # Extract dates from arxiv IDs and return as sorted list
    dates = set()
    for arxiv_id in arxiv_ids:
        # Extract date part (first 7 characters: YYMM.DD)
        if len(arxiv_id) >= 7 and '.' in arxiv_id:
            date_part = arxiv_id[:7]  # e.g., "2510.02"
            dates.add(date_part)
    
    return arxiv_ids, sorted(list(dates))


if __name__ == "__main__":
    import asyncio
    from config.list_urls import get_list_urls, get_list_urls_skip
    from src.utils import save_jsonl
    print(get_existing_arxiv_ids())
    url = get_list_urls_skip("cs.CL", 25, 1000)
    list_content = asyncio.run(get_list_page(url))
    print(list_content)
    # parsed_content = asyncio.run(get_abstracts(list_content[:3]))
    parsed_content = asyncio.run(get_full_content(parsed_content))
    save_jsonl(parsed_content, "data/arxiv_list_content.jsonl")
