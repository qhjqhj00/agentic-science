from bs4 import BeautifulSoup
import pandas as pd
import os
from src.tools import fetch_html_via_api
from src.utils import load_json
from html_to_markdown import convert_to_markdown
from dw.arxiv.parse import *
from dw.arxiv.build_db import *

api_dict = load_json("config/api_dict.json")

async def get_list_page(url):
    response = await fetch_html_via_api(
        url,
        api_dict["search_token"],
        api_dict["web_unlocker_api"],
        use_cache=False,
    )

    fetched_content = response["results"][url]
    soup = BeautifulSoup(fetched_content, features='html.parser')
    content = soup.dl
    date = soup.find('h3')   

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

def insert_paper_to_db(paper, cursor):
    paper_id = paper["id"]
    title = paper["title"]
    authors = paper["authors"]
    subjects = paper["subjects"]
    html_path = paper["html_path"]
    markdown_path = paper["markdown_path"]
    parsed_path = paper["parsed_path"]
    if parsed_path == "":
        status = "not_parsed"
    else:
        status = "parsed"
    current_time = datetime.now().isoformat()
    released_date = extract_date_from_arxiv_id(paper_id)
    cursor.execute('''
            INSERT OR REPLACE INTO arxiv_papers 
            (doc_id, title, author, subject, html_path, markdown_path, parsed_path, 
             status, released_date, first_seen_at, last_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (paper_id, title, authors, subjects, html_path, markdown_path, parsed_path,
              status, released_date, current_time, current_time))
    
async def get_full_content(papers_list, db_path="data/db/arxiv.db"):
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
    os.makedirs("data/parsed", exist_ok=True)
    
    # Generate URLs for all papers
    
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    existing_arxiv_ids = fetch_all_ids(cursor)
    # Filter out papers that already exist in database
    papers_list = [paper for paper in papers_list if paper['id'] not in existing_arxiv_ids]
    
    if not papers_list:
        print("All papers already exist in database, skipping content fetch")
        return []
    
    print(f"Fetching content for {len(papers_list)} new papers")

    urls = [f"https://arxiv.org/html/{paper['id']}" for paper in papers_list]
    
    
    # Fetch all pages concurrently
    response = await fetch_html_via_api(urls, api_dict["search_token"], api_dict["web_unlocker_api"])
    html_contents = response["results"]
    
    # Process each HTML content
    newly_added_ids = []
    for paper in papers_list:
        arxiv_id = paper['id']

        html_path = f"data/raw/{arxiv_id}.html"
        markdown_path = f"data/markdown/{arxiv_id}.md"
        parsed_path = f"data/parsed/{arxiv_id}.json"
        
        # Get HTML content from the dictionary using the URL as key
        url = f"https://arxiv.org/html/{arxiv_id}"
        html = html_contents.get(url)
        
        if html is None:
            continue
        
        # Save HTML content
        save_txt(html, html_path)
        
        # Convert to markdown
        try:
            # Process HTML for better markdown conversion
            processed_html = replace_mathml_with_annotation(html)
            soup = BeautifulSoup(processed_html, "lxml")
            markdown_text = convert_to_markdown(soup)
            markdown_text = unescape_markdown(markdown_text)
            
            # Save markdown content
            save_txt(markdown_text, markdown_path)
            
            
            paper['html_path'] = html_path
            paper['markdown_path'] = markdown_path

            if markdown_text.find("No HTML for") != -1:
                paper['parsed_path'] = ""
            else:
                parsed_text = parse_markdown_sections(markdown_text)
                save_json(parsed_text, parsed_path)
                paper['parsed_path'] = parsed_path

            newly_added_ids.append(arxiv_id)
            insert_paper_to_db(paper, cursor)
        except Exception as e:
            print(f"Error converting {arxiv_id} to markdown: {e}")
            paper['html_path'] = html_path
            paper['markdown_path'] = ""
            paper['parsed_path'] = ""
    conn.commit()
    conn.close()
    print(f"added {len(newly_added_ids)} papers to database")
    return newly_added_ids

if __name__ == "__main__":
    from config.list_urls import get_list_urls_skip
    import asyncio
    url = get_list_urls_skip("cs.CL", 25, 25)
    res = asyncio.run(get_list_page(url))
    res = asyncio.run(get_full_content(res))
    print(res)
