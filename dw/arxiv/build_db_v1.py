import sqlite3
import json
import os
import re
from datetime import datetime
from pathlib import Path

def create_arxiv_v1_db(db_path="data/db/arxiv_v1.db"):
    """Create the arxiv_v1 database with the required schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS arxiv_papers')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS arxiv_papers (
            doc_id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            author TEXT,
            subject TEXT,
            abstract TEXT, 
            html_path TEXT,
            markdown_path TEXT,
            parsed_path TEXT,
            status TEXT,
            publish_date TEXT,
            first_seen_at TEXT,
            last_modified TEXT,
            version TEXT
        )
    ''')
    
    conn.commit()
    return conn, cursor

def parse_arxiv_id_and_version(url):
    """Parse arxiv ID and version from URL like http://arxiv.org/abs/2410.16644v1"""
    pattern = r'http://arxiv\.org/abs/(\d+\.\d+)(?:v(\d+))?'
    match = re.match(pattern, url)
    if match:
        paper_id = match.group(1)
        version = match.group(2) if match.group(2) else "1"
        return paper_id, version
    return None, None

def get_existing_html_files(raw_dir="data/raw"):
    """Get list of existing paper IDs from HTML files in raw directory"""
    existing_ids = set()
    if os.path.exists(raw_dir):
        for filename in os.listdir(raw_dir):
            if filename.endswith('.html'):
                paper_id = filename.replace('.html', '')
                existing_ids.add(paper_id)
    return existing_ids

def process_api_json_files(api_dir="data/api_by_date", existing_html_ids=None):
    """Process all JSON files in api_by_date directory"""
    if existing_html_ids is None:
        existing_html_ids = set()
    
    papers_data = []
    
    for filename in os.listdir(api_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(api_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'items' in data:
                    for item in data['items']:
                        paper_id, version = parse_arxiv_id_and_version(item.get('id', ''))
                        if paper_id:
                            # Extract authors as string
                            authors_list = item.get('authors', [])
                            authors_str = '; '.join([author.get('name', '') for author in authors_list])
                            
                            # Determine status and paths
                            html_path = ""
                            markdown_path = ""
                            parsed_path = ""
                            status = "abstract"
                            
                            if paper_id in existing_html_ids:
                                html_path = f"data/raw/{paper_id}.html"
                                status = "fetched"
                            
                            paper_data = {
                                'doc_id': paper_id,
                                'url': item.get('url', ''),
                                'title': item.get('title', '').strip(),
                                'author': authors_str,
                                'subject': 'cs.AI',  # Not available in the JSON structure shown
                                'abstract': item.get('content_html', ''),
                                'html_path': html_path,
                                'markdown_path': markdown_path,
                                'parsed_path': parsed_path,
                                'status': status,
                                'publish_date': item.get('date_published', ''),
                                'last_modified': item.get('date_modified', ''),
                                'version': version
                            }
                            papers_data.append(paper_data)
                            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return papers_data

def insert_papers_to_db(papers_data, cursor):
    """Insert papers data into the database"""
    current_time = datetime.now().isoformat()
    
    for paper in papers_data:
        cursor.execute('''
            INSERT OR REPLACE INTO arxiv_papers 
            (doc_id, url, title, author, subject, abstract, html_path, markdown_path, 
             parsed_path, status, publish_date, first_seen_at, last_modified, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            paper['doc_id'],
            paper['url'],
            paper['title'],
            paper['author'],
            paper['subject'],
            paper['abstract'],
            paper['html_path'],
            paper['markdown_path'],
            paper['parsed_path'],
            paper['status'],
            paper['publish_date'],
            current_time,
            paper['last_modified'],
            paper['version']
        ))

def main():
    """Main function to build the arxiv_v1 database"""
    # Create database directory if it doesn't exist
    os.makedirs("data/db", exist_ok=True)
    
    # Create database and get connection
    conn, cursor = create_arxiv_v1_db()
    
    # Get existing HTML files
    existing_html_ids = get_existing_html_files()
    print(f"Found {len(existing_html_ids)} existing HTML files")
    
    # Process API JSON files
    papers_data = process_api_json_files(existing_html_ids=existing_html_ids)
    print(f"Processed {len(papers_data)} papers from API data")
    
    # Insert data into database
    insert_papers_to_db(papers_data, cursor)
    conn.commit()
    
    # Print statistics
    cursor.execute("SELECT status, COUNT(*) FROM arxiv_papers GROUP BY status")
    status_counts = cursor.fetchall()
    print("Database statistics:")
    for status, count in status_counts:
        print(f"  {status}: {count}")
    
    cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
    total_count = cursor.fetchone()[0]
    print(f"Total papers: {total_count}")
    
    conn.close()
    print("Database build completed successfully!")

if __name__ == "__main__":
    # main()
    
    # 随机打印3条数据看看
    conn = sqlite3.connect("data/db/arxiv.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status, COUNT(*) FROM arxiv_papers GROUP BY status")
    status_counts = cursor.fetchall()
    print("Database statistics:")
    for status, count in status_counts:
        print(f"  {status}: {count}")
    # cursor.execute("SELECT * FROM arxiv_papers ORDER BY RANDOM() LIMIT 3")
    # sample_papers = cursor.fetchall()
    
    # print("\n随机抽取的3条数据:")
    # for i, paper in enumerate(sample_papers, 1):
    #     print(paper)
    #     print(f"\n第{i}条:")
    #     print(f"  doc_id: {paper[0]}")
    #     print(f"  url: {paper[1]}")
    #     print(f"  title: {paper[2]}")
    #     print(f"  author: {paper[3]}")
    #     print(f"  subject: {paper[4]}")
    #     print(f"  abstract: {paper[5][:100]}..." if paper[5] else "  abstract: None")
    #     print(f"  status: {paper[9]}")
    #     print(f"  version: {paper[13]}")
    
    conn.close()
