from src.utils import *
from html_to_markdown import convert_to_markdown
from dw.arxiv.parse import *

import sqlite3
import json
import os
from datetime import datetime
import re

def create_database(db_path="data/db/arxiv.db"):
    """Create SQLite database with arxiv papers table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS arxiv_papers (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            subject TEXT,
            html_path TEXT,
            markdown_path TEXT,
            parsed_path TEXT,
            status TEXT,
            released_date DATETIME,
            first_seen_at DATETIME,
            last_modified DATETIME
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_date_from_arxiv_id(arxiv_id):
    """Extract release date from arxiv ID"""
    # Pattern for new format: YYMM.NNNNN
    date = arxiv_id[:7]
    year = 2000 + int(date[:2])
    month = int(date[2:4])
    day = int(date[5:7])
    return datetime(year, month, day).isoformat()

def fetch_all_ids(cursor):    
    # Get existing arxiv IDs from database to avoid re-processing
    cursor.execute('SELECT doc_id FROM arxiv_papers')
    existing_records = cursor.fetchall()
    existing_arxiv_ids = {record[0] for record in existing_records}
    return existing_arxiv_ids


def process_and_save_parsed_content(paper):
    """Process markdown content and save parsed JSON"""
    arxiv_id = paper['id']
    markdown_path = paper.get('markdown_path', '')
    parsed_path = f"data/parsed/{arxiv_id}.json"
    
    # Create parsed directory if it doesn't exist
    os.makedirs("data/parsed", exist_ok=True)
    
    if markdown_path and os.path.exists(markdown_path):
        try:
            # Load markdown content
            markdown_text = load_txt(markdown_path)
            if markdown_text.find("No HTML for") != -1:
                return ""
            # Parse markdown sections
            parsed_text = parse_markdown_sections(markdown_text)
            
            # Save parsed content
            save_json(parsed_text, parsed_path)
            
            return parsed_path
        except Exception as e:
            print(f"Error processing {arxiv_id}: {e}")
            return ""
    
    return ""

def build_database_from_jsonl(jsonl_path="data/arxiv_list_content.jsonl", db_path="data/db/arxiv.db"):
    """Build database from JSONL file"""
    
    # Create database
    create_database(db_path)
    
    # Load papers from JSONL
    if not os.path.exists(jsonl_path):
        print(f"JSONL file not found: {jsonl_path}")
        return
    
    papers = load_jsonl(jsonl_path)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    current_time = datetime.now().isoformat()
    
    for paper in papers:
        doc_id = paper['id']
        title = paper.get('title', '')
        author = paper.get('authors', '')
        subject = json.dumps(paper.get('subject_split', []))  # Store as JSON string
        html_path = paper.get('html_path', '')
        markdown_path = paper.get('markdown_path', '')
        
        # Process and save parsed content
        parsed_path = process_and_save_parsed_content(paper)
        if parsed_path == "":
            status = "not_parsed"
        else:
            status = "parsed"
        # Extract release date from arxiv ID
        released_date = extract_date_from_arxiv_id(doc_id)
        
        # Insert or update record
        cursor.execute('''
            INSERT OR REPLACE INTO arxiv_papers 
            (doc_id, title, author, subject, html_path, markdown_path, parsed_path, 
             status, released_date, first_seen_at, last_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (doc_id, title, author, subject, html_path, markdown_path, parsed_path,
              status, released_date, current_time, current_time))

    conn.commit()
    conn.close()
    
    print(f"Database built successfully with {len(papers)} papers")



if __name__ == "__main__":
    # Print 3 random complete records from the database
    import random
    # build_database_from_jsonl()

    conn = sqlite3.connect("data/db/arxiv.db")
    cursor = conn.cursor()

    # Get 3 random records from the database
    # Count papers by status
    cursor.execute("SELECT status, COUNT(*) FROM arxiv_papers GROUP BY status")
    status_counts = cursor.fetchall()
    
    print("Status counts:")
    for status, count in status_counts:
        print(f"{status}: {count}")
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
    total_count = cursor.fetchone()[0]
    print(f"Total papers: {total_count}")

    
    # cursor.execute('SELECT * FROM arxiv_papers WHERE doc_id = "2510.12781"')
    # all_complete_records = cursor.fetchall()
    # print(all_complete_records)
    
    
    conn.close()
    
    


