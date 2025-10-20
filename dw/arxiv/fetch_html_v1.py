import sqlite3
import os
from pathlib import Path
import time
from tqdm import tqdm
from src.tools import fetch_html_via_api
from src.utils import load_json

def convert_abs_to_html_url(abs_url):
    """
    Convert arxiv abstract URL to HTML URL
    Example: http://arxiv.org/abs/2411.01111v1 -> http://arxiv.org/html/2411.01111v1
    """
    return abs_url.replace('/abs/', '/html/')

async def fetch_and_save_html_for_abstracts(db_path="data/db/arxiv_v1.db", raw_dir="data/raw", batch_size=25):
    """
    Fetch HTML content for papers with status='abstract' and save to raw directory
    """
    # Load API configuration
    api_dict = load_json("config/api_dict.json")
    
    # Create raw directory if it doesn't exist
    os.makedirs(raw_dir, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get papers with status='abstract'
    cursor.execute("SELECT doc_id, url FROM arxiv_papers WHERE status = 'abstract'")
    abstract_papers = cursor.fetchall()
    
    print(f"Found {len(abstract_papers)} papers with status='abstract'")
    
    success_count = 0
    failed_count = 0
    
    # Process papers in batches
    for i in range(0, len(abstract_papers), batch_size):
        batch = abstract_papers[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(abstract_papers) + batch_size - 1)//batch_size}")
        
        # Prepare URLs for batch processing
        urls = []
        doc_ids = []
        for doc_id, abs_url in batch:
            html_url = convert_abs_to_html_url(abs_url)
            urls.append(html_url)
            doc_ids.append(doc_id)
        
        try:
            # Fetch all URLs in the batch
            response = await fetch_html_via_api(
                urls,
                api_dict["search_token"],
                api_dict["web_unlocker_api"],
            )
            
            html_contents = response["results"]
            
            # Process each result in the batch
            for doc_id, url in zip(doc_ids, urls):
                html_content = html_contents.get(url)
                
                if html_content:
                    # Save HTML to file
                    html_file_path = os.path.join(raw_dir, f"{doc_id}.html")
                    try:
                        with open(html_file_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        # Update database record
                        cursor.execute("""
                            UPDATE arxiv_papers 
                            SET status = 'fetched', html_path = ? 
                            WHERE doc_id = ?
                        """, (html_file_path, doc_id))
                        
                        success_count += 1
                        
                    except Exception as e:
                        print(f"Failed to save HTML for {doc_id}: {e}")
                        failed_count += 1
                else:
                    print(f"No HTML content received for {doc_id}")
                    failed_count += 1
            conn.commit()
        except Exception as e:
            print(f"Failed to fetch batch: {e}")
            failed_count += len(batch)
        
        # Add small delay between batches to be respectful to the server
        time.sleep(1)
    
    # Commit changes
    
    conn.close()
    
    print(f"Successfully fetched and saved {success_count} HTML files")
    print(f"Failed to fetch {failed_count} papers")

if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_and_save_html_for_abstracts())


