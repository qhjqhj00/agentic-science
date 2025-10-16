import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.arxiv import process_paper
from src.utils import *

def process_papers_batch(db_path="data/db/arxiv.db"):
    """Process all papers in the database with batch size=4"""
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all unprocessed papers
    cursor.execute("SELECT doc_id FROM arxiv_papers WHERE status != 'processed'")
    # cursor.execute("SELECT doc_id FROM arxiv_papers")
    paper_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Found {len(paper_ids)} papers to process")
    
    # Create parsed directory if it doesn't exist
    os.makedirs("data/parsed", exist_ok=True)
    
    # Process papers in batches of 4
    batch_size = 4
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: papers {batch}")
        
        
        # Use ThreadPoolExecutor to process batch concurrently
        with ThreadPoolExecutor(max_workers=batch_size) as executor:    
            # Submit all papers in the batch
            future_to_paper_id = {}
            for paper_id in batch:
                future = executor.submit(process_single_paper, paper_id)
                future_to_paper_id[future] = paper_id
            
            # Process completed futures
            for future in as_completed(future_to_paper_id):
                paper_id = future_to_paper_id[future]
                try:
                    result = future.result()
                    if result:
                        print(f"Result: {result}")
                        # Save result to JSON file
                        output_path = f"data/parsed/{paper_id}.json"
                        parsed_data = load_json(output_path)
                        parsed_data.update(result)
                        save_json(parsed_data, output_path)
                        
                        # Update database status
                        cursor.execute("UPDATE arxiv_papers SET status = 'processed' WHERE doc_id = ?", (paper_id,))
                        conn.commit()
                        print(f"Successfully processed paper {paper_id}")
                    else:
                        print(f"No result returned for paper {paper_id}")
                except Exception as e:
                    print(f"Error processing paper {paper_id}: {str(e)}")
    
    cursor.close()
    conn.close()
    print("Finished processing all papers")

def process_single_paper(paper_id, db_path="data/db/arxiv.db"):
    """Process a single paper"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        result = process_paper(paper_id, cursor)
        return result
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    process_papers_batch()
