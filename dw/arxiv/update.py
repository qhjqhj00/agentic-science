import asyncio
import os
import json
from dw.arxiv.crawler import get_list_page, get_full_content
from dw.arxiv.process import process_papers_batch
from config.list_urls import get_list_urls_skip
from src.utils import save_jsonl, load_jsonl

async def get_recent_arxiv_papers(category: str = "cs.CL", max_papers: int = 100, step: int = 25, jsonl_file: str = "data/arxiv_list_content.jsonl"):
    """
    Get recent arxiv papers by fetching in batches and appending to jsonl file.
    
    Args:
        category: ArXiv category (default: "cs.CL")
        max_papers: Maximum number of papers to fetch
        step: Step size for each batch (default: 25)
    """
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    all_papers = []
    
    for skip in range(0, max_papers, step):
        print(f"Fetching papers {skip} to {skip + step}...")
        
        # Get list page URL
        url = get_list_urls_skip(category, step, skip)
        
        # Get paper list
        list_content = await get_list_page(url)
        
        if not list_content:
            print(f"No more papers found at skip={skip}")
            break
        
        newly_added_ids = await get_full_content(list_content)
        all_papers.extend(newly_added_ids)
        # Sleep for 60 seconds to avoid rate limiting
    
    print(f"Total papers processed: {len(all_papers)}")
    return all_papers

if __name__ == "__main__":
    # Set maximum number of papers to fetch
    MAX_PAPERS = 400
    # Run the script
    asyncio.run(get_recent_arxiv_papers(max_papers=MAX_PAPERS, step=25))
    process_papers_batch()
