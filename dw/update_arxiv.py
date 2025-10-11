import asyncio
import os
import json
from dw.arxiv_tools import get_list_page, get_abstracts, get_full_content, get_existing_arxiv_ids
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
        existing_arxiv_ids, dates = get_existing_arxiv_ids()
        print(f"Fetching papers {skip} to {skip + step}...")
        
        # Get list page URL
        url = get_list_urls_skip(category, step, skip)
        
        # Get paper list
        list_content = await get_list_page(url)
        
        if not list_content:
            print(f"No more papers found at skip={skip}")
            break
        
        # Check if any paper in list_content has the same date as existing papers
        should_break = False
        for paper in list_content:
            arxiv_id = paper.get('id', '')
            if len(arxiv_id) >= 7 and '.' in arxiv_id:
                date_part = arxiv_id[:7]  # e.g., "2510.02"
                if date_part in dates:
                    print(f"Found paper with existing date {date_part}, stopping fetch")
                    should_break = True
                    break
        
        if should_break:
            break
        
        # Get abstracts
        # parsed_content = await get_abstracts(list_content, existing_arxiv_ids)
        
        # Get full content (HTML and markdown)
        parsed_content = await get_full_content(list_content, existing_arxiv_ids)
        
        # Filter out papers that were skipped due to existing IDs
        new_papers = [paper for paper in parsed_content if paper['id'] not in existing_arxiv_ids]
        
        if new_papers:
            # Append to jsonl file
            if os.path.exists(jsonl_file):
                # Append to existing file
                with open(jsonl_file, 'a', encoding='utf-8') as f:
                    for paper in new_papers:
                        f.write(json.dumps(paper, ensure_ascii=False) + "\n")
            else:
                # Create new file
                save_jsonl(new_papers, jsonl_file)
            
            all_papers.extend(new_papers)
            existing_arxiv_ids.extend([paper['id'] for paper in new_papers])
            
            print(f"Added {len(new_papers)} new papers")
        
        else:
            print("No new papers in this batch")
        # Sleep for 60 seconds to avoid rate limiting
    
    print(f"Total papers processed: {len(all_papers)}")
    return all_papers

if __name__ == "__main__":
    # Set maximum number of papers to fetch
    MAX_PAPERS = 700
    # Run the script
    asyncio.run(get_recent_arxiv_papers(max_papers=MAX_PAPERS))
