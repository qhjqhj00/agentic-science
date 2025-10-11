import asyncio
import os
import json
import schedule
import time
from datetime import datetime
from dw.update_arxiv import get_recent_arxiv_papers
from dw.parse_md import parse_markdown_sections
from src.utils import load_jsonl, save_jsonl
import requests

def get_shanghai_time():
    """Get current time in Shanghai timezone"""
    try:
        response = requests.get("https://worldtimeapi.org/api/timezone/Asia/Shanghai")
        if response.status_code == 200:
            data = response.json()
            return datetime.fromisoformat(data['datetime'].replace('Z', '+00:00'))
    except Exception as e:
        print(f"Error getting Shanghai time: {e}")
    return datetime.now()

async def update_papers():
    """Update papers using get_recent_arxiv_papers"""
    print(f"Starting paper update at {get_shanghai_time()}")
    try:
        await get_recent_arxiv_papers(max_papers=500)
        print("Paper update completed successfully")
    except Exception as e:
        print(f"Error updating papers: {e}")

def process_papers_by_date():
    """Process papers from jsonl file and organize by date"""
    print("Processing papers by date...")
    
    # Create by_date directory if it doesn't exist
    os.makedirs("data/by_date", exist_ok=True)
    
    # Load papers from jsonl file
    try:
        papers = load_jsonl("data/arxiv_list_content.jsonl")
    except FileNotFoundError:
        print("No arxiv_list_content.jsonl file found")
        return
    except Exception as e:
        print(f"Error loading jsonl file: {e}")
        return
    
    # Group papers by date
    papers_by_date = {}
    
    for paper in papers:
        try:
            # Get arxiv_id and extract date (first 7 characters)
            arxiv_id = paper.get('id', '')
            if len(arxiv_id) >= 7:
                date_str = arxiv_id[:7]  # e.g., "2510.08" -> "2510.08"
                
                # Parse markdown if markdown_path exists
                markdown_path = paper.get('markdown_path')
                if markdown_path and os.path.exists(markdown_path):
                    try:
                        with open(markdown_path, 'r', encoding='utf-8') as f:
                            md_text = f.read()
                        parsed_content = parse_markdown_sections(md_text)
                        paper['parsed_content'] = parsed_content
                    except Exception as e:
                        print(f"Error parsing markdown for {arxiv_id}: {e}")
                        paper['parsed_content'] = None
                else:
                    paper['parsed_content'] = None
                
                # Group by date
                if date_str not in papers_by_date:
                    papers_by_date[date_str] = []
                papers_by_date[date_str].append(paper)
            
        except Exception as e:
            print(f"Error processing paper {paper.get('id', 'unknown')}: {e}")
    
    # Save papers by date
    for date_str, date_papers in papers_by_date.items():
        try:
            output_file = f"data/by_date/{date_str}.jsonl"
            save_jsonl(date_papers, output_file)
            print(f"Saved {len(date_papers)} papers to {output_file}")
        except Exception as e:
            print(f"Error saving papers for date {date_str}: {e}")

async def daily_update():
    """Perform daily update: fetch papers and process by date"""
    print(f"Starting daily update at {get_shanghai_time()}")
    
    # Update papers
    await update_papers()
    
    # Process papers by date
    process_papers_by_date()
    
    print(f"Daily update completed at {get_shanghai_time()}")

def run_scheduler():
    """Run the scheduler"""
    # Schedule daily update at 10:00 AM Shanghai time
    schedule.every().day.at("10:00").do(lambda: asyncio.run(daily_update()))
    
    print("Scheduler started. Daily updates scheduled for 10:00 AM Shanghai time.")
    print("Press Ctrl+C to stop the scheduler.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nScheduler stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "run-once":
        # Run once for testing
        # Usage: python -m dw.update_by_date run-once
        asyncio.run(daily_update())
    else:
        # Run scheduler
        run_scheduler()
