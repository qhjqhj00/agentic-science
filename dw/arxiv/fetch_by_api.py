import json
import requests
from granary import jsonfeed, rss
from datetime import datetime, timedelta
from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

import os

def parse_authors_from_xml(xml_content):
    """Parse all authors from arxiv XML response"""
    try:
        root = ET.fromstring(xml_content)
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        
        result = []
        for entry in entries:
            authors = entry.findall('{http://www.w3.org/2005/Atom}author')
            author_list = []
            for author in authors:
                name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                if name_elem is not None and name_elem.text:
                    author_list.append({'name': name_elem.text.strip()})
            
            # Get entry ID to match with granary output
            entry_id = entry.find('{http://www.w3.org/2005/Atom}id')
            entry_id_text = entry_id.text if entry_id is not None else None
            
            if author_list and entry_id_text:
                result.append({
                    'id': entry_id_text,
                    'authors': author_list
                })
        
        return result
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return []

days = 365
day_timestamps = [(datetime.now() - timedelta(days=i)).strftime('%Y%m%d') for i in range(1, days)]

fields = ["cs.AI"]
BASE_URL = 'http://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+submittedDate:[{day}0000+TO+{day}2359]&max_results=1000'

feeds = [BASE_URL.format(day=date) for date in day_timestamps]
# feeds = ["https://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+submittedDate:%5B202510170000+TO+202510172359%5D&max_results=2"]
# Create output directory if it doesn't exist
os.makedirs("data/api_by_date", exist_ok=True)

for i, feed in enumerate(tqdm(feeds)):
    date = day_timestamps[i]
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            resp = requests.get(
                feed, headers={"User-Agent": "arxiv-poll"}
            )
            resp.raise_for_status()
            break
        except requests.RequestException:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Failed to fetch {feed} after {max_retries} retries")
                break
            print(f"Retry {retry_count}/{max_retries} for {feed}")
            continue
    
    if retry_count >= max_retries:
        continue
    

    # Parse authors from XML
    xml_authors = parse_authors_from_xml(resp.text)
    print(f"Parsed {len(xml_authors)} entries with authors from XML")

    activities = jsonfeed.activities_to_jsonfeed(rss.to_activities(resp.text))
    # Check if items exist
    if "items" not in activities or not activities["items"]:
        print(f"No items found for {date}")
        continue
    
    print(f"Fetched {date} with {len(activities['items'])} items")

    # Create a mapping of entry ID to authors for quick lookup
    author_map = {entry['id']: entry['authors'] for entry in xml_authors}

    activities["items"] = [
        {
            **activity,
            "content_html": BeautifulSoup(activity["content_html"], "html.parser").get_text(),
            "authors": author_map.get(activity["id"], activity.get("authors", []))
        }
        for activity in activities["items"]
    ]
    # Save to date-specific file
    output_file = f"data/api_by_date/{date}.json"
    with open(output_file, "w") as f:
        json.dump(activities, f, indent=2)