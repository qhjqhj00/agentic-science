from typing import Dict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from pipeline.agent import *
from pipeline.state import *
from pipeline.prompts import *
from pipeline.schema import *
from src.utils import *
from src.tools import *
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import re
from collections import Counter
import json
import sqlite3
import sys
import glob
from datetime import datetime, timedelta

api_dict = load_json("config/api_dict.json")

def filter_footnote_references(text):
    """Remove lines starting with [^number]: pattern from text"""
    import re
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        # Check if line starts with [^number]: pattern
        if not re.match(r'^\[\^\d+\]:', line.strip()):
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def generate_research_report(date: str, output_file: str = None, db_path: str = "data/db/arxiv.db", language: str = "zh") -> None:
    """
    Generate a comprehensive research report by classifying papers into topics and creating summaries.
    
    Args:
        date: Date in ISO format (e.g., "2024-10-15T00:00:00")
        output_file: Path to save the markdown report (if None, will be auto-generated)
        db_path: Path to the database file
        language: Language for the summaries ("en" for English, "zh" for Chinese)
    """
    print("Starting research report generation...")
    
    # Connect to database and get records
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    data_list = get_records_by_date(date, cursor)
    
    if not data_list:
        print(f"No processed papers found for date: {date}")
        conn.close()
        return
    
    print(f"Found {len(data_list)} papers for date: {date}")
    
    # Extract keywords and classify into topics
    print("Extracting keywords and classifying into topics...")
    keyword_freq = get_all_key_words(data_list)
    topics = json.loads(extract_topics_from_keywords(list(keyword_freq.keys())))
    print(f"Extracted {len(topics.get('topics', []))} topics")
    
    # Classify papers by topic
    topic_classification = classify_papers_by_topic_keywords(topics, data_list)
    
    output_dir = "data/reports"
    os.makedirs(output_dir, exist_ok=True)
    # Generate output filename if not provided
    if output_file is None:
        date_str = date.split('T')[0].replace('-', '')
        output_file = f"{output_dir}/{date_str}+NLP领域论文汇总（{language}）.md"
    
    # Prepare markdown content
    markdown_content = []
    
    # Add title
    if language == "zh":
        current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
        report_date = date.split('T')[0]
        year, month, day = report_date.split('-')
        formatted_date = f"{year}年{month}月{day}日"
        markdown_content.append("+++\n")
        markdown_content.append(f"title = '{formatted_date}NLP领域论文汇总（中文）'\n")
        markdown_content.append(f"date = {current_date}\n")
        markdown_content.append("draft = false\n")
        markdown_content.append("+++\n\n")
    else:
        current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
        report_date = date.split('T')[0]
        year, month, day = report_date.split('-')
        formatted_date = f"{year}年{month}月{day}日"
        markdown_content.append("+++\n")
        markdown_content.append(f"title = '{formatted_date}NLP领域论文汇总（英文）'\n")
        markdown_content.append(f"date = {current_date}\n")
        markdown_content.append("draft = false\n")
        markdown_content.append("+++\n\n")
    
    # Generate table of contents with paper counts
    for i, topic in enumerate(topic_classification, 1):
        topic_name = topic["topic"]
        paper_count = len(topic["paper_ids"])
        if paper_count > 0:  # Only include topics with papers
            markdown_content.append(f"- [Topic {i}: {topic_name}](#topic-{i}-{topic_name.lower().replace(' ', '-')}) ({paper_count} papers)\n")
    markdown_content.append("<!--more-->\n\n")
    markdown_content.append("\n---\n\n")
    
    # Generate content for each topic
    topic_counter = 1
    for topic in topic_classification:
        topic_name = topic["topic"]
        paper_ids = topic["paper_ids"]
        
        if not paper_ids:  # Skip empty topics
            continue
            
        topic_papers = get_papers_by_ids(data_list, paper_ids)
        
        print(f"Generating summary for Topic {topic_counter}: {topic_name} ({len(topic_papers)} papers)")
        
        # Add topic header
        markdown_content.append(f"## Topic {topic_counter}: {topic_name}\n\n")
        
        # Generate topic summary
        summary = generate_topic_summary(topic_name, topic_papers, language)
        
        if summary:
            summary = filter_footnote_references(summary)
            markdown_content.append(summary)
            markdown_content.append("\n\n---\n\n")
        else:
            if language == "zh":
                markdown_content.append("生成摘要时出现错误。\n\n---\n\n")
            else:
                markdown_content.append("Error generating summary for this topic.\n\n---\n\n")
        
        topic_counter += 1
    
    # Generate references section
    if language == "zh":
        markdown_content.append("## 参考文献\n\n")
    else:
        markdown_content.append("## References\n\n")
    
    # Generate reference entries
    for paper in data_list:
        paper_idx = paper["_idx"]
        title = paper.get("title", "Unknown Title")
        # Construct abs_url from doc_id if not available
        doc_id = paper.get("doc_id", "")
        abs_url = f"https://arxiv.org/abs/{doc_id}" if doc_id else "#"
        markdown_content.append(f"[^{paper_idx}]: [{title}]({abs_url})\n\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(markdown_content))
        print(f"Research report saved to: {output_file}")
    except Exception as e:
        print(f"Error saving report to {output_file}: {str(e)}")
    finally:
        conn.close()

def get_records_by_date(date, cursor):
    """
    Get records by date.
    
    Args:
        date: Date in ISO format (e.g., "2024-10-15T00:00:00")
        cursor: Database cursor object
    """
    cursor.execute("SELECT * FROM arxiv_papers WHERE released_date = ? AND status = 'processed'", (date,))
    records = cursor.fetchall()
    data_list = []
    for i,record in enumerate(records):
        data_list.append({
            "_idx": i,
            "doc_id": record[0],
            "title": record[1],
            "author": record[2],
            "parsed_info": load_json(record[6]),
        })
    return data_list

def get_all_key_words(data_list):
    """
    Get all keywords from records.
    """
    keywords = []
    for data in data_list:
        keywords.extend(data["parsed_info"]["paper_info"]["keywords"]["keywords"])
    keyword_freq = Counter(keywords)
    return keyword_freq

def classify_papers_by_topic_keywords(topic_dict, data_list):
    """
    Classify papers by topic based on keyword matching.
    
    Args:
        topic_dict: Dictionary with topics and their keywords
        data_list: List of paper data with parsed_info containing keywords
    
    Returns:
        List of dictionaries with topic and paper_ids (all paper IDs for each topic)
    """
    # Create topic keyword mapping
    topic_keywords = {}
    for topic_info in topic_dict.get("topics", []):
        topic_name = topic_info["topic"]
        keywords = [kw.lower() for kw in topic_info["keywords"]]
        topic_keywords[topic_name] = keywords
    
    # Initialize result structure
    topic_classification = {}
    for topic_name in topic_keywords.keys():
        topic_classification[topic_name] = []
    topic_classification["misc"] = []
    
    # Classify each paper
    for data in data_list:
        paper_id = data["doc_id"]
        paper_keywords = data["parsed_info"]["paper_info"]["keywords"]["keywords"]
        paper_keywords_lower = [kw.lower() for kw in paper_keywords]
        
        best_topic = "misc"
        best_ratio = 0.0
        
        # Calculate matching ratio for each topic
        for topic_name, topic_kws in topic_keywords.items():
            if not topic_kws:  # Skip empty keyword lists
                continue
                
            # Count matches
            matches = sum(1 for kw in paper_keywords_lower if kw in topic_kws)
            ratio = matches / len(topic_kws) if len(topic_kws) > 0 else 0.0
            
            # Update best match
            if ratio > best_ratio:
                best_ratio = ratio
                best_topic = topic_name
        
        # Assign paper to best matching topic
        topic_classification[best_topic].append(paper_id)
    
    # Convert to output format - include all topics even if empty
    result = []
    for topic_name in topic_keywords.keys():
        result.append({
            "topic": topic_name,
            "paper_ids": topic_classification[topic_name]
        })
    
    # Add misc category if it has papers
    if topic_classification["misc"]:
        result.append({
            "topic": "misc",
            "paper_ids": topic_classification["misc"]
        })
    
    return result

def generate_topic_summary(topic_name, topic_papers, language: str = "en"):
    """
    Generate a summary for a topic.
    
    Args:
        topic_name: Name of the topic
        topic_papers: List of papers in the topic
        language: Language for the summary
    """
    prompt = topic_summary_prompt(topic_name, topic_papers, language)
    agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
    response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, stream=True)
    return response

def extract_topics_from_keywords(keywords_list):
    """
    Extract topics from a list of keywords.
    
    Args:
        keywords_list: List of keywords to be categorized into topics
    
    Returns:
        str: Formatted prompt for topic extraction
    """
    prompt = keyword_topic_extraction_prompt(keywords_list)
    agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
    response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, schema=TopicSchema, stream=False)
    return response

def get_papers_by_ids(data_list, ids):
    """
    Get papers from data_list by their IDs.
    
    Args:
        data_list: List of paper dictionaries
        ids: List of paper IDs to retrieve
    
    Returns:
        List of paper dictionaries corresponding to the given IDs
    """
    # Create a mapping from doc_id to paper data for efficient lookup
    id_to_paper = {paper.get("doc_id"): paper for paper in data_list}
    
    # Return papers matching the requested IDs
    filtered_papers = [id_to_paper[paper_id] for paper_id in ids]

    return filtered_papers


if __name__ == "__main__":
    
    # Check if refresh mode is enabled
    refresh_mode = len(sys.argv) > 1 and sys.argv[1] == "--refresh"
    
    if refresh_mode:
        print("Running in refresh mode - regenerating all reports from 2025-10-02 onwards")
        # Generate all dates from 2025-10-02 to today
        start_date = datetime(2025, 10, 2)
        today = datetime.now()
        current_date = start_date
        
        dates_to_generate = []
        while current_date <= today:
            dates_to_generate.append(current_date.strftime("%Y-%m-%dT00:00:00"))
            current_date += timedelta(days=1)
    else:
        print("Running in incremental mode - generating missing reports")
        # Get today's date
        today = datetime.now()
        
        # Parse existing report files to find the latest date
        reports_dir = "data/reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir, exist_ok=True)
        
        # Find all existing report files
        report_files = glob.glob(os.path.join(reports_dir, "*.md"))
        existing_dates = set()
        
        for file_path in report_files:
            filename = os.path.basename(file_path)
            # Extract date from filename like "20251014+NLP领域论文汇总（zh）.md"
            if filename.startswith("2025") and len(filename) >= 8:
                date_str = filename[:8]  # Extract YYYYMMDD
                try:
                    # Convert to datetime object
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    existing_dates.add(date_obj.date())
                except ValueError:
                    continue
        
        # Find the latest existing date
        if existing_dates:
            latest_existing_date = max(existing_dates)
            print(f"Latest existing report date: {latest_existing_date}")
            # Start from the day after the latest existing date
            start_date = datetime.combine(latest_existing_date, datetime.min.time()) + timedelta(days=1)
        else:
            print("No existing reports found, starting from 2025-10-02")
            start_date = datetime(2025, 10, 2)
        
        # Generate dates from start_date to today
        current_date = start_date
        dates_to_generate = []
        
        while current_date.date() <= today.date():
            dates_to_generate.append(current_date.strftime("%Y-%m-%dT00:00:00"))
            current_date += timedelta(days=1)
    
    print(f"Dates to generate reports for: {len(dates_to_generate)} days")
    for date in dates_to_generate:
        print(f"  - {date}")
    
    # Generate reports for all identified dates
    for date in dates_to_generate:
        print(f"\nGenerating reports for {date}")
        try:
            # Generate Chinese report
            print(f"  Generating Chinese report...")
            generate_research_report(date, language="zh")
            
            # Generate English report
            print(f"  Generating English report...")
            generate_research_report(date, language="en")
            
            print(f"  ✓ Completed reports for {date}")
        except Exception as e:
            print(f"  ✗ Error generating reports for {date}: {str(e)}")
            continue
    
    print(f"\nReport generation completed!")

