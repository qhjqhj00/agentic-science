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
from datetime import datetime
import re
api_dict = load_json("config/api_dict.json")

def parse_author(parsed_content, authors, markdown_path):
    if "author" not in parsed_content or parsed_content["author"]["content"] is None:
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()
        author_in_paper = " ".join(content.split()[:1024])
    else:
        author_in_paper = parsed_content["author"]["content"]
    prompt = author_prompt(authors, author_in_paper)
    agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
    response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, schema=AuthorsSchema, stream=False)
    return response

def parse_paper(title, content):
    prompt = PAPER_ANALYSIS_PROMPT.format(title=title, content=content)
    agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
    response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, schema=PaperAnalysisSchema, stream=False)
    return response

def map_sections(parsed_content, target_sections=None):
    parsed_keys = list(parsed_content.keys())
    prompt = section_mapping_prompt(parsed_keys, target_sections)
    agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
    response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, schema=SectionMappingSchema, stream=False)
    return response

def process_paper(line):
    """Process a paper, catching and logging any errors."""
    try:
        paper_id = line["id"]
        title = line["title"]
        authors = line["authors"]
        markdown_path = line["markdown_path"]

        parsed_content = line["parsed_content"]

        authors_info = parse_author(parsed_content, authors, markdown_path)
        line["authors_info"] = authors_info

        mapped_sections = json.loads(map_sections(parsed_content))
        content = ""
        for section in mapped_sections["parsed_sections"]:
            content += f"{section}: {parsed_content[section]['content']}\n\n"
        paper_info = parse_paper(title, content)
        line["paper_info"] = paper_info
        line["abs_url"] = f"https://arxiv.org/abs/{paper_id}"
        return line
    except Exception as e:
        print(f"Error processing paper {line.get('id', 'unknown')}: {str(e)}")
        return None

def process_jsonl_file(input_file: str, output_file: str = "data/processed.jsonl", batch_size: int = 4):
    """
    Process a JSONL file by applying process_paper to each line in parallel.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        batch_size: Number of papers to process in parallel
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load input data
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} papers from {input_file}")
    
    processed_results = []
    
    # Process in batches with progress bar
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Submit all tasks
        future_to_line = {executor.submit(process_paper, line): line for line in data}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_line), total=len(data), desc="Processing papers"):
            result = future.result()
            if result is not None:
                processed_results.append(result)
    
    # Save results
    save_jsonl(processed_results, output_file)
    print(f"Successfully processed {len(processed_results)} out of {len(data)} papers")
    print(f"Results saved to {output_file}")
    
    return processed_results

def classify_papers_by_topic(processed_results: List[Dict]) -> Dict:
    """
    Classify processed papers into topics based on their titles and scenarios.
    
    Args:
        processed_results: List of processed paper dictionaries containing paper_info
        
    Returns:
        Dictionary containing topic classification results
    """
    # Extract title and scenario from processed results
    papers_for_classification = []
    for i, paper in enumerate(processed_results):
        try:
            title = paper.get("title", "")
            scenario = ""
            if "paper_info" in paper and paper["paper_info"] is not None:
                if isinstance(paper["paper_info"], str):
                    paper["paper_info"] = json.loads(paper["paper_info"])
                scenario_and_task = paper["paper_info"].get("scenario_and_task", {})
                if scenario_and_task:
                    scenario = scenario_and_task.get("scenario", "")
                    task = scenario_and_task.get("task", "")
            
            papers_for_classification.append({
                "id": i,
                "title": title,
                "scenario": scenario,
                "task": task
            })
        except Exception as e:
            print(f"Error extracting info from paper {i}: {str(e)}")
            continue
    
    if not papers_for_classification:
        print("No papers available for classification")
        return {"results": []}
    
    # Generate classification prompt
    prompt = paper_topic_classification_prompt(papers_for_classification)
    
    # Get agent and classify
    agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
    response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, schema=PaperTopicClassificationSchema, stream=False)
    
    return response


def generate_topic_summary(topic: str, papers: List[Dict], language: str = "en") -> str:
    """
    Generate a summary for a specific topic based on the papers assigned to it.
    
    Args:
        topic: The name of the topic/research area
        papers: List of paper dictionaries containing paper information
        language: Language for the summary ("en" for English, "zh" for Chinese, etc.)
        
    Returns:
        String containing the generated topic summary
    """
    if not papers:
        print(f"No papers provided for topic: {topic}")
        return ""
    
    try:
        # Generate summary prompt
        prompt = topic_summary_prompt(topic, papers, language)
        print(f"Generating summary for topic: {topic}")
        
        # Get agent and generate summary
        agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
        response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, stream=True)
        
        return response
        
    except Exception as e:
        print(f"Error generating summary for topic {topic}: {str(e)}")
        return ""


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

def generate_research_report(processed_papers: List[Dict], output_file: str = "data/test.md", language: str = "zh") -> None:
    """
    Generate a comprehensive research report by classifying papers into topics and creating summaries.
    
    Args:
        processed_papers: List of processed paper dictionaries
        output_file: Path to save the markdown report
        language: Language for the summaries ("en" for English, "zh" for Chinese)
    """
    print("Starting research report generation...")
    
    # Add index to papers
    for i, paper in enumerate(processed_papers):
        paper["_idx"] = i
    
    # Classify papers by topic
    print("Classifying papers by topic...")
    topic_classification_result = json.loads(classify_papers_by_topic(processed_papers))
    
    if not topic_classification_result.get("results"):
        print("No classification results found")
        return
    
    # Prepare markdown content
    markdown_content = []
    
    # Add title
    if language == "zh":
        current_date = datetime.now().strftime("%Y年%m月%d日")
        markdown_content.append(f"# 研究论文主题分析报告 - {current_date}\n")
        markdown_content.append("## 目录\n")
    else:
        current_date = datetime.now().strftime("%Y-%m-%d")
        markdown_content.append(f"# Research Paper Topic Analysis Report - {current_date}\n")
        markdown_content.append("## Table of Contents\n")
    
    # Generate table of contents with paper counts
    for i, topic in enumerate(topic_classification_result["results"], 1):
        topic_name = topic["topic"]
        paper_count = len(topic["ids"])
        markdown_content.append(f"- [Topic {i}: {topic_name}](#topic-{i}-{topic_name.lower().replace(' ', '-')}) ({paper_count} papers)\n")
    
    markdown_content.append("\n---\n\n")
    
    # Generate content for each topic
    for i, topic in enumerate(topic_classification_result["results"], 1):
        topic_name = topic["topic"]
        topic_ids = topic["ids"]
        topic_papers = [processed_papers[idx] for idx in topic_ids]
        
        print(f"Generating summary for Topic {i}: {topic_name} ({len(topic_papers)} papers)")
        
        # Add topic header
        markdown_content.append(f"## Topic {i}: {topic_name}\n\n")
        
        # Generate topic summary
        summary = generate_topic_summary(topic_name, topic_papers, language)
        # Remove lines starting with [^number]: pattern from summary

        
        if summary:
            summary = filter_footnote_references(summary)
        if summary:
            markdown_content.append(summary)
            markdown_content.append("\n\n---\n\n")
        else:
            if language == "zh":
                markdown_content.append("生成摘要时出现错误。\n\n---\n\n")
            else:
                markdown_content.append("Error generating summary for this topic.\n\n---\n\n")
    
    # Write to file
    # Generate references section
    if language == "zh":
        markdown_content.append("## 参考文献\n\n")
    else:
        markdown_content.append("## References\n\n")
    
    # Collect all paper IDs that were referenced in the summaries
    
    # Generate reference entries
    for i, paper in enumerate(processed_papers):
        title = paper.get("title", "Unknown Title")
        abs_url = paper.get("abs_url", "#")
        markdown_content.append(f"[^{i}]: [{title}]({abs_url})\n\n")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(markdown_content))
        print(f"Research report saved to: {output_file}")
    except Exception as e:
        print(f"Error saving report to {output_file}: {str(e)}")


if __name__ == "__main__":
    tmp_data = load_jsonl("data/processed.jsonl")
    # print(tmp_data[0])
    language = "en"
    generate_research_report(tmp_data, output_file=f"examples/{language}.md", language=language)
