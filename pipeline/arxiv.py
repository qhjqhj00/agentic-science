from pipeline.agent import *
from pipeline.prompts import *
from pipeline.schema import *
from src.utils import *
from src.tools import *
import json

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

def paper_attention_score(title, authors):
    prompt = paper_attention_score_prompt(title, authors)
    agent = get_agent(api_dict["local_agent"]["base_url"], api_dict["local_agent"]["api_key"])
    response = stream_completion(agent, api_dict["local_agent"]["model"], prompt, schema=PaperAttentionScoreSchema, stream=False)
    return response

def process_paper(paper_id, cursor):
    """Process a paper, catching and logging any errors."""
    cursor.execute("SELECT * FROM arxiv_papers WHERE doc_id = ?", (paper_id,))
    data = cursor.fetchone()
    if data is None:
        print(f"Paper {paper_id} not found in database")
        return None
    processed_info = {}
    try:
        title, authors, markdown_path, parsed_path = data[1], data[2], data[5], data[6]
        if parsed_path == "":
            return {}
        parsed_content = load_json(parsed_path)

        authors_info = json.loads(parse_author(parsed_content, authors, markdown_path))
        processed_info["authors_info"] = authors_info
        attention_score = json.loads(paper_attention_score(title, authors_info))
        processed_info["attention_score"] = attention_score

        mapped_sections = json.loads(map_sections(parsed_content))
        content = ""
        for section in mapped_sections["parsed_sections"]:
            if section not in parsed_content:
                continue
            content += f"{section}: {parsed_content[section]['content']}\n\n"
        paper_info = json.loads(parse_paper(title, content))
        processed_info["paper_info"] = paper_info
        processed_info["abs_url"] = f"https://arxiv.org/abs/{paper_id}"
        return processed_info
    except Exception as e:
        print(f"Error processing paper {paper_id}: {str(e)}")
        return {}

if __name__ == "__main__":
    import sqlite3
    paper_id = "2510.12133"
    conn = sqlite3.connect("data/db/arxiv.db")
    cursor = conn.cursor()
    print(process_paper(paper_id, cursor))
    cursor.close()
    conn.close()