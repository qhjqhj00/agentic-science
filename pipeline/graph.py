from typing import Dict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from pipeline.agent import *
from pipeline.state import *
from pipeline.prompts import *
from pipeline.schema import *
from src.utils import *
from src.tools import *

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

def process_paper(line):
    paper_id = line["id"]
    title = line["title"]
    authors = line["authors"]
    markdown_path = line["markdown_path"]

    parsed_content = line["parsed_content"]

    authors_info = parse_author(parsed_content, authors, markdown_path)
    print(json.loads(authors_info))
    
if __name__ == "__main__":
    test_data = load_jsonl("data/by_date/2510.08.jsonl")
    process_paper(test_data[1])
