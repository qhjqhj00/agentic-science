def author_prompt(authors, author_in_paper):
    return f"""You are tasked with extracting author information from a research paper. You need to match the author names with their affiliations and other information.

Input:
- authors: {authors}
- author_in_paper: {author_in_paper}

Instructions:
1. Match each author name from the authors list with information in author_in_paper
2. Look for numerical superscripts after author names (like 1, 2, etc.) that indicate affiliations
3. Extract the corresponding affiliations/organizations based on these numbers
4. If no author information is found in author_in_paper, only output the author name
5. Include any additional miscellaneous information (like email addresses) in the misc field

Output format (JSON):
[
  {{
    "name": "Author Name",
    "org": ["Organization 1", "Organization 2"],
    "misc": ["email@example.com", "other info"]
  }},
  ...
]

Example:
Input authors: Watcharapong Timklaypachara, Monrada Chiewhawan, Nopporn Lekuthai, Titipat Achakulvisut
Input author_in_paper: Watcharapong Timklaypachara1,2 Monrada Chiewhawan1,2 Nopporn Lekuthai1,2  \\nTitipat Achakulvisut1  \\n1Department of Biomedical Engineering, Faculty of Engineering, Mahidol University  \\n2Faculty of Medicine Ramathibodi Hospital, Mahidol University, Bangkok, Thailand

Output:
[
  {{
    "name": "Watcharapong Timklaypachara",
    "org": ["Department of Biomedical Engineering, Faculty of Engineering, Mahidol University", "Faculty of Medicine Ramathibodi Hospital, Mahidol University, Bangkok, Thailand"],
    "misc": []
  }},
  {{
    "name": "Monrada Chiewhawan",
    "org": ["Department of Biomedical Engineering, Faculty of Engineering, Mahidol University", "Faculty of Medicine Ramathibodi Hospital, Mahidol University, Bangkok, Thailand"],
    "misc": []
  }},
  {{
    "name": "Nopporn Lekuthai",
    "org": ["Department of Biomedical Engineering, Faculty of Engineering, Mahidol University", "Faculty of Medicine Ramathibodi Hospital, Mahidol University, Bangkok, Thailand"],
    "misc": []
  }},
  {{
    "name": "Titipat Achakulvisut",
    "org": ["Department of Biomedical Engineering, Faculty of Engineering, Mahidol University"],
    "misc": []
  }}
]

Please extract the author information and return only the JSON array:"""


PAPER_ANALYSIS_PROMPT = """You are an expert researcher analyzing academic papers. Given the following sections of a research paper, please analyze and extract key information.

Paper sections:
- Title: {title}
- content: {content}

Please analyze the paper and answer the following questions:

1. What scenario and task does this paper focus on?
2. What problems does this paper solve and what is the value of these problems?
3. What methods does this paper propose to solve these problems?
4. What are the main innovations of the proposed methods?
5. What datasets and models were used in the experiments, and what baselines were compared?

Output format (JSON):
{{
  "scenario_and_task": {{
    "scenario": "Description of the application scenario or domain",
    "task": "Specific task or problem being addressed"
  }},
  "problems_and_value": {{
    "problems": ["Problem 1", "Problem 2", "..."],
    "value": "Why these problems are important and valuable to solve"
  }},
  "proposed_methods": {{
    "main_method": "Primary method or approach proposed",
    "key_components": ["Component 1", "Component 2", "..."],
    "technical_details": "Brief description of how the method works"
  }},
  "innovations": {{
    "main_innovations": ["Innovation 1", "Innovation 2", "..."],
    "novelty_description": "What makes this work novel compared to existing approaches"
  }},
  "experimental_setup": {{
    "datasets": ["Dataset 1", "Dataset 2", "..."],
    "models": ["Base model 1", "Base model 2", "..."],
    "baselines": ["Baseline 1", "Baseline 2", "..."],
    "evaluation_metrics": ["Metric 1", "Metric 2", "..."]
  }}
}}

Instructions:
1. Extract information directly from the provided sections
2. If information for any field is not found in the provided content, set the value to null
3. Be concise but comprehensive in your descriptions
4. Focus on factual information rather than subjective interpretations
5. Ensure all arrays contain relevant items found in the text

Please analyze the paper and return only the JSON structure:"""


def section_mapping_prompt(parsed_sections: list, target_sections: list = None):
    """
    Generate a prompt to map parsed section names to target section names.
    
    Args:
        parsed_sections: List of section names from the parsed paper
        target_sections: List of target section names we need (default: ["abstract", "introduction", "method", "experiment"])
    
    Returns:
        String prompt for section mapping
    """
    if target_sections is None:
        target_sections = ["abstract", "introduction", "method", "experiment"]
    return f"""You are a research paper section mapper. Your task is to map parsed section names to target section names based on semantic similarity and content relevance.

Given parsed section names: {parsed_sections}
Target section names: {target_sections}

For each target section, find the most appropriate parsed section name that matches it. Consider these mapping rules:

- "abstract" matches: Abstract, Summary, Overview
- "introduction" matches: Introduction, Background, Motivation
- "method" matches: Method, Methodology, Approach, Proposed Method, Technical Approach, Algorithm
- "experiment" matches: Experiment, Experiments, Evaluation, Results, Empirical Study, Performance Analysis

Output format: A JSON object where keys are target section names and values are the best matching parsed section names (or null if no good match exists).

Example:
If parsed_sections = ["Abstract", "Background", "Methodology", "Evaluation"] and target_sections = ["abstract", "introduction", "method", "experiment"]
Output: {{"parsed_sections":["Abstract", "Background", "Methodology", "Evaluation"]}}

If parsed_sections = ["Summary", "Our Approach"] and target_sections = ["abstract", "introduction", "method", "experiment"]  
Output: {{"parsed_sections":["Summary", "Our Approach"]}}

Please return only the JSON array:"""


def paper_topic_classification_prompt(papers: list):
    """
    Generate a prompt to classify papers into topics based on their titles and scenarios.
    
    Args:
        papers: List of dictionaries containing 'id', 'title', and 'scenario' for each paper
    
    Returns:
        String prompt for paper topic classification
    """
    papers_text = ""
    for paper in papers:
        papers_text += f"id: {paper['id']}\n"
        papers_text += f"title: {paper['title']}\n"
        papers_text += f"scenario: {paper['scenario']}\n"
        papers_text += f"task: {paper['task']}\n\n"
    
    return f"""You are a research paper topic classifier. Your task is to analyze the given papers and classify them into at most 10 meaningful topics based on their titles and scenarios.

Papers to classify:
{papers_text}

Instructions:
1. Create at most 10 topics that best represent the main research areas covered by these papers
2. Each topic should have a clear, descriptive name (e.g., "KV compression", "Multi-modal learning", "Reasoning optimization")
3. One topic can be "misc" for papers that don't fit well into the main topics
4. Assign each paper ID to the most appropriate topic
5. Each paper should be assigned to exactly one topic
6. Topic names should be concise but descriptive of the research area
7. Each paper must be assigned to exactly one topic - no paper should appear in multiple topics
8. Topic names should be readable and use proper spacing between words (e.g., "Memory Efficiency and Compression" not "MemoryEfficiencyAndCompression")




Output format: A JSON object with a "results" array where each element contains a "topic" name and an "ids" array of paper IDs belonging to that topic.

Example output:
{{
  "results": [
    {{"topic": "KV compression", "ids": [0, 3, 7]}},
    {{"topic": "Multi-modal learning", "ids": [1, 4]}},
    {{"topic": "Reasoning optimization", "ids": [2, 5]}},
    {{"topic": "misc", "ids": [6]}}
  ]
}}

Please analyze the papers and return only the JSON structure:"""

def topic_summary_prompt(topic: str, papers: list, language: str = "en"):
    """
    Generate a prompt to create a comprehensive summary of papers within a specific topic.
    
    Args:
        topic: The name of the topic/research area
        papers: List of dictionaries containing paper information with 'id', 'title', 'authors', and 'paper' fields
        language: Output language ('en' for English, 'zh' for Chinese)
    
    Returns:
        String prompt for generating topic summary
    """
    papers_text = ""
    for paper in papers:
        papers_text += f"id: {paper['_idx']}\n"
        papers_text += f"title: {paper['title']}\n"
        papers_text += f"authors: {paper['authors_info']}\n"
        papers_text += f"paper information: {paper['paper_info']}\n\n"
    
    if language == "zh":
        return f"""你是一个研究论文分析专家。请为给定主题下的论文集合生成一个全面的总结报告。

主题: {topic}

论文信息:
{papers_text}

请按照以下要求生成总结:

1. **主题概述**: 简要介绍该研究主题的背景和重要性

2. **各论文贡献**: 分别对每篇论文进行详细分析，突出不同论文对该主题的独特贡献，格式如下：
   "来自[机构]的[第一作者]等人提出了[具体方法/模型]，主要贡献是[创新点和解决的问题]。在[数据集]上的实验表明，相比[基线方法]取得了[具体提升效果][^论文id]。"

3. **技术趋势**: 总结该主题下不同论文采用的主要技术路线和方法演进

4. **数据集和评估**: 汇总论文中使用的主要数据集和评估指标

要求:
- 使用中文输出
- 重点突出每篇论文的独特贡献和创新点
- 作者信息简化为"来自[机构]的[第一作者]等人"的格式
- 在描述每篇论文时，必须在句末添加引用格式 [^论文id]
- 在介绍各论文贡献时，每篇论文用"- "开头分点列出
- 内容要准确、简洁、有条理
- 不需要在最后整理参考文献列表
- 不要翻译作者名字，保持原文


请生成详细的主题总结报告:"""
    
    else:  # English
        return f"""You are a research paper analysis expert. Please generate a comprehensive summary report for the given collection of papers under a specific topic.

Topic: {topic}

Paper Information:
{papers_text}

Please generate a summary following these requirements:

1. **Topic Overview**: Briefly introduce the background and importance of this research topic

2. **Individual Paper Contributions**: Analyze each paper in detail, highlighting the unique contributions of different papers to this topic, using the following format:
   "[First author] from [Institution] and colleagues proposed [specific method/model], with main contributions being [innovation points and problems solved]. Experiments on [datasets] showed [specific improvements] compared to [baseline methods][^paper_id]."

3. **Technical Trends**: Summarize the main technical approaches and methodological evolution adopted by different papers in this topic

4. **Datasets and Evaluation**: Compile the main datasets and evaluation metrics used in the papers

Requirements:
- Output in English
- Emphasize the unique contributions and innovations of each paper
- Simplify author information to "[First author] from [Institution] and colleagues" format
- Must add citation format [^paper_id] at the end when describing each paper
- When introducing individual paper contributions, list each paper starting with "**"
- Content should be accurate, concise, and well-organized
- Do not need to organize a reference list at the end
- Do not translate author names, keep them in original form

Please generate a detailed topic summary report:"""
