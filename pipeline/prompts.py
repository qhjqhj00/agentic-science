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

def paper_attention_score_prompt(title, authors):
    authors_text = ""
    org_list = []
    for author in authors["authors"]:
        authors_text += f"{author['name']}\n"
        org_list.extend(author['org'])
    org_list = list(set(org_list))
    if org_list:
      org_text = "\n".join(org_list)
    else:
      org_text = "No affiliations found"
    return f"""You are an expert researcher evaluating the attention-grabbing potential of academic papers. Based on the paper title, authors, and their affiliations, please rate the paper's "attention score" from 1-5.

Paper Information:
- Title: {title}
- Authors: 
{authors_text}
- Affiliations:
{org_text}


Rating Criteria:

5 points: Highly attention-grabbing
- Title is very compelling (e.g., "GPT-5 Technical Report", "Attention Is All You Need", breakthrough announcements)
- Authors are from top-tier CS institutions (OpenAI, Google, Meta, Stanford, CMU, MIT, etc.)
- Combination of prestigious institutions and exciting title

4 points: Good attention-grabbing
- Authors from well-known CS institutions (top universities like Berkeley, UIUC, NUS, PKU, Tsinghua, or major tech companies like Microsoft, NVIDIA, Baidu, ByteDance, Alibaba, etc.)
- Title is appealing and addresses relevant research areas
- Strong institutional reputation with interesting research direction

3 points: Moderately attention-grabbing  
- Authors from less well-known institutions or regional universities
- Title is fairly ordinary or addresses common research problems without novel angles
- Limited institutional prestige and conventional research topics

2 points: Below average attention-grabbing
- Institutions are reasonably authoritative (US universities, reputable Chinese/European institutions)
- Title addresses popular/trending topics or tasks but lacks novelty
- Good balance of institutional credibility but topic relevance is limited

1 point: Low attention-grabbing
- Authors from less prestigious or unknown institutions
- Title focuses on niche domains or very specific tasks (e.g., tasks for minority languages)
- Limited broad appeal or institutional backing

Instructions:
1. Consider both the title's appeal and institutional prestige
2. Weight recent trending topics and breakthrough claims higher
3. Recognize top-tier institutions in computer science and AI
4. Be objective about the likely attention the paper would receive in the research community

Output format (JSON):
{{"score": 5}}

Please evaluate the paper and return only the JSON with the attention score:"""


PAPER_ANALYSIS_PROMPT = """You are an expert researcher analyzing academic papers. Given the following sections of a research paper, please analyze and extract key information.

Paper sections:
- Title: {title}
- content: {content}

Please analyze the paper and answer the following three questions:

1. Scenario: What is the core problem this paper aims to solve? Why is it important?
2. Value: What value does this paper bring in terms of technology/data/theory? What new methods, datasets, or theories are proposed? If it's a survey paper, how does it analyze the field? What makes it different from previous work?
3. Insight: What insights can this paper provide through experiments or analysis? What are the experimental conclusions? On which datasets does it outperform which baselines?
4. Keywords: What are the 3 most representative keywords that capture the core content and contributions of this paper? These should be technical terms or phrases that best describe the main research areas, methods, or applications (e.g., "long-context processing", "agentic tool-using", "video retrieval", "parameter pruning").

Output format (JSON):
{{
  "scenario": "Detailed description of the core problem being addressed and its importance. Explain the application domain, the specific challenges, and why solving this problem matters for the field or real-world applications.",
  "value": "Comprehensive description of the technical, data, or theoretical contributions. Detail the new methods, algorithms, datasets, or theoretical frameworks proposed. For survey papers, describe how the field is analyzed and organized. Explain what makes this work novel compared to existing approaches and what gaps it fills.",
  "insight": "Rich description of the insights gained from experiments or analysis. Include specific experimental findings, performance improvements, dataset comparisons, baseline comparisons, and key conclusions. Mention specific metrics, improvements, and what the results reveal about the problem or solution.",
  "keywords": "List of 3 most representative keywords that capture the core content and contributions of this paper. These should be technical terms or phrases that best describe the main research areas, methods, or applications (e.g., \"long-context processing\", \"agentic tool-using\", \"video retrieval\", \"parameter pruning\")."
}}

Instructions:
1. Extract information directly from the provided sections
2. If information for any field is not found in the provided content, set the value to null
3. Provide rich and comprehensive descriptions for each field
4. Focus on factual information rather than subjective interpretations
5. Include specific details about methods, datasets, baselines, and results when available

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
    # Sort papers by attention score in descending order
    papers = sorted(papers, key=lambda x: x.get('parsed_info', {}).get('attention_score', {}).get('score', 0), reverse=True)

    papers_text = ""
    for paper in papers:
      author_text = ""
      for author in paper['parsed_info']['authors_info']['authors']:
        org_text = ", ".join(author['org'])
        author_text += f"{author['name']} from {org_text}\n"
      paper_info_text = ""
      for key, value in paper['parsed_info']['paper_info'].items():
        if key == "keywords":
          continue
        paper_info_text += f"{key}: {value}\n"
      attention_score_text = f"attention score: {paper['parsed_info']['attention_score']['score']}\n"
      paper_id_text = f"paper id: {paper['_idx']}\n"
      papers_text += f"title: {paper['title']}\n"
      papers_text += f"{paper_id_text}\n"
      papers_text += f"authors: {author_text}\n"
      papers_text += f"paper information: {paper_info_text}\n"
        # papers_text += f"attention score: {attention_score_text}\n\n"
      
    if language == "zh":
        return f"""你是一个研究论文分析专家。请为给定主题下的论文集合生成一个全面的总结报告。

主题: {topic}

论文信息:
{papers_text}

请按照以下要求生成总结:

1. **主题概述**: 简要介绍该研究主题的背景和重要性

2. **各论文贡献**: 分别对每篇论文进行详细分析，突出不同论文对该主题的独特贡献，格式如下：
   "来自[机构]的[第一作者]等人研究了[研究内容/问题]，提出了[具体方法/模型]来解决[核心问题]。该方法的主要创新点是[创新点]，产生的价值在于[实际价值和意义]。在[数据集]上的实验表明，相比[基线方法]取得了[具体提升效果]，得出的结论是[主要结论][^论文id]。"

3. **技术趋势**: 总结该主题下不同论文采用的主要技术路线和方法演进

4. **数据集和评估**: 汇总论文中使用的主要数据集和评估指标

要求:
- 使用中文输出
- 重点内容可以用**加粗**等方式强调，增强可读性
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
   "[First author] from [Institution] and colleagues studied [research content/problem], proposing [specific method/model] to solve [core problem]. The main innovation points of this method are [innovation points], and the value lies in [practical value and significance]. Experiments on [datasets] showed [specific improvement effects] compared to [baseline methods], concluding that [main conclusions][^paper_id]."

3. **Technical Trends**: Summarize the main technical approaches and methodological evolution adopted by different papers in this topic

4. **Datasets and Evaluation**: Compile the main datasets and evaluation metrics used in the papers

Requirements:
- Output in English
- Emphasize the unique contributions and innovations of each paper
- Simplify author information to "[First author] from [Institution] and colleagues" format
- Must add citation format [^paper_id] at the end when describing each paper
- When introducing individual paper contributions, list each paper starting with "- "
- Content should be accurate, concise, and well-organized
- Do not need to organize a reference list at the end
- Do not translate author names, keep them in original form

Please generate a detailed topic summary report:"""

def keyword_topic_extraction_prompt(keywords_list):
    """
    Generate a prompt to extract topics from a list of keywords.
    
    Args:
        keywords_list: List of keywords to be categorized into topics
    
    Returns:
        str: Formatted prompt for topic extraction
    """
    keywords_text = "\n".join([f"- {keyword}" for keyword in keywords_list])
    
    return f"""You are a research topic analysis expert. Please analyze the following list of keywords and group them into 10 meaningful research topics that cover as many keywords as possible.

Keywords to analyze:
{keywords_text}

Requirements:
1. Create exactly 10 topics that best represent the research areas covered by these keywords
2. Each topic should be a broad, meaningful research area name
3. Group related keywords under the same topic (e.g., "chain-of-thought reasoning", "multimodal reasoning", "parallel reasoning" should all be grouped under "reasoning")
4. Try to cover as many keywords as possible - minimize orphaned keywords
5. Topic names should be concise and descriptive
6. Each topic should contain at least 2 keywords when possible

Output format (JSON):
[
    {{"topic": "topic_name_1", "keywords": ["keyword1", "keyword2", "keyword3"]}},
    {{"topic": "topic_name_2", "keywords": ["keyword4", "keyword5"]}},
    {{"topic": "topic_name_3", "keywords": ["keyword6", "keyword7", "keyword8"]}},
    ...
    {{"topic": "topic_name_10", "keywords": ["keyword_n"]}}
]

Please analyze the keywords and generate the topic groupings:"""


