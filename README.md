# 🧠 Agentic-Science

**Agentic-Science** is an automated system for **daily collection, analysis, and summarization of arXiv papers**.  
It intelligently retrieves newly published arXiv papers each day, analyzes their content, categorizes them into coherent research themes, and aggregates structured daily reports in both English and Chinese.  
Examples of generated reports can be found in:  
📄 [`examples/en.md`](examples/en.md) | [`examples/zh.md`](examples/zh.md)

---

## 🌍 Overview

Agentic-Science aims to **transform raw academic streams into structured scientific intelligence**.  
Every day, the system:
1. Fetches the latest arXiv papers from designated categories (e.g., `cs.CL`, `cs.LG`, `cs.AI`, etc.).
2. Parses and summarizes each paper’s abstract, introduction, and methods.
3. Clusters papers into semantically related **topics** using embedding-based similarity and LLM reasoning.
4. Produces **daily research digests** that summarize insights across these themes — capturing the evolving landscape of AI and related fields.

---

## 🚀 Features

- **Automatic paper collection** from arXiv OAI-PMH and RSS feeds  
- **Semantic clustering** for topic-based grouping of research trends  
- **Multi-language summarization** (English & Chinese)  
- **Structured daily reports** with title, abstract, highlights, and cross-paper synthesis  
- **Pluggable pipelines** for expansion to new domains or summarization strategies  

---

## 🧩 Current To-Do List

1. **HTML adaptation** —  
   Current full-text fetching is based on arXiv HTML pages; papers without HTML versions are not yet supported.

2. **Parsing robustness** —  
   Paper content extraction still faces boundary issues (e.g., inaccurate content segmentation or insufficient pipeline context).

3. **Scalability** —  
   Paper data is currently stored as local files. As the dataset grows, it will be migrated to a scalable data warehouse (e.g., Elasticsearch or similar).

4. **Topic & Author Timelines** —  
   Upcoming features include:
   - **Topic Timeline**: Generate historical research evolution reports for a given theme.  
   - **Author Timeline**: Summarize the research trajectory of specific authors.

5. **Agentic Search Integration** —  
   Future versions will include **agentic retrieval and QA**, enabling:
   - Interactive question-answering on report content  
   - Cross-paper reasoning and evidence synthesis for open-ended research questions

---

## 📬 Coming Soon: Daily Subscription

Agentic-Science will provide a **daily email subscription** service delivering the latest summarized reports directly to your inbox.  
Researchers and enthusiasts will be able to subscribe to receive:
- Daily research digests  
- Topic-focused reports  
- Personalized updates  

💡 Stay tuned — we’ll soon open a **sign-up form** for early subscribers!

---

## 🧰 Tech Stack

- **Python 3.10+**
- **FastAPI + aiohttp** for async paper retrieval
- **LangChain / OpenAI API** for LLM-based summarization and clustering
- **Markdown & HTML report generators**
- Planned integration with **Elasticsearch / Milvus** for scalable storage

---

## 🧑‍💻 Example Outputs

- 📘 [`example/en.md`](example/en.md) — English Daily Summary  
- 📗 [`example/zh.md`](example/zh.md) — Chinese Daily Summary  

Each report provides:
- 🏷️ Categorized research topics  
- 🧩 Paper summaries with source links  
- 🔍 Aggregated insights across related works  

---

## 📄 License

This project is released under the **Apache 2.0 License**.

---

### ✨ Vision

> **From data to discovery — Agentic-Science builds an evolving, intelligent system that understands and organizes global scientific progress.**