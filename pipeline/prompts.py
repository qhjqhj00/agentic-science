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