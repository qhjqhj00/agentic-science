

def get_list_urls(category: str, num: int):
    if num not in [25, 50, 100, 250, 500, 1000, 2000]:
        raise ValueError(f"Invalid number of papers: {num}. Please choose from [25, 50, 100, 250, 500, 1000, 2000]")
    return f"https://arxiv.org/list/{category}/recent?show={num}"