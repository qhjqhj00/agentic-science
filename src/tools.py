import requests
import time
import random

import asyncio
import aiohttp

async def get_one_page(url, max_retries=5, timeout=30, session=None):
    """
    Robustly fetch a web page asynchronously with retry logic and error handling.
    
    Args:
        url (str): The URL to fetch
        max_retries (int): Maximum number of retry attempts
        timeout (int): Request timeout in seconds
        session (aiohttp.ClientSession): Optional session to reuse
    
    Returns:
        str or None: Page content if successful, None otherwise
    """
    if not url or not isinstance(url, str):
        return None
    
    # Create session if not provided
    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(max_retries + 1):
            try:
                async with session.get(
                    url, 
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        content = await response.text()
                        return content
                    elif response.status == 403:
                        if attempt < max_retries:
                            # Exponential backoff with jitter for 403 errors
                            wait_time = min(300, (2 ** attempt) * 10 + random.uniform(0, 10))
                            print(f"403 error, retrying in {wait_time:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                            await asyncio.sleep(wait_time)
                            continue
                    elif response.status in [429, 502, 503, 504]:
                        if attempt < max_retries:
                            # Rate limiting or server errors - retry with backoff
                            wait_time = min(60, (2 ** attempt) * 5 + random.uniform(0, 5))
                            print(f"Server error {response.status}, retrying in {wait_time:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                            await asyncio.sleep(wait_time)
                            continue
                    else:
                        # Other HTTP errors - don't retry
                        print(f"HTTP error {response.status} for URL: {url}")
                        return None
                        
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    print(f"Timeout error, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(random.uniform(1, 3))
                    continue
                else:
                    print(f"Timeout error after {max_retries + 1} attempts for URL: {url}")
                    return None
                    
            except aiohttp.ClientConnectionError:
                if attempt < max_retries:
                    print(f"Connection error, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(random.uniform(2, 5))
                    continue
                else:
                    print(f"Connection error after {max_retries + 1} attempts for URL: {url}")
                    return None
                    
            except aiohttp.ClientError as e:
                print(f"Client error: {e}")
                return None
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
        
        print(f"Failed to fetch URL after {max_retries + 1} attempts: {url}")
        return None
        
    finally:
        if close_session:
            await session.close()


async def get_multiple_pages(urls: list, max_concurrent: int = 10, session: aiohttp.ClientSession = None):
    """
    Concurrently fetch multiple pages using get_one_page function.
    
    Args:
        urls: List of URLs to fetch
        max_concurrent: Maximum number of concurrent requests (default: 10)
        session: Optional aiohttp session to reuse
    
    Returns:
        List of results in the same order as input URLs. None for failed requests.
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()
    
    try:
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(url):
            async with semaphore:
                return await get_one_page(url, session=session)
        
        # Create tasks for all URLs
        tasks = [asyncio.create_task(fetch_with_semaphore(url)) for url in urls]
        
        # Wait for all tasks to complete with progress bar
        from tqdm.asyncio import tqdm
        results = []
        for task in tqdm.as_completed(tasks, desc="Fetching pages"):
            result = await task
            results.append(result)
        
        # Reorder results to match input URL order
        task_to_result = {}
        for i, task in enumerate(tasks):
            for j, result in enumerate(results):
                if task.done() and task.result() == result:
                    task_to_result[i] = result
                    break
        
        # Create ordered results list
        ordered_results = []
        for i in range(len(urls)):
            if i in task_to_result:
                ordered_results.append(task_to_result[i])
            else:
                ordered_results.append(None)
        
        results = ordered_results
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error fetching {urls[i]}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
        
    finally:
        if close_session:
            await session.close()

async def fetch_html_via_api(urls, token, api_endpoint):
    """
    Fetch HTML content via API
    
    Args:
        urls: Single URL string or list of URLs
        token: API token for authentication
        api_endpoint: API endpoint
    Returns:
        Single HTML string if input is single URL, or list of HTML strings if input is list
    """
    import aiohttp
    
    # Handle single URL case
    single_url = isinstance(urls, str)
    if single_url:
        urls = [urls]
    
    async with aiohttp.ClientSession() as session:
        try:
            data = {
                "urls": urls,
                "token": token
            }
            
            async with session.post(
                api_endpoint,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)

            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Assuming API returns {"results": [html_content1, html_content2, ...]}
                    return result
                else:
                    print(f"Error fetching URLs: HTTP {response.status}")
                    return None 
                    
        except Exception as e:
            print(f"Error fetching URLs: {e}")
            return None if single_url else {"results": {}}



if __name__ == "__main__":
    from config.list_urls import get_list_urls
    from src.utils import load_json
    api_dict = load_json("config/api_dict.json")
    # urls = get_list_urls("cs.CL", 25)
    html = asyncio.run(fetch_html_via_api("https://arxiv.org/html/2510.08575v1", api_dict["search_token"], api_dict["web_unlocker_api"]))
    print(html)
