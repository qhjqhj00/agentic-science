from openai import OpenAI

def get_agent(base_url, api_key):
    llm = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    return llm