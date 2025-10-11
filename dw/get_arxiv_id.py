import requests

# url = "https://oaipmh.arxiv.org/oai"
# params = {
#     "verb": "ListIdentifiers",
#     "metadataPrefix": "oai_dc",
#     "set": "cs:cs.CL"
# }

# try:
#     response = requests.get(url, params=params)
#     response.raise_for_status()  # 如果请求失败则会抛出异常
#     print(response.text)
# except requests.exceptions.RequestException as e:
#     print(f"请求失败: {e}")
#     # 如果请求失败，打印出实际发送的URL
#     print(f"发送的URL是: {response.url}")

import requests
r = requests.get("https://worldtimeapi.org/api/timezone/Asia/Shanghai")
print(r.json()["datetime"])