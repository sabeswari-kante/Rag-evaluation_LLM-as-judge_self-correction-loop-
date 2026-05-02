
from unittest import result
from dotenv import load_dotenv
load_dotenv()
import os
from typing import Any,Dict,List
from langchain_core.documents import Document
import certifi
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
import ssl 

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 2, max_breadth = 20,max_pages = 750)
tavily_crawl = TavilyCrawl()

def tool_crawl():
    res = tavily_crawl.invoke({
        'url':'https://python.langchain.com/',
        'max_depth':2,
        'extract_depth': 'advanced',
        # 'instructions': 'ai agents'
    })

    print('Tavily processing is on going..')

    all_doc = res['results']
    print('SToring metadata-----')
    all_doc = [Document(page_content= result['raw_content'],metadata={'source':result['url']}) for result in res['results']]
    # print(all_doc)
    return all_doc

if __name__ == "__main__":
    tool_crawl()


