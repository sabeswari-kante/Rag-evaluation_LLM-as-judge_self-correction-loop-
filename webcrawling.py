
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
tavily_map = TavilyMap(max_depth =5, max_breadth = 20,max_pages = 1000)
tavily_crawl = TavilyCrawl()

def tool_crawl():
    # TEMP DEBUG - remove after
    map_res = tavily_map.invoke({
        'url': 'https://python.langchain.com/',
        'max_depth': 2,
        'max_breadth': 5,
        'max_pages': 10,
    })
    print("MAP RESPONSE:", map_res)
    print("MAP KEYS:", map_res.keys() if hasattr(map_res, 'keys') else type(map_res))
    return []  # stop here for now
    
def tool_crawl():
    print("Step 1: Mapping all URLs...")
    map_res = tavily_map.invoke({
        'url': 'https://python.langchain.com/',
        'max_depth': 5,
        'max_breadth': 20,
        'max_pages': 1000,
    })

    all_urls = map_res.get('results', [])
    print(f"Found {len(all_urls)} URLs")

    filtered_urls = [
        url for url in all_urls
        if 'python.langchain.com' in url
        and '%7B' not in url          # remove template URLs like ${CHAT_APP_URL}
        and '#' not in url
        and not url.endswith(('.png', '.jpg', '.svg', '.pdf'))
    ]
    print(f"Filtered to {len(filtered_urls)} clean URLs")

    # content extracting from her in batches  20
    all_results_data = []
    batch_size = 20

    for i in range(0, len(filtered_urls), batch_size):
        batch = filtered_urls[i:i + batch_size]
        try:
            extract_res = tavily_extract.invoke({
                'urls': batch,
                'extract_depth': 'advanced',
            })
            all_results_data.extend(extract_res.get('results', []))
            print(f"Batch {i//batch_size + 1}/{(len(filtered_urls)//batch_size) + 1} done")
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
            continue

    print('Tavily processing is on going..')
    # all_results_data = res['results']
    #to fix deduplicates by content
    seen_urls= set()
    seen_fingerprints = set()
    unique_results = []
    for result in all_results_data:
        content = result['raw_content'].strip()
        url = result.get('url', '')
        if not content or len(content) < 100:  # empty/small pages
            continue
        if url in seen_urls:
            continue
        # catches same content, different URL
        fingerprint = content[:300].lower().replace(" ", "")
        if fingerprint in seen_fingerprints:
            continue

        seen_urls.add(url)
        seen_fingerprints.add(fingerprint)
        unique_results.append(result)
    print(f'Deduplicated {len(all_results_data) - len(unique_results)} duplicate results')

    print('SToring metadata-----')
    all_doc = [Document(page_content= result['raw_content'],metadata={'source':result['url']}) for result in unique_results]
    # print(all_doc)
    return all_doc

if __name__ == "__main__":
    tool_crawl()


