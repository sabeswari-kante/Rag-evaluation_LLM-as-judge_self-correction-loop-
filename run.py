from webcrawling import tool_crawl
from index_db import main as index_main
from core import run_llm

if __name__ == "__main__":
    print("-------------Crawling-------------")
    
    print("-------------Indexing-------------")
    index_main()   
    
    print("-------------Querying-------------")
    result = run_llm(query="what are ai agents?")
    print(result["answer"])