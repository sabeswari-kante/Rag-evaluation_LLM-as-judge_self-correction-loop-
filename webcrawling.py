

from dotenv import load_dotenv
load_dotenv()
import os
from typing import Any,Dict,List
from langchain_core.documents import Document
import os
import time
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader


os.makedirs("data", exist_ok=True)

ESSAYS = {
    "do_things_that_dont_scale": "http://paulgraham.com/ds.html",
    "how_to_get_startup_ideas":  "http://paulgraham.com/startupideas.html",
    "keep_your_identity_small":  "http://paulgraham.com/identity.html",
}


def scrape_essay(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Paul Graham's site uses <table> layout — extract all meaningful text blocks
    paragraphs = soup.find_all(["p", "font"])
    text = "\n\n".join(
        p.get_text(separator=" ", strip=True)
        for p in paragraphs
        if len(p.get_text(strip=True)) > 50
    )
    return text


def download_all_essays() -> list[str]:
    saved_paths = []

    for filename, url in ESSAYS.items():
        filepath = os.path.join('data', f"{filename}.txt")

        try:
            text = scrape_essay(url)

            if len(text) < 200:
                print(f"  WARNING: Very short content ({len(text)} chars) — skipping")
                continue

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  Saved → {filepath} ({len(text)} chars)")
            saved_paths.append(filepath)

        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(1)  

    return saved_paths


def load_with_langchain(paths: list[str]) -> list[Document]:
    documents = []

    for path in paths:
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()

            #  metadata storing
            for doc in docs:
                doc.metadata["filename"] = os.path.basename(path)
                doc.metadata["char_count"] = len(doc.page_content)
                doc.metadata["word_count"] = len(doc.page_content.split())

            documents.extend(docs)
            print(f"  LangChain loaded: {os.path.basename(path)} "
                  f"| {docs[0].metadata['word_count']} words")

        except Exception as e:
            print(f"  ERROR loading {path}: {e}")

    return documents


def print_summary(documents: list[Document]):
    print("\n" + "="*50)
    print("DATA DOWNLOAD SUMMARY")
    print("="*50)
    total_words = sum(d.metadata["word_count"] for d in documents)
    total_chars = sum(d.metadata["char_count"] for d in documents)

    for doc in documents:
        print(f"  {doc.metadata['filename']:<40} "
              f"{doc.metadata['word_count']:>6} words  "
              f"{doc.metadata['char_count']:>7} chars")

    print(f"  {'TOTAL':<40} {total_words} words  {total_chars} chars")
    print('saved')



if __name__ == "__main__":
    saved_paths = download_all_essays()
    documents = load_with_langchain(saved_paths)
    print_summary(documents)
    


