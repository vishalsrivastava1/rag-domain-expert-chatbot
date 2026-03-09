import os
import re
import time
import requests
from urllib.parse import urlparse
from ddgs import DDGS

QUERIES = [
    "site:nasa.gov filetype:pdf NASA",
    "site:nasa.gov filetype:pdf NASA report",
    "site:nasa.gov filetype:pdf NASA mission",
    "site:nasa.gov filetype:pdf NASA technical report",
    "site:nasa.gov filetype:pdf NASA research"
]

MAX_PDFS = 60
SAVE_DIR = "data/raw"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)


def safe_filename(name: str) -> str:
    name = re.sub(r'[^\w\-. ]', '_', name)
    return name[:120]


def get_pdf_links(queries, max_results_per_query=50):
    links = []
    seen = set()

    with DDGS() as ddgs:
        for query in queries:
            print(f"Searching: {query}")
            try:
                results = ddgs.text(query, max_results=max_results_per_query)
                for r in results:
                    url = r.get("href") or r.get("url")
                    if not url:
                        continue

                    if ".pdf" in url.lower() and url not in seen:
                        seen.add(url)
                        links.append(url)

            except Exception as e:
                print(f"[SEARCH FAIL] {query} -> {e}")

    return links


def download_pdf(url: str, save_dir: str, idx: int):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        parsed = urlparse(url)
        original_name = os.path.basename(parsed.path)

        if not original_name.lower().endswith(".pdf") or len(original_name.strip()) == 0:
            original_name = f"nasa_doc_{idx}.pdf"

        filename = safe_filename(f"{idx:02d}_{original_name}")
        file_path = os.path.join(save_dir, filename)

        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"[OK] Downloaded: {filename}")
        return filename

    except Exception as e:
        print(f"[FAIL] {url} -> {e}")
        return None


def main():
    pdf_links = get_pdf_links(QUERIES, max_results_per_query=60)
    print(f"\nFound {len(pdf_links)} candidate PDF links")

    downloaded = 0
    with open("docs/source_list.csv", "w", encoding="utf-8") as meta:
        meta.write("doc_id,file_name,source_url,type,notes\n")

        for url in pdf_links:
            if downloaded >= MAX_PDFS:
                break

            file_name = download_pdf(url, SAVE_DIR, downloaded + 1)
            if file_name:
                meta.write(f'{downloaded+1},"{file_name}","{url}",pdf,"NASA public PDF"\n')
                downloaded += 1

            time.sleep(1)

    print(f"\nDone. Downloaded {downloaded} PDFs into {SAVE_DIR}")


if __name__ == "__main__":
    main()