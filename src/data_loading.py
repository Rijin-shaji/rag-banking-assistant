import os

g = Github()
downloaded_files = []
repo = g.get_repo("Rijin-shaji/rag-banking-assistant")
files = repo.get_contents("data")
DOWNLOAD_DIR = "D:/New folder (2)/bank_langchain/datas"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

try:
    with open("processed_files.json", "r") as f:
        processed = json.load(f)
except:
    processed = {}

for file in files:

    if not file.name.endswith(".pdf"):
        continue

    current_sha = file.sha

    if processed.get(file.name) == current_sha:
        print(f"Skipping {file.name}")
        continue

    print(f"Processing {file.name}")

    pdf_data = requests.get(file.download_url).content

    file_path = os.path.join(DOWNLOAD_DIR, file.name)

    with open(file_path, "wb") as f:
        f.write(pdf_data)

    downloaded_files.append(file_path)
    processed[file.name] = current_sha

with open("processed_files.json", "w") as f:
    json.dump(processed, f, indent=4)

def get_downloaded_files():
    return downloaded_files
