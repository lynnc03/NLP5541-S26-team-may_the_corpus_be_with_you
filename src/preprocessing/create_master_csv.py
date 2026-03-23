#create_master_csv.py

import os
import re
import pandas as pd

ROOT = "."


##not all files have audio, save this for later when adding in
AUDIO_EXTS = (".wav", ".mp3", ".aif", ".aiff", ".flac", ".m4a")

#parse the headers in the .cha file
def header_parser(text: str) -> list:
    file_info = []
    for line in text.splitlines():
        if line.startswith("@ID:"):
            main_line = line.split("@ID:", 1)[1]
            main_line = main_line.strip()
            info_parts = [part.strip() for part in main_line.split("|")]

            info = {
                "age": info_parts[3] if len(info_parts) > 3 else "",
                "sex": info_parts[4] if len(info_parts) > 4 else "",
                "role": info_parts[7] if len(info_parts) > 7 else ""
            }

            file_info.append(info)
    
    return file_info

data_rows = []

for corpus in os.listdir(ROOT):
    corpus_dir = os.path.join(ROOT, corpus)
    if not os.path.isdir(corpus_dir): 
        continue

    all_files = []
    for dirpath, _, filenames in os.walk(corpus_dir):
        for fname in filenames:
            all_files.append(os.path.join(dirpath, fname))


    has_audio = any(f.lower().endswith(AUDIO_EXTS) for f in all_files)

    for fpath in all_files:
        if not fpath.lower().endswith(".cha"):
            continue

        relative_corpus_path = os.path.relpath(fpath, corpus_dir)
        relative_root_path = os.path.relpath(fpath, ROOT)

        file_id = os.path.splitext(relative_corpus_path)[0].replace(os.sep, "_")
        file_path = f"./{relative_root_path.replace(os.sep, '/')}"

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception as e:
            print("couldn't read file {fpath}")
            continue

        has_mor = int("%mor:" in text)
        id_records = header_parser(text)

        id_line = id_records[0] if id_records else None

        data_rows.append({
            "file_id": file_id,
            "file_path": fpath,
            "file_type": "cha",
            "label": "", #added manually from directory to ensure no parsing errors here
            "has_audio": "", #add manually based on website directory
            #esp important to add label manually bc some files have things like "siblingsli" for a normal control sibling
            #could also do some sort of agent identification task if wanted
            "age": id_line["age"] if id_line else "",
            "sex": id_line["sex"] if id_line else "",
            "include_v1": "", #add manually based on if want to include
            "label_binary": ""
        })

OUT_PATH = os.path.join(ROOT, "talkbank_project", "file_info", "files_master.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.DataFrame(data_rows).sort_values(["corpus", "file_id"]).reset_index(drop=True)
df.to_csv(OUT_PATH, index=False)


print(df.head(20).to_string(index=False))
print(f"\nWrote {len(df)} rows to {OUT_PATH}")