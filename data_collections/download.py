import os
from download_rcsb import RCSBDownloader
import sys


os.system(f"mkdir ../finetuning/data/pdb_files")
os.system(f"mkdir ../finetuning/data/metadata_files")
downloader = RCSBDownloader(pdb_folder=f"../finetuning/data/pdb_files", metadata_folder=f"../finetuning/data/metadata_files")
with open(f"../finetuning/data/pdb_ids.txt", "r") as f:
    pdb_ids = f.read()
    pdb_ids = pdb_ids.strip().split(", ")
    
for pdb_id in pdb_ids:
    try:
        downloader.download(pdb_id)
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")
downloader.save_csv(f"../finetuning/data/pdb_list.csv")