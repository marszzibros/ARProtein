import os
from download_rcsb import RCSBDownloader
import sys

file_name = sys.argv[1]

os.system(f"mkdir ../data/{file_name}")
os.system(f"mkdir ../data/{file_name}/pdb_files")
os.system(f"mkdir ../data/{file_name}/metadata_files")

downloader = RCSBDownloader(pdb_folder=f"../data/{file_name}/pdb_files", metadata_folder=f"../data/{file_name}/metadata_files")
with open(f"../data/{file_name}.txt", "r") as f:
    pdb_ids = f.read()
    pdb_ids = pdb_ids.strip().split(",")
    
for pdb_id in pdb_ids:
    try:
        downloader.download(pdb_id)
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")
downloader.save_csv(f"../data/{file_name}/pdb_list.csv")