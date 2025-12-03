import requests
import json
import os
import pandas as pd
class RCSBDownloader:
    
    def __init__(self, pdb_folder="pdb_files", metadata_folder="metadata_files"):
        self.PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
        self.PDB_METADATA_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

        self.pdb_folder = pdb_folder
        self.metadata_folder = metadata_folder

        self.file_pairs = []

    def download(self, pdb_id, out_file=None):
        pdb_id = pdb_id.upper()
        url = self.PDB_URL.format(pdb_id=pdb_id)

        r = requests.get(url)
        r.raise_for_status()
        
        with open(os.path.join(self.pdb_folder, f"{pdb_id}.pdb"), "wb") as f:
            f.write(r.content)

        # Download metadata
        meta_res = requests.get(self.PDB_METADATA_URL.format(pdb_id=pdb_id))
        meta_res.raise_for_status()
        metadata = meta_res.json()

        with open(os.path.join(self.metadata_folder, f"{pdb_id}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        self.file_pairs.append((f"{pdb_id}.pdb", f"{pdb_id}_metadata.json"))

    def save_csv(self, csv_path="../data/pdb_list.csv"):
        df = pd.DataFrame(self.file_pairs, columns=["pdb_file", "metadata_file"])
        df.to_csv(csv_path, index=False)