import os
import json
import pandas as pd

search_terms = ['Ankyrin_repeat_protein', 'DARPin']
filter_terms = ['filament', 'trpv', 'transient receptor potential vanilloid']

metadata_folder = "data/{search_term}/metadata_files"
pdb_folder = "data/{search_term}/pdb_files"
csv_file = "data/{search_term}/pdb_list.csv"

def contains_forbidden(text):
    text_lower = text.lower()
    return any(f in text_lower for f in filter_terms)

# two search terms
for folder in search_terms:
    protein_chains = -1
    polymer_composition = ""

    # read files and make paths absolute
    df = pd.read_csv(csv_file.format(search_term=folder))
    df['metadata_file'] = df['metadata_file'].apply(lambda x: os.path.join(metadata_folder.format(search_term=folder), x))
    df['pdb_file'] = df['pdb_file'].apply(lambda x: os.path.join(pdb_folder.format(search_term=folder), x))

    # open json files and filter
    filtered_data = []
    for index, row in df.iterrows():
        with open(row['metadata_file'], 'r') as f:
            metadata = json.load(f)
        
        if contains_forbidden(metadata["struct"]["title"]):
            continue
        protein_chains = metadata["rcsb_entry_info"]["deposited_polymer_entity_instance_count"]
        polymer_composition = metadata["rcsb_entry_info"]["polymer_composition"]

        filtered_data.append([row['pdb_file'], row['metadata_file'], protein_chains, polymer_composition])
    filtered_df = pd.DataFrame(filtered_data, columns=['pdb_file', 'metadata_file', 'protein_chains', 'polymer_composition'])
    filtered_df.to_csv(f"data/{folder}/filtered_pdb_list.csv", index=False)
