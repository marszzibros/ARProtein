import pandas as pd

df_1 = pd.read_csv("data/Ankyrin_repeat_protein/filtered_pdb_list.csv")
df_2 = pd.read_csv("data/DARPin/filtered_pdb_list.csv")

combined_df = pd.concat([df_1, df_2], ignore_index=True)
combined_df['PDB_ID'] = combined_df['pdb_file'].apply(lambda x: x.split('/')[-1].split('.')[0])
combined_df.drop_duplicates(subset=['PDB_ID'], inplace=True)


protein_chains_limit = 2
complex_chains_limit = 2

final_df = combined_df[
    (combined_df['protein_chains'] >= protein_chains_limit) &
    (combined_df['complex_chains'] >= complex_chains_limit)
]
final_df.to_csv("data/selected_proteins.csv", index=False)

# save pdb ids to a text file in comma-separated format
with open("data/selected_proteins.txt", "w") as f:
    f.write(",".join(final_df['PDB_ID'].tolist()))
