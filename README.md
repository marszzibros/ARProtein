# Protein Structure Selection Pipeline

This repository identifies, downloads, and filters protein structures from the **Protein Data Bank (PDB)** related to:

- **Ankyrin repeat proteins**
- **DARPins (Designed Ankyrin Repeat Proteins)**

The pipeline collects initial PDB IDs, downloads metadata and structure files, removes duplicates, and applies additional biological and structural filters to select relevant protein complexes.

---

## 1. Initial PDB ID Collection

PDB IDs were obtained using two keyword searches on the [Protein Data Bank](https://www.rcsb.org/):

1. **"Ankyrin repeat protein"**  
2. **"DARPin"**

The resulting ID lists are stored in:

- `data/Ankyrin_repeat_protein.txt`  
- `data/DARPin.txt`

---

## 2. Downloading PDB and Metadata Files

All structure files (`.pdb`) and metadata files associated with the selected IDs are downloaded using:

data_collections/data_collection.sh

Downloaded files are saved in:

- `data/{search_term}/metadata_files/`
- `data/{search_term}/pdb_files/`

> **Note:** Due to GitHub storage limits, the full dataset is stored in the VACC cluster at:  
> `/gpfs2/scratch/jjung2/ARProtein/`

---

## 3. Removing Duplicate Entries

Since some structures may appear in both search terms, the lists are combined and **duplicate PDB IDs are removed** to create a unified non-redundant set of structures.

---

## 4. Filtering by Protein Type

Metadata files are scanned for the following keywords:

- `filament`
- `trpv`
- `transient receptor potential vanilloid`

Only structures containing at least one of these terms are retained.

---

## 5. Structural Filtering Criteria

Protein structures are further filtered:

- **Number of protein chains ≥ 2**
- **polymer composition = heteromeric protein**

Only PDB files that meet both criteria are included in the final dataset.

---

## 6. Final Output

The list of all structures that pass every filtering step is saved in:

- `data/selected_proteins.txt`

This file contains the final curated set of PDB IDs.

---

## Advanced Search Instructions

To verify or refine selections using the PDB web interface:

1. Go to the [Protein Data Bank](https://www.rcsb.org/).
2. Open **Advanced Search**.
3. Navigate to **Structure Attributes → ID and Keywords → Entry ID**.
4. After **“is any of”**, paste the contents of `selected_proteins.txt` ID files

This will reproduce the selected structures directly in the PDB interface.


