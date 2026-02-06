# python3 finetune.py --pdb_dir data/pdb_files --epochs 20 --lr 0.01 --model_type=ligand_mpnn
import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from model_utils import ProteinMPNN
from data_utils import parse_PDB, featurize, get_score, alphabet

class PDBDataset:
    def __init__(self, pdb_list, pdb_dir, device, model_type, atom_context_num=16):
        self.pdb_list = pdb_list
        self.pdb_dir = pdb_dir
        self.device = device
        self.model_type = model_type
        self.atom_context_num = atom_context_num

    def __len__(self):
        return len(self.pdb_list)

    def __getitem__(self, idx):
        pdb_id = self.pdb_list[idx]
        # Try different extensions if needed, but assuming .pdb
        pdb_path = os.path.join(self.pdb_dir, pdb_id + ".pdb")
        if not os.path.exists(pdb_path):
            # Try finding it recursively or with other extensions if needed
            # For now, just logging error
            return None


        # Parse PDB
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb_path,
            device=self.device,
            chains=[],
            parse_all_atoms=True, 
            parse_atoms_with_zero_occupancy=False
        )
        
        # Add chain_mask - all residues are designable (1) by default
        protein_dict["chain_mask"] = torch.ones_like(protein_dict["mask"])
        # Featurize
        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=True , 
            number_of_ligand_atoms=self.atom_context_num,
            model_type=self.model_type
        )
        feature_dict["batch_size"] = 1 
        return feature_dict
    

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load list
    if args.json_list:
        with open(args.json_list, 'r') as f:
            pdb_ids = json.load(f)
    else:
        # Default to all .pdb files in directory if no json list
        pdb_ids = [f[:-4] for f in os.listdir(args.pdb_dir) if f.endswith(".pdb")]

    random.seed(42)
    random.shuffle(pdb_ids)

    split_idx = int(0.9 * len(pdb_ids))
    train_ids = pdb_ids[:split_idx]
    test_ids = pdb_ids[split_idx:]
            
    train_dataset = PDBDataset(train_ids, args.pdb_dir, device, args.model_type)
    test_dataset = PDBDataset(test_ids, args.pdb_dir, device, args.model_type)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device) if args.checkpoint_path else None
    
    k_neighbors = checkpoint["num_edges"] if checkpoint else 48
    atom_context_num = checkpoint["atom_context_num"] if checkpoint and "atom_context_num" in checkpoint else 16
    
    # Model
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors, 
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type
    )
    
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_dataset)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(
            0.0,
            (total_steps - step) / max(1, total_steps - warmup_steps),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"TRAINING with {len(train_dataset)} PDBs TESTING with {len(test_dataset )} PDBs")

    for epoch in range(args.epochs):
        total_loss = 0
        count = 0
        
        train_indices = np.random.permutation(len(train_dataset))
        # *=====================================*
        # TRAIN
        # *=====================================*
        for i in train_indices:

            feature_dict = train_dataset[i]
            if feature_dict is None:
                continue
            optimizer.zero_grad()
            
            # Prepare additional features needed for score/sample
            B, L = feature_dict["S"].shape
            feature_dict['randn'] = torch.randn([B, L], device=device)
            feature_dict['symmetry_residues'] = [[]]
            feature_dict['symmetry_weights'] = [[]]
            

            
            output_dict = model.score(feature_dict, use_sequence=True)
            # Loss
            loss, _ = get_score(
                feature_dict["S"].long(),
                output_dict["log_probs"],
                feature_dict["mask"] * feature_dict["chain_mask"]
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            count += 1
            if count % args.log_interval == 0:
                print(f"Epoch {epoch}, Step {count}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / count if count > 0 else 0
        print(f"[TRAIN] Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        model.eval()
        test_total_loss = 0.0
        test_count = 0
        test_indices = np.random.permutation(len(test_dataset))
        # *=====================================*
        # TEST
        # *=====================================*
        with torch.no_grad():
            for i in test_indices:
                feature_dict = test_dataset[i]
                if feature_dict is None:
                    continue

                B, L = feature_dict["S"].shape
                feature_dict["randn"] = torch.randn([B, L], device=device)
                feature_dict["symmetry_residues"] = [[]]
                feature_dict["symmetry_weights"] = [[]]

                output_dict = model.score(feature_dict, use_sequence=True)

                loss, _ = get_score(
                    feature_dict["S"].long(),
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )

                test_total_loss += float(loss.item())
                test_count += 1

        test_avg_loss = test_total_loss / test_count if test_count > 0 else 0.0
        print(f"[TEST]  Epoch {epoch} finished. Test Avg Loss: {test_avg_loss:.4f}")
        model.train()

        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        save_path = os.path.join(args.out_dir, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'num_edges': k_neighbors,
            'atom_context_num': atom_context_num
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_list", type=str, default="", help="Path to JSON list of PDB IDs. Optional if pdb_dir contains all files.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files")
    parser.add_argument("--out_dir", type=str, default="./training_output", help="Output directory")
    parser.add_argument("--checkpoint_path", type=str, default="./model_params/ligandmpnn_v_32_030_25.pt", help="Initial checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="protein_mpnn", choices=["protein_mpnn", "ligand_mpnn", "soluble_mpnn", "global_label_membrane_mpnn", "per_residue_label_membrane_mpnn"])
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
