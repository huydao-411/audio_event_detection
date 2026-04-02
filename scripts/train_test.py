"""
Script to train and test the AST model using a dataset loaded from a CSV metadata file.
"""

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ast_model import AudioSpectrogramTransformer
from scripts.train import Trainer
from utils.dataset import create_data_loaders

def main():
    parser = argparse.ArgumentParser(description="Train and Test AST Model via CSV dataset")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--csv_path', type=str, default='data/processed/spectrograms/processed_metadata.csv', help='Path to processed metadata CSV')
    args = parser.parse_args()

    # Define absolute paths
    config_path = str(PROJECT_ROOT / args.config)
    csv_path = str(PROJECT_ROOT / args.csv_path)

    print("\n" + "="*60)
    print("Audio Event Detection - Train and Test Pipeline")
    print("="*60)
    
    # Check if essential files exist
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
        
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure you have run the preprocessing script to generate the metadata CSV.")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    # 1. Create Data Loaders
    print(f"\nLoading and splitting data from CSV: {csv_path}")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from utils.dataset import AudioEventDataset
    
    # Load configuration for batch size and hardware settings
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    batch_size = config_dict['training']['batch_size']
    num_workers = config_dict.get('hardware', {}).get('num_workers', 4)
    pin_memory = config_dict.get('hardware', {}).get('pin_memory', True)

    metadata_df = pd.read_csv(csv_path)
    
    # Drop rows with NaN labels if any
    metadata_df = metadata_df.dropna(subset=['label'])
    metadata_df['label'] = metadata_df['label'].astype(int)
    
    print(f"\nClass distribution in metadata:")
    print(metadata_df['target_class'].value_counts())
    
    # Shuffle and split: 80% train, 10% val, 10% test
    # Stratify ensures balanced class distributions in all splits
    train_df, temp_df = train_test_split(metadata_df, test_size=0.2, random_state=42, stratify=metadata_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Dataset split completed! Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}\n")
    
    train_dataset = AudioEventDataset(train_df, config_path, mode='train')
    val_dataset = AudioEventDataset(val_df, config_path, mode='val')
    test_dataset = AudioEventDataset(test_df, config_path, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)
    
    # 2. Initialize Model
    print("\nInitializing AST model...")
    model = AudioSpectrogramTransformer(config_path)
    
    # 3. Create Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_path=config_path,
        device=str(device)
    )
    
    # 4. Training
    print("\nStarting the training process...")
    trainer.train()
    
    # 5. Testing
    print("\n" + "="*60)
    print("Evaluating Best Model on Test Set")
    print("="*60)
    
    # Load best model checkpoint saved during training
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
        
    best_model_path = os.path.join(PROJECT_ROOT, config_yaml['paths']['checkpoint_dir'], 'best_model.pth')
    
    if os.path.exists(best_model_path):
        print(f"Loading best weights from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: best_model.pth not found. Evaluating test set using the latest weights.")
    
    # Override the validation loader with test loader to leverage Trainer's validate method for testing
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()
    
    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    if 'accuracy' in test_metrics:
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    if 'precision' in test_metrics:
        print(f"Test Precision: {test_metrics['precision']:.4f}")
    if 'recall' in test_metrics:
        print(f"Test Recall: {test_metrics['recall']:.4f}")
    if 'f1_score' in test_metrics:
        print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
