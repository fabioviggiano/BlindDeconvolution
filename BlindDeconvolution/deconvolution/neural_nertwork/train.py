# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os # Aggiunto import

# Assicurati che il terminale venga eseguito dalla cartella `neural_network`
# o adatta i percorsi di conseguenza.
from dataset import DeblurDataset
from model import UNet

# --- Parametri di Configurazione ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 25
TRAIN_DIR = '../../data_processed/train'
VAL_DIR = '../../data_processed/val'
MODEL_SAVE_PATH = 'unet_deblur_model.pth' # Percorso dove salvare il checkpoint

# (Le funzioni train_one_epoch e validate_model rimangono identiche)
def train_one_epoch(loader, model, optimizer, loss_fn):
    loop = tqdm(loader, leave=True)
    running_loss = 0.0
    model.train()
    for batch_idx, (blurred, sharp) in enumerate(loop):
        blurred = blurred.to(DEVICE)
        sharp = sharp.to(DEVICE)
        predictions = model(blurred)
        loss = loss_fn(predictions, sharp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(loader)

def validate_model(loader, model, loss_fn):
    running_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for blurred, sharp in loader:
            blurred = blurred.to(DEVICE)
            sharp = sharp.to(DEVICE)
            predictions = model(blurred)
            running_val_loss += loss_fn(predictions, sharp).item()
    return running_val_loss / len(loader)


def main():
    print(f"Utilizzando il dispositivo: {DEVICE}")
    
    # 1. Caricare i Dati
    IMAGE_SIZE = 256
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    ])
    
    train_dataset = DeblurDataset(root_dir=TRAIN_DIR, transform=transformations)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    val_dataset = DeblurDataset(root_dir=VAL_DIR, transform=transformations)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Inizializzare il Modello e l'Ottimizzatore
    model = UNet(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- NUOVA SEZIONE: Caricamento del Checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(MODEL_SAVE_PATH):
        print("=> Trovato checkpoint, carico lo stato...")
        checkpoint = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"=> Checkpoint caricato! Riprendo l'addestramento dall'epoca {start_epoch}")
    else:
        print("=> Nessun checkpoint trovato, inizio un nuovo addestramento.")
    # --- FINE NUOVA SEZIONE ---

    # 3. Ciclo di Addestramento e Validazione
    for epoch in range(start_epoch, NUM_EPOCHS): # MODIFICATO: parte da start_epoch
        print(f"\n--- Epoca {epoch+1}/{NUM_EPOCHS} ---")
        
        avg_train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        avg_val_loss = validate_model(val_loader, model, loss_fn)
        
        print(f"Loss di Training Media: {avg_train_loss:.4f} | Loss di Validazione Media: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"=> Nuovo miglior modello trovato con validation loss: {best_val_loss:.4f}. Salvo il checkpoint...")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()