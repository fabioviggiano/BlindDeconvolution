
# -*- coding: utf-8 -*-
# Dove definirai l'architettura U-Net

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Blocco convoluzionale: (Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # --- Encoder (Contracting Path) ---
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        
        # --- Decoder (Expansive Path) ---
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128) # Il doppio degli input channels per la skip connection
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)  # Il doppio degli input channels per la skip connection

        # --- Output Layer ---
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)

        # Decoder con Skip Connections
        u1 = self.up1(x3)
        # Concatena l'output dell'upsampling con l'output corrispondente dell'encoder
        skip1 = torch.cat([u1, x2], dim=1) 
        u1 = self.conv3(skip1)

        u2 = self.up2(u1)
        skip2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv4(skip2)

        # Output finale
        logits = self.outc(u2)
        return logits

### 3. `neural_network/train.py` - Lo Script di Addestramento

## Scopo:** Questo è il cuore del processo. Importa il modello e il dataset, definisce i parametri di addestramento (learning rate, numero di epoche, ecc.), e implementa il ciclo di training che aggiorna i pesi della rete per minimizzare l'errore tra l'immagine ricostruita e quella nitida originale.

##*Cosa devi fare:**
## Crea il file `neural_network/train.py`. Questo è uno scheletro più complesso, ma rappresenta un ciclo di training standard.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import DeblurDataset
from model import UNet

# --- Parametri di Configurazione ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 25
TRAIN_DIR = '../data_processed/train'
VAL_DIR = '../data_processed/val'
MODEL_SAVE_PATH = 'unet_deblur_model.pth'

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    running_loss = 0.0

    for batch_idx, (blurred, sharp) in enumerate(loop):
        blurred = blurred.to(DEVICE)
        sharp = sharp.to(DEVICE)

        # Forward
        predictions = model(blurred)
        loss = loss_fn(predictions, sharp)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def main():
    print(f"Utilizzando il dispositivo: {DEVICE}")
    
    # 1. Caricare i Dati
    transformations = transforms.Compose([transforms.ToTensor()])
    train_dataset = DeblurDataset(root_dir=TRAIN_DIR, transform=transformations)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = DeblurDataset(root_dir=VAL_DIR, transform=transformations)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Inizializzare il Modello, la Loss e l'Ottimizzatore
    model = UNet(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.MSELoss() # Mean Squared Error è una buona scelta per la ricostruzione
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Ciclo di Addestramento
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoca {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, None)
        
        # --- Ciclo di Validazione ---
        model.eval() # Metti il modello in modalità valutazione
        val_loss = 0
        with torch.no_grad():
            for blurred, sharp in val_loader:
                blurred = blurred.to(DEVICE)
                sharp = sharp.to(DEVICE)
                predictions = model(blurred)
                val_loss += loss_fn(predictions, sharp).item()
        
        model.train() # Riporta il modello in modalità training
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Loss di Training: {train_loss:.4f} | Loss di Validazione: {avg_val_loss:.4f}")
        
        # Salva il modello se la validation loss migliora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("=> Modello salvato!")

if __name__ == "__main__":
    main()