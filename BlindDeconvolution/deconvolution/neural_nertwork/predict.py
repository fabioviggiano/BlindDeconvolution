# coding: utf-8

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from model import UNet
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils

# --- Parametri ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = 'unet_deblur_model.pth' # Il modello salvato da train.py
TEST_IMAGE_DIR = '../../data_processed/test/blurred'
GROUND_TRUTH_DIR = '../../data_processed/test/sharp'
OUTPUT_DIR = '../../results_nn'

def deblur_image(model, image_path):
    # Trasformazioni per l'input (le stesse usate nel training)
    IMAGE_SIZE = 256

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    ])

    # Carica l'immagine
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    input_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE) # Aggiungi la dimensione del batch

    # Metti il modello in modalità valutazione
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Riconverti l'output in un'immagine OpenCV
    output_image = output_tensor.squeeze(0).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0)) # Da (C, H, W) a (H, W, C)
    output_image = np.clip(output_image, 0, 1) # Assicura che i valori siano in [0, 1]
    
    # Riconverti in BGR per il salvataggio con OpenCV
    output_image_bgr = cv2.cvtColor((output_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return output_image, output_image_bgr


def main():
    # Crea la cartella di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Carica il modello addestrato
    model = UNet(in_channels=3, out_channels=3).to(DEVICE)
    print("Caricamento checkpoint da:", MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Modello caricato con successo.")

    # 2. Itera sulle immagini di test 
    image_paths = []
    for root, _, files in os.walk(TEST_IMAGE_DIR):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))

    print(f"\nTrovate {len(image_paths)} immagini nel test set. Inizio la valutazione...")
    
    all_metrics = []

    for blurred_path in image_paths:
        filename = os.path.basename(blurred_path)
        print(f"Processando {filename}...")
        
        # Deblur
        deblurred_img_norm, deblurred_img_bgr = deblur_image(model, blurred_path)
        
        # Salva l'output
        output_filename = f"{os.path.splitext(filename)[0]}_deblurred.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, output_filename), deblurred_img_bgr)

        # Calcola le metriche
        # Ricostruisci il percorso del ground truth basandoti sul percorso relativo
        relative_path = os.path.relpath(blurred_path, TEST_IMAGE_DIR)
        sharp_path = os.path.join(GROUND_TRUTH_DIR, relative_path)
        
        if os.path.exists(sharp_path):
            ground_truth = utils.loadImage(sharp_path, grayscale=False, normalize=True)
            # Assicurati che le dimensioni corrispondano per il calcolo delle metriche
            h, w, _ = deblurred_img_norm.shape
            ground_truth = cv2.resize(ground_truth, (w, h))

            metrics = utils.calculateMetrics(ground_truth, deblurred_img_norm)
            metrics['filename'] = filename
            all_metrics.append(metrics)
            print(f"  -> PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
            
    # Potresti salvare queste metriche in un nuovo file CSV per un facile confronto
    print("\nValutazione completata.")
    # Esempio: print(all_metrics)

if __name__ == "__main__":
    main()