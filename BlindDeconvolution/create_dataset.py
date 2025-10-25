# coding: utf-8
# Script per generare le coppie (blurred, sharp)

import os
import cv2
import random
import numpy as np
import argparse
import shutil
from tqdm import tqdm
import utils

def create_blur_dataset(source_base_dir, output_base_dir, num_versions=1):
    """
    Scansiona le cartelle di input, genera immagini sfocate e le salva
    nella struttura di output corretta.

    Args:
        source_base_dir (str): Percorso alla cartella 'data' che contiene 'train' e 'test'.
        output_base_dir (str): Percorso dove salvare il nuovo dataset (es. 'data_processed').
        num_versions (int): Quante versioni sfocate generare per ogni immagine originale.
    """
    print(f"Inizio la generazione del dataset da '{source_base_dir}' a '{output_base_dir}'...")
    
    # Definiamo i range dei parametri per creare un blur variegato
    BLUR_PARAMS = {
        'motion': {
            'motion_len': (10, 40),      # Lunghezza del movimento
            'motion_angle': (0, 180),   # Angolo del movimento
            'kernel_size': (25, 45)     # Dimensione del kernel (deve essere dispari)
        },
        'gaussian': {
            'gaussian_sigma': (1.0, 4.0), # Deviazione standard
            'kernel_size': (15, 35)      # Dimensione del kernel (deve essere dispari)
        }
    }

    # Processiamo sia il set di training che quello di test
    for dataset_type in ['train', 'test']:
        source_dir = os.path.join(source_base_dir, dataset_type)
        if not os.path.isdir(source_dir):
            print(f"Attenzione: La cartella sorgente '{source_dir}' non esiste. Salto...")
            continue

        print(f"\n--- Processando il set: {dataset_type} ---")
        
        # Trova tutte le immagini .png nelle sottocartelle
        image_paths = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    image_paths.append(os.path.join(root, file))

        # Creazione delle cartelle di output
        output_sharp_dir = os.path.join(output_base_dir, dataset_type, 'sharp')
        output_blurred_dir = os.path.join(output_base_dir, dataset_type, 'blurred')
        os.makedirs(output_sharp_dir, exist_ok=True)
        os.makedirs(output_blurred_dir, exist_ok=True)

        # Itera su ogni immagine con una barra di progresso (tqdm)
        for img_path in tqdm(image_paths, desc=f"Generando {dataset_type} set"):
            # Carica l'immagine originale (nitida)
            sharp_image = utils.loadImage(img_path, grayscale=True, normalize=True)
            if sharp_image is None:
                continue
            
            # Genera N versioni sfocate per ogni immagine nitida
            for i in range(num_versions):
                # Scegli a caso il tipo di blur
                blur_type = random.choice(['motion', 'gaussian'])
                params = BLUR_PARAMS[blur_type]

                # Estrai parametri casuali dai range definiti
                k_size = random.randrange(params['kernel_size'][0], params['kernel_size'][1] + 1, 2) # Assicura dispari
                
                motion_len = 0
                motion_angle = 0
                gaussian_sigma = 0

                if blur_type == 'motion':
                    motion_len = random.randint(*params['motion_len'])
                    motion_angle = random.uniform(*params['motion_angle'])
                else: # gaussian
                    gaussian_sigma = random.uniform(*params['gaussian_sigma'])

                # Crea l'immagine sfocata usando la funzione da utils.py
                blurred_image, _ = utils.createSyntheticBlur(
                    sharp_image,
                    kernel_size=k_size,
                    blur_type=blur_type,
                    motion_len=motion_len,
                    motion_angle=motion_angle,
                    gaussian_sigma=gaussian_sigma
                )
                
                # Costruisce il nome del file di output per evitare sovrascritture
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                sub_folder_name = os.path.basename(os.path.dirname(img_path))
                
                # Usiamo una struttura di sottocartelle per mantenere l'ordine originale
                final_sharp_dir = os.path.join(output_sharp_dir, sub_folder_name)
                final_blurred_dir = os.path.join(output_blurred_dir, sub_folder_name)
                os.makedirs(final_sharp_dir, exist_ok=True)
                os.makedirs(final_blurred_dir, exist_ok=True)

                output_filename = f"{base_filename}_v{i}.png"
                
                # Salva l'immagine sfocata e la sua corrispondente versione nitida
                # Le immagini devono essere convertite da float [0,1] a uint8 [0,255]
                cv2.imwrite(os.path.join(final_blurred_dir, output_filename), (blurred_image * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(final_sharp_dir, output_filename), (sharp_image * 255).astype(np.uint8))

    print("\nGenerazione del dataset completata.")


def split_train_val(base_dir, val_split_ratio=0.2):

    # Suddivide il set di training generato in training e validazione.

    # Args:
    #   base_dir (str): La cartella di output dove è stato generato il dataset.
    #   val_split_ratio (float): La percentuale di dati da spostare nel set di validazione.

    print(f"\nInizio la suddivisione in training/validation (ratio: {val_split_ratio})...")
    
    train_sharp_dir = os.path.join(base_dir, 'train', 'sharp')
    train_blurred_dir = os.path.join(base_dir, 'train', 'blurred')
    
    # Se le cartelle non esistono, non fare nulla
    if not os.path.isdir(train_sharp_dir):
        print("Cartella di training non trovata. Impossibile creare il set di validazione.")
        return

    # Crea le cartelle per il set di validazione
    val_sharp_dir = os.path.join(base_dir, 'val', 'sharp')
    val_blurred_dir = os.path.join(base_dir, 'val', 'blurred')
    os.makedirs(val_sharp_dir, exist_ok=True)
    os.makedirs(val_blurred_dir, exist_ok=True)

    # Prendi la lista di tutte le immagini e mescolala
    # Usiamo le immagini 'sharp' come riferimento
    all_images = []
    for root, _, files in os.walk(train_sharp_dir):
        for file in files:
            all_images.append(os.path.join(root, file))

    random.shuffle(all_images)
    
    # Calcola quanti file spostare
    num_val_files = int(len(all_images) * val_split_ratio)
    val_files = all_images[:num_val_files]

    print(f"Spostamento di {num_val_files} immagini nel set di validazione...")
    
    # Sposta i file
    for sharp_path in tqdm(val_files, desc="Spostando in validation"):
        # Ricava il percorso relativo (es. 'C002/0_v0.png')
        relative_path = os.path.relpath(sharp_path, train_sharp_dir)
        
        # Costruisci il percorso del file blurred corrispondente
        blurred_path = os.path.join(train_blurred_dir, relative_path)
        
        # Definisci i percorsi di destinazione completi
        dest_sharp_path = os.path.join(val_sharp_dir, relative_path)
        dest_blurred_path = os.path.join(val_blurred_dir, relative_path)
        
        # --- MODIFICA CHIAVE ---
        # Crea le sottocartelle di destinazione se non esistono
        os.makedirs(os.path.dirname(dest_sharp_path), exist_ok=True)
        os.makedirs(os.path.dirname(dest_blurred_path), exist_ok=True)
        # --- FINE MODIFICA ---
        
        # Sposta i file nei nuovi percorsi
        shutil.move(sharp_path, dest_sharp_path)
        shutil.move(blurred_path, dest_blurred_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera un dataset di immagini sfocate per il training di reti neurali.")
    parser.add_argument('--source', type=str, default='data', help="Cartella sorgente contenente le sottocartelle 'train' e 'test' con le immagini nitide.")
    parser.add_argument('--output', type=str, default='data_processed', help="Cartella di destinazione per il nuovo dataset.")
    parser.add_argument('--versions', type=int, default=1, help="Quante versioni sfocate generare per ogni immagine originale.")
    parser.add_argument('--val_split', type=float, default=0.2, help="Percentuale del set di training da usare come set di validazione (es. 0.2 per il 20%).")
    
    args = parser.parse_args()

    # Esegui la creazione del dataset
    create_blur_dataset(args.source, args.output, args.versions)
    
    # Esegui la suddivisione in training/validation
    if args.val_split > 0:
        split_train_val(args.output, args.val_split)
        
    print("\nProcesso terminato con successo!")