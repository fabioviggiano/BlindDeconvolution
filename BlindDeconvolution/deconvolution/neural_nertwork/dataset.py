# coding: utf-8
# Per caricare i dati durante il training

import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeblurDataset(Dataset):
    """
    Dataset personalizzato per caricare le coppie di immagini (sfocata, nitida).
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): La cartella che contiene le sottocartelle 'sharp' e 'blurred'.
                            (es. 'data_processed/train' o 'data_processed/val')
            transform (callable, optional): Trasformazioni da applicare a entrambe le immagini.
        """
        self.root_dir = root_dir
        self.sharp_dir = os.path.join(root_dir, 'sharp')
        self.blurred_dir = os.path.join(root_dir, 'blurred')
        self.transform = transform

        # Crea una lista di tutti i percorsi delle immagini
        # Usiamo le immagini sfocate come riferimento per l'elenco
        self.image_files = []
        for root, _, files in os.walk(self.blurred_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        """
        Restituisce il numero totale di immagini nel dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Carica e restituisce una coppia di immagini (sfocata, nitida) all'indice idx.
        """
        # Percorso dell'immagine sfocata
        blurred_img_path = self.image_files[idx]

        # Ricava il percorso della corrispondente immagine nitida
        # Sostituendo la parte 'blurred' del percorso con 'sharp'
        relative_path = os.path.relpath(blurred_img_path, self.blurred_dir)
        sharp_img_path = os.path.join(self.sharp_dir, relative_path)

        # Carica le immagini con OpenCV (in BGR)
        blurred_image_bgr = cv2.imread(blurred_img_path)
        sharp_image_bgr = cv2.imread(sharp_img_path)
        
        # Converte da BGR a RGB, che è più standard per PyTorch
        blurred_image = cv2.cvtColor(blurred_image_bgr, cv2.COLOR_BGR2RGB)
        sharp_image = cv2.cvtColor(sharp_image_bgr, cv2.COLOR_BGR2RGB)

        # Applica le trasformazioni se definite
        if self.transform:
            blurred_image = self.transform(blurred_image)
            sharp_image = self.transform(sharp_image)

        return blurred_image, sharp_image

# Esempio di come si potrebbe usare questo file in train.py
if __name__ == '__main__':
    # Definiamo le trasformazioni: converti in Tensore e normalizza i pixel in [0, 1]
    transformations = transforms.Compose([
        transforms.ToTensor(), 
    ])
    
    # Crea un'istanza del dataset per il training
    train_dataset = DeblurDataset(root_dir='../data_processed/train', transform=transformations)
    
    # Stampa le dimensioni del dataset
    print(f"Trovate {len(train_dataset)} immagini nel set di training.")
    
    # Prende un campione e ne verifica le dimensioni
    blurred_sample, sharp_sample = train_dataset[0]
    print(f"Dimensioni del tensore dell'immagine sfocata: {blurred_sample.shape}")
    print(f"Dimensioni del tensore dell'immagine nitida: {sharp_sample.shape}")