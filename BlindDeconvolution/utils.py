import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def loadImage(path, grayscale=True, normalize=True):
    """
    Carica un'immagine dal percorso specificato.
    
    Args:
        path (str): Percorso del file immagine.
        grayscale (bool): Se True, converte l'immagine in scala di grigi.
        normalize (bool): Se True, normalizza i valori dei pixel nell'intervallo [0, 1].
        
    Returns:
        np.ndarray: L'immagine caricata come array NumPy.
    """
    try:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Impossibile trovare l'immagine al percorso: {path}")
            
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if normalize:
            img = img.astype(np.float32) / 255.0
            
        return img
    except Exception as e:
        print(f"Errore durante il caricamento dell'immagine: {e}")
        return None

def psf2otf(psf, output_shape):
    """
    Converte una Point Spread Function (PSF, il nostro kernel) 
    in una Optical Transfer Function (OTF) della dimensione richiesta.
    
    Questo e' fondamentale per eseguire la convoluzione nel dominio della frequenza,
    che e' molto piu' veloce di quella nel dominio spaziale.
    
    Args:
        psf (np.ndarray): Il kernel di blur.
        output_shape (tuple): La dimensione dell'immagine target.
        
    Returns:
        np.ndarray: L'OTF (complessa) pronta per la moltiplicazione.
    """
    # 1. Calcola le dimensioni di padding
    psf_shape = psf.shape
    pad_height = (output_shape[0] - psf_shape[0])
    pad_width = (output_shape[1] - psf_shape[1])
    
    # 2. Esegui il padding
    # Mettiamo il kernel nell'angolo in alto a sinistra di un array di zeri
    padded_psf = np.pad(psf, 
                        ((0, pad_height), (0, pad_width)), 
                        'constant')
    
    # 3. Sposta il centro del kernel
    # Per la FFT, il punto (0,0) deve rappresentare il centro del kernel
    padded_psf = np.roll(padded_psf, -int(psf_shape[0] // 2), axis=0)
    padded_psf = np.roll(padded_psf, -int(psf_shape[1] // 2), axis=1)
    
    # 4. Calcola la FFT
    return np.fft.fft2(padded_psf)

def showResults(original, kernel, deblurred):
    """Visualizza i risultati."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Immagine Sfocata (Input)")
    axes[0].axis('off')
    
    # Normalizza il kernel per la visualizzazione
    kernel_display = kernel / kernel.max()
    axes[1].imshow(kernel_display, cmap='gray')
    axes[1].set_title("Kernel Stimato")
    axes[1].axis('off')
    
    axes[2].imshow(deblurred, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Immagine Ricostruita (Output)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def createSyntheticBlur(image, kernel_size=35, motion_angle=45, motion_len=50):
    """
    Crea un'immagine sfocata artificialmente a partire da una nitida.
    Restituisce l'immagine sfocata e il kernel ground truth.
    """
    # Crea il kernel di motion blur
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Disegna una linea nel kernel (algoritmo di Bresenham semplificato)
    dx = np.cos(np.deg2rad(motion_angle))
    dy = np.sin(np.deg2rad(motion_angle))
    
    for i in range(motion_len):
        x = int(center + i * dx)
        y = int(center + i * dy)
        if x >= 0 and x < kernel_size and y >= 0 and y < kernel_size:
            kernel[y, x] = 1
            
    kernel /= kernel.sum() # Normalizza il kernel
    
    # Applica il blur tramite convoluzione
    blurred_image = cv2.filter2D(image, -1, kernel)
    
    return blurred_image, kernel

def calculateMetrics(ground_truth, reconstructed):
    """
    Calcola e stampa PSNR e SSIM.
    Le immagini devono essere in float [0, 1].
    """
    psnr = peak_signal_noise_ratio(ground_truth, reconstructed, data_range=1.0)
    ssim = structural_similarity(ground_truth, reconstructed, data_range=1.0, channel_axis=None)
    
    print(f"Metrica di Valutazione:")
    print(f"  - PSNR: {psnr:.2f} dB")
    print(f"  - SSIM: {ssim:.4f}")
    return psnr, ssim

def calculateSharpness(image):
    """
    Calcola la nitidezza usando la varianza del Laplaciano.
    """
    return cv2.Laplacian(image, -1).var()