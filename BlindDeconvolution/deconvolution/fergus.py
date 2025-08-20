# deconvolution/fergus.py

import numpy as np
# Assicuriamoci che il nostro file principale importi utils,
# quindi possiamo usarlo qui come import assoluto.
import utils 

def deblurFergus(blurred_image, kernel_size, num_iterations=15, noise_level=0.01):
    """
    Implementazione (semplificata) dell'algoritmo di Fergus et al. (2006)
    usando un approccio EM Variazionale con prior Gaussiano.
    
    Args:
        blurred_image (np.ndarray): Immagine di input sfocata.
        kernel_size (int): Dimensione del lato del kernel.
        num_iterations (int): Numero di cicli E-M.
        noise_level (float): Stima iniziale della varianza del rumore.
                             Questo e' un iperparametro CRUCIALE.
    
    Returns:
        tuple: (immagine_deblurrata, kernel_stimato)
    """
    if kernel_size % 2 == 0:
        raise ValueError("La dimensione del kernel deve essere dispari.")

    print("Inizio deconvolution con l'algoritmo di Fergus...")

    # --- 1. Inizializzazione ---
    # Inizializziamo l'immagine latente (la media della nostra distribuzione Q(L))
    latent_image = blurred_image.copy()
    
    # Inizializziamo il kernel (la media della nostra distribuzione Q(K))
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1.0

    # 'gamma' e' l'inverso della varianza del rumore. E' il peso del termine di fedelta' ai dati.
    gamma = 1.0 / (noise_level**2)

    # Pre-calcoliamo le OTF dei filtri gradiente (come in Shan)
    dx = np.array([[-1, 1]])
    dy = np.array([[-1], [1]])
    Dx_otf = utils.psf2otf(dx, blurred_image.shape)
    Dy_otf = utils.psf2otf(dy, blurred_image.shape)

    # FFT dell'immagine sfocata
    B_fft = np.fft.fft2(blurred_image)

    # --- 2. Ciclo di Ottimizzazione EM Variazionale ---
    for i in range(num_iterations):
        print(f"Iterazione {i+1}/{num_iterations} (E-Step -> M-Step)...")
        
        # --- E-Step: Stima della distribuzione dell'Immagine Latente Q(L) ---
        # Teniamo K fisso e aggiorniamo L.
        # La soluzione e' simile a un filtro di Wiener, dove il prior sui gradienti
        # agisce come regolarizzatore.
        
        K_otf = utils.psf2otf(kernel, blurred_image.shape)
        
        # Il prior sui gradienti nel dominio della frequenza.
        # Il paper di Fergus usa un prior complesso qui. Noi usiamo un prior L2 semplice.
        # Il peso di questo prior e' legato alla stima del rumore 'gamma'.
        # Per semplicita', usiamo un peso fisso o lo leghiamo a gamma.
        # Proviamo con un lambda fisso come in Shan per iniziare.
        lambda_grad = 0.005 
        
        prior_term_L = lambda_grad * (np.conj(Dx_otf) * Dx_otf + np.conj(Dy_otf) * Dy_otf)
        
        # Risolviamo per la media della distribuzione dell'immagine, E[L]
        numerator_L = np.conj(K_otf) * B_fft
        denominator_L = np.conj(K_otf) * K_otf + (1/gamma) * prior_term_L
        
        L_fft = numerator_L / denominator_L
        latent_image = np.real(np.fft.ifft2(L_fft))
        latent_image = np.clip(latent_image, 0, 1) # Applica un vincolo di realismo

        # --- M-Step: Stima della distribuzione del Kernel Q(K) ---
        # Teniamo L fisso e aggiorniamo K. E' molto simile allo step del kernel di Shan.
        
        L_otf = utils.psf2otf(latent_image, blurred_image.shape)

        # Risolviamo per la media della distribuzione del kernel, E[K]
        numerator_K = np.conj(L_otf) * B_fft
        denominator_K = np.conj(L_otf) * L_otf
        
        # Aggiungiamo un piccolo valore per la stabilita' numerica
        K_fft_est = numerator_K / (denominator_K + 1e-8)
        kernel_est = np.real(np.fft.ifft2(K_fft_est))
        
        # Applichiamo i vincoli sul kernel (positivita' e somma a 1)
        k_h, k_w = kernel.shape
        center_y, center_x = kernel_est.shape[0] // 2, kernel_est.shape[1] // 2
        
        kernel = kernel_est[
            center_y - k_h//2 : center_y + k_h//2 + 1,
            center_x - k_w//2 : center_x + k_w//2 + 1
        ]
        
        kernel[kernel < 0] = 0
        kernel_sum = kernel.sum()
        if kernel_sum > 1e-6:
            kernel /= kernel_sum
            
    print("Deconvolution (Fergus) completata.")
    return latent_image, kernel