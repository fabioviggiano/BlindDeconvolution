import numpy as np
import utils

def deblurShan(blurred_image, kernel_size, num_iterations=15, lambda_prior=5e-3):
    """
    Implementazione dell'algoritmo di Blind Deconvolution di Shan et al. (2008).
    
    Args:

        blurred_image (np.ndarray): Immagine di input sfocata (normalizzata a [0,1]).
        kernel_size (int): Dimensione del lato del kernel di blur da stimare (deve essere dispari).
        num_iterations (int): Numero di iterazioni alternate.
        lambda_prior (float): Peso del prior sul gradiente dell'immagine.
    
    Returns:
        tuple: (immagine_deblurrata, kernel_stimato)
    """
    if kernel_size % 2 == 0:
        raise ValueError("La dimensione del kernel deve essere dispari.")

    print("Inizio deconvolution con l'algoritmo di Shan...")

    # --- 1. Inizializzazione ---
    # La migliore stima iniziale per l'immagine latente è l'immagine sfocata stessa
    latent_image = blurred_image.copy() 
    
    # Inizializziamo il kernel come una "funzione delta": un singolo pixel al centro.
    # Questo rappresenta l'ipotesi iniziale di "nessun blur".
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1.0

    # Trasformate di Fourier dell'immagine sfocata (costante durante il processo)
    B_fft = np.fft.fft2(blurred_image)
    
    # Definiamo i filtri per calcolare i gradienti (derivate parziali x e y)
    dx = np.array([[-1, 1]])
    dy = np.array([[-1], [1]])
    
    # Calcoliamo le loro OTF per usarle nel dominio della frequenza
    Dx_otf = utils.psf2otf(dx, blurred_image.shape)
    Dy_otf = utils.psf2otf(dy, blurred_image.shape)

    # --- 2. Ciclo di Ottimizzazione Alternata ---
    for i in range(num_iterations):
        print(f"Iterazione {i+1}/{num_iterations}...")
        
        # --- a) Stima dell'Immagine Latente (L) con K fisso ---
        # Questo è un problema di deconvolution NON-blind.
        # La funzione di costo è: ||L * K - B||² + lambda * ||grad(L)||²
        # La soluzione può essere trovata efficientemente nel dominio della frequenza.
        
        K_otf = utils.psf2otf(kernel, blurred_image.shape)
        
        # Termine del prior (regolarizzazione) nel dominio della frequenza
        prior_term = lambda_prior * (np.conj(Dx_otf) * Dx_otf + np.conj(Dy_otf) * Dy_otf)
        
        # Risolviamo per L nel dominio della frequenza
        numerator = np.conj(K_otf) * B_fft
        denominator = np.conj(K_otf) * K_otf + prior_term
        
        L_fft = numerator / denominator
        
        # Torniamo al dominio spaziale
        latent_image = np.real(np.fft.ifft2(L_fft))
        
        
        # --- b) Stima del Kernel (K) con L fisso ---
        # La funzione di costo è: ||L * K - B||²
        # Questo si risolve con un metodo basato su gradiente.
        
        L_otf = utils.psf2otf(latent_image, blurred_image.shape)
        
        # Calcolo del gradiente nel dominio della frequenza
        numerator_grad_K = np.conj(L_otf) * B_fft
        denominator_grad_K = np.conj(L_otf) * L_otf
        
        # Gradiente della funzione di costo rispetto a K
        # grad = ifft(L_conj * (L_otf * K_otf - B_fft))
        # Per semplicità, usiamo una soluzione diretta simile a quella per L
        K_fft_est = numerator_grad_K / denominator_grad_K
        kernel_est = np.real(np.fft.ifft2(K_fft_est))
        
        # Estraiamo la porzione centrale e applichiamo i vincoli
        k_h, k_w = kernel.shape
        center_y, center_x = kernel_est.shape[0] // 2, kernel_est.shape[1] // 2
        
        # Ritagliamo la stima del kernel dalle sue "code" prodotte dalla FFT
        kernel = kernel_est[
            center_y - k_h//2 : center_y + k_h//2 + 1,
            center_x - k_w//2 : center_x + k_w//2 + 1
        ]
        
        # Applichiamo i vincoli fondamentali sul kernel:
        # 1. Non-negatività: i valori non possono essere negativi
        kernel[kernel < 0] = 0
        # 2. Conservazione dell'energia: la somma dei suoi elementi deve essere 1
        kernel_sum = kernel.sum()
        if kernel_sum > 1e-6: # Evita divisione per zero
            kernel /= kernel_sum

    print("Deconvolution completata.")
    return latent_image, kernel