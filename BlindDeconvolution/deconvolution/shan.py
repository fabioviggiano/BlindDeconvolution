# shan.py (Versione 2.0)
import numpy as np
import cv2
import utils

def deblurShan(blurred_image, kernel_size, num_iterations, lambda_prior, initial_kernel):
    """
    Funzione operaia a singola scala. Ora accetta un kernel iniziale.
    """
    kernel = initial_kernel.copy()
    
    latent_image = blurred_image.copy()
    B_fft = np.fft.fft2(blurred_image)
    dx = np.array([[-1, 1]]); dy = np.array([[-1], [1]])
    Dx_otf = utils.psf2otf(dx, blurred_image.shape)
    Dy_otf = utils.psf2otf(dy, blurred_image.shape)
    lambda_kernel = 1e-3

    for i in range(num_iterations):
        K_otf = utils.psf2otf(kernel, blurred_image.shape)
        prior_term = lambda_prior * (np.conj(Dx_otf) * Dx_otf + np.conj(Dy_otf) * Dy_otf)
        numerator = np.conj(K_otf) * B_fft
        denominator = np.conj(K_otf) * K_otf + prior_term
        L_fft = numerator / denominator
        latent_image = np.real(np.fft.ifft2(L_fft))
        
        L_otf = utils.psf2otf(latent_image, blurred_image.shape)
        numerator_grad_K = np.conj(L_otf) * B_fft
        denominator_grad_K = np.conj(L_otf) * L_otf + lambda_kernel
        K_fft_est = numerator_grad_K / denominator_grad_K
        kernel_est = np.real(np.fft.ifft2(K_fft_est))
        
        k_h, k_w = kernel.shape
        center_y, center_x = kernel_est.shape[0] // 2, kernel_est.shape[1] // 2
        
        kernel = kernel_est[center_y - k_h//2 : center_y + k_h//2 + 1, center_x - k_w//2 : center_x + k_w//2 + 1]
        kernel[kernel < 0] = 0
        kernel_sum = kernel.sum()
        if kernel_sum > 1e-6: kernel /= kernel_sum

    return latent_image, kernel

def deblurShanPyramidal(blurred_image, kernel_size, num_iterations=15, lambda_prior=5e-3, num_levels=4):
    """
    Versione piramidale corretta con scaling del kernel.
    """
    print("Inizio deconvolution con l'algoritmo di Shan (Piramidale v2.0)...")

    pyramid = [blurred_image]
    for i in range(num_levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    pyramid.reverse()

    # Inizializziamo il kernel al livello piu' basso
    k = np.zeros((kernel_size, kernel_size)); k[kernel_size//2, kernel_size//2] = 1.0

    for level, image_level in enumerate(pyramid):
        print(f"--- Processando livello piramidale {level+1}/{num_levels} (Dimensioni: {image_level.shape}) ---")
        
        # Eseguiamo l'algoritmo a questa scala, partendo dal kernel precedente
        _, k = deblurShan(image_level, k.shape[0], num_iterations, lambda_prior, k)

        # Se non siamo all'ultimo livello, ingrandiamo il kernel per il prossimo
        if level < num_levels - 1:
            # --- MODIFICA CRUCIALE: SCALING INTELLIGENTE ---
            # Moltiplichiamo il kernel per 2 (perche' pyrUp raddoppia le dimensioni)
            # e poi lo ritagliamo alla dimensione massima originale.
            k_up = cv2.pyrUp(k) * 4 # Moltiplichiamo per 4 per conservare l'energia
            
            new_h, new_w = k_up.shape
            center_y, center_x = new_h // 2, new_w // 2
            
            # Ritagliamo per assicurarci che non superi la dimensione massima
            crop_h, crop_w = kernel_size // 2, kernel_size // 2
            k = k_up[center_y - crop_h : center_y + crop_h + 1,
                     center_x - crop_w : center_x + crop_w + 1]
            
            if k.sum() > 0: k /= k.sum()

    print("Stima finale del kernel completata. Eseguo deconvolution finale...")
    
    # Usiamo la funzione operaia per la deconvolution finale non-blind
    final_image, final_kernel = deblurShan(blurred_image, k.shape[0], num_iterations, lambda_prior, k)

    return final_image, final_kernel