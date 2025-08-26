# fergus.py
import numpy as np
import utils 

def deblurFergus(blurred_image, kernel_size, num_iterations=15, noise_level=0.01):
    """
    Implementazione (semplificata) dell'algoritmo di Fergus et al. (2006).
    """
    if kernel_size % 2 == 0:
        raise ValueError("La dimensione del kernel deve essere dispari.")

    print("Inizio deconvolution con l'algoritmo di Fergus...")

    latent_image = blurred_image.copy()
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1.0

    gamma = 1.0 / (noise_level**2) if noise_level > 0 else 1e10

    dx = np.array([[-1, 1]]); dy = np.array([[-1], [1]])
    Dx_otf = utils.psf2otf(dx, blurred_image.shape)
    Dy_otf = utils.psf2otf(dy, blurred_image.shape)

    B_fft = np.fft.fft2(blurred_image)

    for i in range(num_iterations):
        print(f"Iterazione {i+1}/{num_iterations} (E-Step -> M-Step)...")
        
        # --- E-Step: Stima dell'Immagine Latente Q(L) (INVARIATO) ---
        K_otf = utils.psf2otf(kernel, blurred_image.shape)
        lambda_grad = 0.005 
        prior_term_L = lambda_grad * (np.conj(Dx_otf) * Dx_otf + np.conj(Dy_otf) * Dy_otf)
        numerator_L = np.conj(K_otf) * B_fft
        denominator_L = np.conj(K_otf) * K_otf + (1/gamma) * prior_term_L
        L_fft = numerator_L / denominator_L
        latent_image = np.real(np.fft.ifft2(L_fft))
        latent_image = np.clip(latent_image, 0, 1)

        # --- M-Step: Stima della distribuzione del Kernel Q(K) (MODIFICATO) ---
        L_otf = utils.psf2otf(latent_image, blurred_image.shape)
        numerator_K = np.conj(L_otf) * B_fft
        
        # --- MODIFICA CRUCIALE QUI ---
        # Aggiungiamo un piccolo valore costante per la stabilità
        denominator_K = np.conj(L_otf) * L_otf + 1e-4
        # ---------------------------

        K_fft_est = numerator_K / denominator_K
        kernel_est = np.real(np.fft.ifft2(K_fft_est))
        
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