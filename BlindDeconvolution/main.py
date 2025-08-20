import os
import argparse
import sys
import cv2
import numpy as np
import utils

from deconvolution import deblurShan, deblurFergus

def saveResults(deblurred_image, estimated_kernel, base_filename, method):
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Normalizza il kernel per una migliore visualizzazione prima di salvare
    kernel_norm = estimated_kernel / estimated_kernel.max() if estimated_kernel.max() > 0 else estimated_kernel
    
    output_path = f"results/{base_filename}_{method}_deblurred.png"
    kernel_path = f"results/{base_filename}_{method}_kernel.png"
    
    deblurred_to_save = (np.clip(deblurred_image, 0, 1) * 255).astype(np.uint8)
    kernel_to_save = (np.clip(kernel_norm, 0, 1) * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, deblurred_to_save)
    cv2.imwrite(kernel_path, kernel_to_save)
    print(f"Risultati salvati in '{output_path}' e '{kernel_path}'")

def main():
    parser = argparse.ArgumentParser(description="Esegue la Blind Deconvolution su un'immagine.")
    
    parser.add_argument('--method', type=str, default='shan', choices=['shan', 'fergus'],
                        help="L'algoritmo da utilizzare ('shan' o 'fergus').")
    parser.add_argument('--image', type=str, required=True,
                        help="Percorso dell'immagine sfocata da processare.")
    parser.add_argument('--kernel_size', type=int, default=21,
                        help="Dimensione del kernel da stimare (deve essere dispari).")
    parser.add_argument('--iterations', type=int, default=20,
                        help="Numero di iterazioni per l'algoritmo.")
    parser.add_argument('--synthetic', action='store_true', 
                        help="Se presente, esegue un test sintetico su un'immagine nitida.")
    parser.add_argument('--noise', type=float, default=0.01,
                        help="Livello di rumore stimato (usato da Fergus).")
    parser.add_argument('--lambda_prior', type=float, default=0.002,
                        help="Peso del prior sul gradiente per l'algoritmo di Shan.")
    
    args = parser.parse_args()

    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        print("Errore: kernel_size deve essere un numero dispari e positivo.")
        sys.exit(1)

    print(f"Caricamento immagine da: {args.image}")
    
    sharp_image = utils.loadImage(args.image, grayscale=True, normalize=True)
    if sharp_image is None:
        print(f"Errore: Impossibile caricare l'immagine da {args.image}. Controlla il percorso.")
        sys.exit(1)

    if args.synthetic:
        print("--- ESECUZIONE IN MODALITA' SINTETICA ---")
        blurred_image, gt_kernel = utils.createSyntheticBlur(sharp_image)
        ground_truth_image = sharp_image
    else:
        blurred_image = sharp_image
        ground_truth_image = None

    input_sharpness = utils.calculateSharpness(blurred_image)
    print(f"Nitidezza Immagine Input: {input_sharpness:.2f}")

    try:
        if args.method == 'shan':
            print("Esecuzione con l'algoritmo di Shan...")
            deblurred_image, estimated_kernel = deblurShan(
                blurred_image,
                args.kernel_size,
                num_iterations=args.iterations,
                lambda_prior=args.lambda_prior
            )
        elif args.method == 'fergus':
            print("Esecuzione con l'algoritmo di Fergus...")
            deblurred_image, estimated_kernel = deblurFergus(
                blurred_image,
                args.kernel_size,
                num_iterations=args.iterations,
                noise_level=args.noise # Iperparametro di Fergus
            )
        else:
            print(f"Metodo '{args.method}' non riconosciuto.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Errore durante l'esecuzione dell'algoritmo: {e}")
        sys.exit(1)

    # Valutazione e Risultati
    output_sharpness = utils.calculateSharpness(deblurred_image)
    print(f"Nitidezza Immagine Output: {output_sharpness:.2f}")

    if args.synthetic and ground_truth_image is not None:
        utils.calculateMetrics(ground_truth_image, deblurred_image)

    utils.showResults(blurred_image, estimated_kernel, deblurred_image)
    base_filename = os.path.splitext(os.path.basename(args.image))[0]
    saveResults(deblurred_image, estimated_kernel, base_filename, args.method)

if __name__ == "__main__":
    main()