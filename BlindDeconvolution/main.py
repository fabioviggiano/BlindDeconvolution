import os
import argparse
import sys
import cv2
import numpy as np
import utils
import logging
import json
import time

from deconvolution import deblurShanPyramidal, deblurFergus

def setup_logging(base_filename):
    log_dir = "results"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{base_filename}.log")

    logger = logging.getLogger(base_filename)
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(ch_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def saveResults(deblurred_image, estimated_kernel, base_filename, method):
    if not os.path.exists('results'):
        os.makedirs('results')
    
    kernel_norm = estimated_kernel / estimated_kernel.max() if estimated_kernel.max() > 0 else estimated_kernel
    
    output_path = f"results/{base_filename}_{method}_deblurred.png"
    kernel_path = f"results/{base_filename}_{method}_kernel.png"
    
    deblurred_to_save = (np.clip(deblurred_image, 0, 1) * 255).astype(np.uint8)
    kernel_to_save = (np.clip(kernel_norm, 0, 1) * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, deblurred_to_save)
    cv2.imwrite(kernel_path, kernel_to_save)

    return output_path, kernel_path

def saveMetadata(base_filename, method, args, input_sharpness, output_sharpness, exec_time, metrics=None):
    # Converte argparse.Namespace in un dict con tipi serializzabili
    params = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
              for k, v in vars(args).items()}

    metadata = {
        "method": method,
        "parameters": params,
        "input_sharpness": float(input_sharpness),
        "output_sharpness": float(output_sharpness),
        "execution_time_sec": float(exec_time),
    }
    if metrics:
        metadata["metrics"] = metrics

    json_path = f"results/{base_filename}_{method}_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return json_path

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
    parser.add_argument('--pyramid_levels', type=int, default=3,
                        help="Numero di livelli della piramide (usato da Shan).")
    parser.add_argument('--edgetaper', action='store_true',
                        help="Applica edgetapering all'immagine prima della deconvoluzione.")
    parser.add_argument('--no_show', action='store_true',
                        help="Se presente, non mostra i risultati a video.")
    
    args = parser.parse_args()

    base_filename = os.path.splitext(os.path.basename(args.image))[0]
    logger = setup_logging(base_filename)

    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        logger.error("kernel_size deve essere un numero dispari e positivo.")
        sys.exit(1)

    logger.info(f"Caricamento immagine da: {args.image}")
    
    input_img = utils.loadImage(args.image, grayscale=True, normalize=True)
    if input_img is None:
        logger.error(f"Impossibile caricare l'immagine da {args.image}. Controlla il percorso.")
        sys.exit(1)

    if args.synthetic:
        logger.info("--- Esecuzione in modalità sintetica ---")
        blurred_image, gt_kernel = utils.createSyntheticBlur(input_img)
        ground_truth_image = input_img
    else:
        blurred_image = input_img
        ground_truth_image = None

    input_sharpness = utils.calculateSharpness(blurred_image)
    logger.info(f"Nitidezza Immagine Input: {input_sharpness:.2f}")

    try:
        start_time = time.time()

        if args.method == 'shan':
            logger.info("Esecuzione con l'algoritmo di Shan...")
            deblurred_image, estimated_kernel = deblurShanPyramidal(
                blurred_image,
                args.kernel_size,
                num_iterations=args.iterations,
                lambda_prior=args.lambda_prior,
                num_levels=args.pyramid_levels   
            )
        elif args.method == 'fergus':
            logger.info("Esecuzione con l'algoritmo di Fergus...")
            deblurred_image, estimated_kernel = deblurFergus(
                blurred_image,
                args.kernel_size,
                num_iterations=args.iterations,
                noise_level=args.noise
            )
        else:
            logger.error(f"Metodo '{args.method}' non riconosciuto.")
            sys.exit(1)

        exec_time = time.time() - start_time

    except Exception as e:
        logger.exception(f"Errore durante l'esecuzione dell'algoritmo: {e}")
        sys.exit(1)

    output_sharpness = utils.calculateSharpness(deblurred_image)
    logger.info(f"Nitidezza Immagine Output: {output_sharpness:.2f}")
    logger.info(f"Tempo di esecuzione: {exec_time:.2f} secondi")

    metrics = None
    if args.synthetic and ground_truth_image is not None:
        metrics = utils.calculateMetrics(ground_truth_image, deblurred_image)
        if metrics:
            logger.info(f"Metriche sintetiche: {metrics}")

    if not args.no_show:
        utils.showResults(blurred_image, estimated_kernel, deblurred_image)

    output_path, kernel_path = saveResults(deblurred_image, estimated_kernel, base_filename, args.method)
    logger.info(f"Risultati salvati: immagine {output_path}, kernel {kernel_path}")

    metadata_path = saveMetadata(base_filename, args.method, args, input_sharpness, output_sharpness, exec_time, metrics)
    logger.info(f"Metadati salvati in {metadata_path}")

if __name__ == "__main__":
    main()
