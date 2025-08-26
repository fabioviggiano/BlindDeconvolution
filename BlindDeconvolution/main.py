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

    # Evita di aggiungere handler duplicati se la funzione viene chiamata più volte
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

def saveResults(deblurred_image, estimated_kernel, base_filename, method):
    if not os.path.exists('results'):
        os.makedirs('results')    

    kernel_norm = estimated_kernel / estimated_kernel.max() if estimated_kernel.max() > 0 else estimated_kernel
    
    # Crea un nome file univoco basato sui parametri, se necessario, o usa un timestamp
    # Per ora, manteniamo la versione semplice
    output_path = f"results/{base_filename}_{method}_deblurred.png"
    kernel_path = f"results/{base_filename}_{method}_kernel.png"
    
    deblurred_to_save = (np.clip(deblurred_image, 0, 1) * 255).astype(np.uint8)
    kernel_to_save = (np.clip(kernel_norm, 0, 1) * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, deblurred_to_save)
    cv2.imwrite(kernel_path, kernel_to_save)

    return output_path, kernel_path

def saveMetadata(base_filename, method, args, input_sharpness, output_sharpness, exec_time, metrics=None):
    params = {k: v for k, v in vars(args).items()}

    metadata = {
        "method": method,
        "parameters": params,
        "input_sharpness": float(input_sharpness),
        "output_sharpness": float(output_sharpness),
        "execution_time_sec": float(exec_time),
    }
    if metrics:
        metadata["metrics"] = metrics

    # Crea un nome di file univoco per evitare sovrascritture durante i test
    timestamp = int(time.time())
    json_path = f"results/{base_filename}_{method}_{timestamp}_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return json_path

def main():
    parser = argparse.ArgumentParser(
        description="Esegue la Blind Deconvolution su un'immagine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Mostra i valori di default nell'help
    )
    
    # --- Parametri Principali ---
    g_main = parser.add_argument_group('Parametri Principali')
    g_main.add_argument('--image', type=str, required=True,
                        help="Percorso dell'immagine da processare.")
    g_main.add_argument('--method', type=str, default='shan', choices=['shan', 'fergus'],
                        help="Algoritmo da utilizzare.")
    g_main.add_argument('--kernel_size', type=int, default=35,
                        help="Dimensione del kernel da stimare (deve essere dispari).")
    g_main.add_argument('--iterations', type=int, default=20,
                        help="Numero di iterazioni per l'algoritmo.")
    
    # --- Parametri specifici per Algoritmo ---
    g_alg = parser.add_argument_group('Parametri specifici per Algoritmo')
    g_alg.add_argument('--lambda_prior', type=float, default=0.005,
                       help="[Shan] Peso del prior sul gradiente.")
    g_alg.add_argument('--lambda_kernel', type=float, default=0.001,
                       help="[Shan] Peso del prior di regolarizzazione per la stima del kernel.")
    g_alg.add_argument('--pyramid_levels', type=int, default=3,
                       help="[Shan] Numero di livelli della piramide.")
    g_alg.add_argument('--noise', type=float, default=0.01,
                       help="[Fergus] Livello di rumore stimato.")

    # --- NUOVO: Parametri per Test Sintetici ---
    g_synth = parser.add_argument_group('Parametri per Test Sintetici (usati solo con --synthetic)')
    g_synth.add_argument('--synthetic', action='store_true', 
                         help="Attiva la modalità test sintetico su un'immagine nitida.")
    g_synth.add_argument('--blur_type', type=str, default='motion', choices=['motion', 'gaussian'],
                         help="Tipo di blur da generare sinteticamente.")
    g_synth.add_argument('--motion_len', type=int, default=50,
                         help="[Motion Blur] Lunghezza della scia di movimento.")
    g_synth.add_argument('--motion_angle', type=float, default=45.0,
                         help="[Motion Blur] Angolo della scia di movimento (in gradi).")
    g_synth.add_argument('--gaussian_sigma', type=float, default=4.0,
                         help="[Gaussian Blur] Deviazione standard della gaussiana.")

    # --- Parametri di Output ---
    g_out = parser.add_argument_group('Parametri di Output e Visualizzazione')
    g_out.add_argument('--no_show', action='store_true',
                       help="Se presente, non mostra i risultati a video.")

    args = parser.parse_args()

    # Creiamo un nome base univoco per i file di output
    base_filename = f"{os.path.splitext(os.path.basename(args.image))[0]}_{args.method}"
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
        logger.info(f"Generazione blur di tipo '{args.blur_type}' con kernel_size={args.kernel_size}")
        
        # --- MODIFICATO: Chiamata alla funzione createSyntheticBlur aggiornata ---
        # Ora passiamo tutti i parametri dalla riga di comando a utils.py
        blurred_image, gt_kernel = utils.createSyntheticBlur(
            input_img,
            kernel_size=args.kernel_size,
            blur_type=args.blur_type,
            motion_angle=args.motion_angle,
            motion_len=args.motion_len,
            gaussian_sigma=args.gaussian_sigma
        )
        ground_truth_image = input_img
    else:
        blurred_image = input_img
        ground_truth_image = None

    input_sharpness = utils.calculateSharpness(blurred_image)
    logger.info(f"Nitidezza Immagine Input: {input_sharpness:.4f}")

    try:
        start_time = time.time()

        if args.method == 'shan':
            logger.info("Esecuzione con l'algoritmo di Shan...")
            deblurred_image, estimated_kernel = deblurShanPyramidal(
                blurred_image,
                args.kernel_size,
                num_iterations=args.iterations,
                lambda_prior=args.lambda_prior,
                lambda_kernel_reg=args.lambda_kernel,
                num_levels=args.pyramid_levels   
            )
        elif args.method == 'fergus':
            # Nota: deblurFergus non è ancora implementato, questo è un placeholder
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
    logger.info(f"Nitidezza Immagine Output: {output_sharpness:.4f}")
    logger.info(f"Tempo di esecuzione: {exec_time:.2f} secondi")

    metrics = None
    if args.synthetic and ground_truth_image is not None:
        metrics = utils.calculateMetrics(ground_truth_image, deblurred_image)
        # Il logging delle metriche ora è gestito dentro calculateMetrics

    if not args.no_show:
        utils.showResults(blurred_image, estimated_kernel, deblurred_image)

    output_path, kernel_path = saveResults(deblurred_image, estimated_kernel, base_filename, args.method)
    logger.info(f"Risultati salvati: immagine {output_path}, kernel {kernel_path}")
    
    # Passiamo un nome file più specifico per i metadati
    metadata_filename_base = f"{os.path.splitext(os.path.basename(args.image))[0]}"
    metadata_path = saveMetadata(metadata_filename_base, args.method, args, input_sharpness, output_sharpness, exec_time, metrics)
    logger.info(f"Metadati salvati in {metadata_path}")


if __name__ == "__main__":
    main()