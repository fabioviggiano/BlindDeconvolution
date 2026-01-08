## Setup e Installazione

1.  **Clonare il repository:**
    ```bash
    git clone https://github.com/fabioviggiano/BlindDeconvolution.git
    cd BlindDeconvolution
    ```

2.  **Installare le dipendenze:**
    Si consiglia di utilizzare un ambiente virtuale Python.
    ```bash
    pip install -r requirements.txt
    ```

## Utilizzo

### Esecuzione dell'Algoritmo di Shan (Model-Based)

Lo script `main.py` permette di eseguire esperimenti di deconvolution su immagini reali o di generare test sintetici.

**Esempio di test sintetico con motion blur:**
```bash
python main.py --image data/test/C081/15.png --synthetic --blur_type motion --motion_len 30 --kernel_size 35
```

**Esempio di test sintetico con blur gaussiano:**
```bash
python main.py --image data/test/C081/15.png --synthetic --blur_type gaussian --gaussian_sigma 2.5 --kernel_size 21
```

Per una lista completa dei parametri configurabili, eseguire:
```bash
python main.py --help
```
### Esecuzione dell'Approccio con Rete Neurale (U-Net)

L'utilizzo dell'approccio basato su deep learning è diviso in tre fasi: generazione del dataset, addestramento del modello e valutazione (inferenza).

#### Fase 1: Generazione del Dataset

Questo script deve essere eseguito una sola volta per creare il set di dati necessario all'addestramento e alla validazione.

1.  **Assicurarsi di essere nella cartella principale del progetto.**
2.  **Lanciare lo script:**
    ```bash
    python create_dataset.py
    ```
    Questo comando creerà una nuova cartella `data_processed/` con i set di `train`, `val` e `test`. Per generare più varianti di blur per ogni immagine (consigliato), usare l'argomento `--versions`:
    ```bash
    python create_dataset.py --versions 3
    ```

#### Fase 2: Addestramento del Modello

Questo processo è computazionalmente intensivo e serve per addestrare la U-Net.

1.  **Navigare nella cartella della rete neurale:**
    ```bash
    cd neural_network
    ```
2.  **Lanciare lo script di addestramento:**
    ```bash
    python train.py
    ```
    Lo script addestrerà il modello per 25 epoche e salverà automaticamente il modello migliore (in base alla performance sul set di validazione) nel file `unet_deblur_model.pth`. Se lo script viene interrotto e rilanciato, riprenderà l'addestramento dall'ultimo checkpoint salvato.

#### Fase 3: Valutazione del Modello (Inferenza)

Una volta che il modello è stato addestrato, questo script lo utilizza per processare le immagini del test set.

1.  **Assicurarsi di essere nella cartella della rete neurale:**
    ```bash
    cd neural_network
    ```
2.  **Lanciare lo script di valutazione:**
    ```bash
    python predict.py
    ```
    Lo script caricherà il modello `unet_deblur_model.pth`, processerà tutte le immagini presenti in `data_processed/test/`, salverà i risultati deblurrati nella cartella `results_nn/` e stamperà a schermo le metriche PSNR e SSIM per ogni immagine.
