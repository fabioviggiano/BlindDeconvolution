# Blind Deconvolution: Confronto tra Metodi Model-Based e Reti Neurali

Questo repository contiene il codice e i risultati per il progetto di Computational Imaging. L'obiettivo è implementare e confrontare due approcci distinti per la *blind deconvolution*: un metodo classico *model-based* e un approccio *end-to-end* basato su reti neurali.

Il problema affrontato è il recupero di un'immagine nitida a partire da una sua versione affetta da blur (gaussiano o da movimento) senza conoscere a priori le caratteristiche esatte del kernel di sfocatura.

## Struttura del Progetto

```
BlindDeconvolution/
├── deconvolution/      # Codice sorgente degli algoritmi model-based
│   ├── shan.py         # Implementazione dell'algoritmo di Shan (2008)
│   └── fergus.py       # (Placeholder per l'algoritmo di Fergus)
├── neural_network/     # (In sviluppo) Codice per l'approccio con U-Net
├── data/               # Immagini di input e dataset per il training
├── results/            # Risultati degli esperimenti (immagini, kernel, metadati)
├── main.py             # Script principale per lanciare gli esperimenti model-based
├── utils.py            # Funzioni di utilità (caricamento, blur sintetico, metriche)
├── requirements.txt    # Dipendenze del progetto
├── shan_results_summary.csv # Riepilogo quantitativo dei test su Shan
└── README.md           # Questo file
```

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

## Risultati Parziali: Analisi dell'Algoritmo di Shan

Una serie di test sistematici è stata condotta per valutare le performance dell'algoritmo di Shan al variare del tipo e dell'intensità del blur. I risultati completi sono documentati nel file [shan_results_summary.csv](shan_results_summary.csv).

Di seguito una sintesi rappresentativa:

| Test_ID | Descrizione del Test                               | Livello Blur | PSNR (dB) | SSIM   |
| :------ | :------------------------------------------------- | :----------- | :-------- | :----- |
| T02     | Blur Gaussiano Leggero (parametri prof)            | Leggero      | 23.32     | 0.6787 |
| T06     | Motion Blur Moderato                               | Moderato     | 18.21     | 0.3758 |
| T07     | Gaussian Blur Moderato                             | Moderato     | 12.71     | 0.0555 |
| T10     | Motion Blur Molto Intenso (livello approvato)      | Intenso      | 19.30     | 0.3906 |

**Conclusioni Preliminari:**
- L'algoritmo è molto efficace su blur di lieve entità.
- Le performance degradano rapidamente con l'aumentare dell'intensità del blur, specialmente per il blur di tipo gaussiano.

## Prossimi Passi

Il lavoro prosegue con l'implementazione del secondo approccio previsto:
1.  **Creazione di un Dataset:** Generazione di un vasto set di coppie di immagini (sfocata, nitida) per l'addestramento supervisionato.
2.  **Implementazione U-Net:** Sviluppo di un'architettura di rete neurale di tipo U-Net.
3.  **Addestramento e Valutazione:** Addestramento del modello e confronto delle sue performance con la baseline model-based, utilizzando le stesse metriche (PSNR, SSIM).

## Riferimenti Scientifici

-   **Shan, Q., Jia, J., & Agarwala, A.** (2008). *High-quality motion deblurring from a single image.*
-   **Levin, A., Weiss, Y., Durand, F., & Freeman, W. T.** (2009). *Understanding and evaluating blind deconvolution algorithms.*