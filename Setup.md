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
