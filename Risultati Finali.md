La valutazione è stata condotta utilizzando le metriche quantitative PSNR (Peak Signal-to-Noise Ratio) e SSIM (Structural Similarity Index).

| Metodo | Scenario (Intensità Blur) | PSNR Medio (dB) | SSIM Medio | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Shan et al.** | Realistico | 23.32 | 0.6800 | Buona ricostruzione su blur lievi, sensibile agli iperparametri. |
| **Shan et al.** | Forte | 12.92 | 0.6000 | Crollo delle performance con artefatti evidenti su blur intensi. |
| **U-Net** | Vario (Test Set) | **34.18** | **0.9056** | Eccellente robustezza e generalizzazione anche in casi critici. |

## Risultati quantitativi

La tabella nel paragrafo precedente mostra le metriche medie di performance (PSNR e SSIM) calcolate sui rispettivi set di test. Valori più alti indicano una migliore qualità di ricostruzione.

I risultati dimostrano una netta superiorità del modello U-Net. L'approccio deep learning non solo supera il metodo classico in tutti gli scenari, ma si dimostra particolarmente robusto nel gestire sfocature aggressive, un'area in cui l'algoritmo di Shan mostra i suoi limiti.

## Analisi Visiva dei Risultati della U-Net

Di seguito sono riportati tre casi studio rappresentativi delle performance del modello U-Net.

---
### 1. Caso Eccellente (File: 27_v2.png)
*   **Metriche:** PSNR: 40.34 dB, SSIM: 0.9645
*   **Analisi:** La ricostruzione è quasi indistinguibile dall'originale, dimostrando il massimo potenziale del modello.
![Caso Eccellente](https://github.com/fabioviggiano/BlindDeconvolution/blob/master/BlindDeconvolution/Report%20Finale/01_BestCase.jpg)

---
### 2. Caso Medio (File: 103_v0.png)
*   **Metriche:** PSNR: 34.40 dB, SSIM: 0.9177
*   **Analisi:** Un risultato rappresentativo delle performance medie. Il blur è stato rimosso con successo, ripristinando la maggior parte dei dettagli strutturali.
![Caso Medio](https://github.com/fabioviggiano/BlindDeconvolution/blob/master/BlindDeconvolution/Report%20Finale/02_AverageCase.jpg))

---
### 3. Caso Difficile (File: 222_v0.png)
*   **Metriche:** PSNR: 22.53 dB, SSIM: 0.7258
*   **Analisi:** In presenza di un blur complesso, il modello riesce a recuperare la struttura principale dell'immagine, sebbene con alcuni artefatti visibili. Questo evidenzia i limiti del metodo pur mostrando un miglioramento significativo rispetto all'input.
![Caso Difficile](https://github.com/fabioviggiano/BlindDeconvolution/blob/master/BlindDeconvolution/Report%20Finale/03_Worstacase.jpg))

## Conclusioni: 

Mentre l'approccio classico di Shan fornisce risultati validi in condizioni controllate, dimostra fragilità all'aumentare dell'intensità del blur. La U-Net supera significativamente il metodo classico, dimostrando che l'approccio data-driven è più robusto nel gestire la natura ill-posed della blind deconvolution, evitando i minimi locali tipici dell'approccio MAP.

## Struttura del Repository

BlindDeconvolution/

• src/classical: Implementazione dell'algoritmo di Shan e script di calibrazione.
• src/deep_learning: Pipeline di training e inferenza per la U-Net (PyTorch).
• data: Script per la generazione del dataset sintetico (create_dataset.py).
• results: Confronti visivi e log delle metriche.

--------------------------------------------------------------------------------

Progetto sviluppato nell'ambito del corso di Computational Imaging per l' esame dello stesso.
