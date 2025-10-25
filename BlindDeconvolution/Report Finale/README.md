# Report Finale: Confronto tra Metodi di Blind Deconvolution

Questo documento riassume i risultati finali del progetto, confrontando un metodo classico (algoritmo di Shan) con un approccio basato su deep learning (U-Net) per il deblurring di immagini mediche (TC).

## Risultati Quantitativi

La tabella seguente mostra le metriche medie di performance (PSNR e SSIM) calcolate sui rispettivi set di test. Valori più alti indicano una migliore qualità di ricostruzione.

| Metodo                  | Livello di Blur        | PSNR Medio (dB) | SSIM Medio |
| :---------------------- | :--------------------- | :-------------: | :--------: |
| Shan (Classico)         | Realistico             |      23.32      |   0.6800   |
| Shan (Classico)         | Forte                  |      12.92      |   0.0600   |
| **U-Net (Deep Learning)** | **Vario (981 immagini)** |    **34.18**    | **0.9056** |

## Conclusione

I risultati quantitativi dimostrano una netta superiorità del modello U-Net. L'approccio deep learning non solo supera il metodo classico in tutti gli scenari, ma si dimostra particolarmente robusto nel gestire sfocature aggressive, un'area in cui l'algoritmo di Shan mostra i suoi limiti.

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
![Caso Medio](02_Caso_Medio.png)

---
### 3. Caso Difficile (File: 222_v0.png)
*   **Metriche:** PSNR: 22.53 dB, SSIM: 0.7258
*   **Analisi:** In presenza di un blur complesso, il modello riesce a recuperare la struttura principale dell'immagine, sebbene con alcuni artefatti visibili. Questo evidenzia i limiti del metodo pur mostrando un miglioramento significativo rispetto all'input.
![Caso Difficile](03_Caso_Difficile.png)
