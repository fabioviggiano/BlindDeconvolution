--------------------------------------------------------------------------------
Blind Deconvolution: Confronto tra Approcci Model-Based e Data-Driven
Descrizione del Progetto
Questo repository ospita un'analisi comparativa tra due paradigmi fondamentali per la risoluzione del problema della Blind Deconvolution: il recupero di un'immagine nitida (latent image) a partire da una versione sfocata (blurred image) quando il kernel di sfocatura è sconosciuto.
Il progetto esplora la transizione dalle tecniche classiche di ottimizzazione matematica alle moderne soluzioni basate sul Deep Learning, valutandone le prestazioni su sfocature di tipo Motion e Gaussian.
Metodi Confrontati
1. Approccio Classico (Model-Based): Shan et al. (2008)
Implementazione dell'algoritmo di Shan et al., che utilizza un modello matematico esplicito basato su:
• Stima MAP (Maximum A Posteriori): Ottimizzazione alternata per stimare iterativamente l'immagine latente e il kernel di sfocatura.
• Prior Sparsi: Utilizzo di statistiche sui gradienti delle immagini naturali per guidare la ricostruzione.
• Limitazioni Teoriche: Come evidenziato da Levin et al., gli approcci MAP ingenui tendono spesso a favorire la soluzione banale "nessuna sfocatura" (kernel delta), specialmente se non vincolati correttamente.
2. Approccio Moderno (Data-Driven): U-Net
Implementazione di una Convolutional Neural Network (CNN) con architettura U-Net:
• Apprendimento Supervisionato: Il modello apprende la mappatura inversa (deblurring) osservando un vasto dataset di coppie sfocata/nitida generate sinteticamente.
• Architettura: Struttura encoder-decoder con skip connections per preservare i dettagli spaziali ad alta frequenza necessari per una ricostruzione di qualità.
• Dataset: Training effettuato su immagini mediche (dataset Mayo/C081) con applicazione randomica di kernel di sfocatura.
Risultati Chiave
La valutazione è stata condotta utilizzando le metriche quantitative PSNR (Peak Signal-to-Noise Ratio) e SSIM (Structural Similarity Index).
Metodo
Scenario (Intensità Blur)
PSNR Medio (dB)
SSIM Medio
Note
Shan et al.
Realistico
23.32
0.6800
Buona ricostruzione su blur lievi, sensibile agli iperparametri.
Shan et al.
Forte
12.92
0.6000
Crollo delle performance con artefatti evidenti su blur intensi.
U-Net
Vario (Test Set)
34.18
0.9056
Eccellente robustezza e generalizzazione anche in casi critici.
Conclusioni: Mentre l'approccio classico di Shan fornisce risultati validi in condizioni controllate, dimostra fragilità all'aumentare dell'intensità del blur. La U-Net supera significativamente il metodo classico, dimostrando che l'approccio data-driven è più robusto nel gestire la natura ill-posed della blind deconvolution, evitando i minimi locali tipici dell'approccio MAP.
Struttura del Repository
• src/classical: Implementazione dell'algoritmo di Shan e script di calibrazione.
• src/deep_learning: Pipeline di training e inferenza per la U-Net (PyTorch).
• data: Script per la generazione del dataset sintetico (create_dataset.py).
• results: Confronti visivi e log delle metriche.

--------------------------------------------------------------------------------
Progetto sviluppato nell'ambito del corso di Computational Imaging.
