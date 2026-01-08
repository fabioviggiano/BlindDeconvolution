--------------------------------------------------------------------------------
# Blind Deconvolution: 

## Confronto tra approcci Model-Based e Data-Driven

##  Descrizione del progetto

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
