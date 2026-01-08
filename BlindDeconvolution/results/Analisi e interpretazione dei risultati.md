# Analisi e interpretazione dei risultati

Questa tabella permette di trarre conclusioni chiare e **basate sui dati**, in linea con quanto richiesto in un progetto di ricerca sperimentale.

## 1. Conferma totale del feedback del professore

Il confronto tra **T01** (PSNR: 12.92) e **T02** (PSNR: 23.32) dimostra in modo quantitativo l’impatto del suggerimento del professore.  
Il passaggio da un blur *“esagerato”* a uno *“realistico”* ha:

- quasi **raddoppiato il PSNR**
- portato l’**SSIM** da un valore pessimo (**0.06**) a uno buono (**0.68**)

Questi risultati rappresentano la prova più forte che la direzione intrapresa è corretta.

## 2. Chiara degradazione delle performance con l’aumentare del blur

La tendenza osservata è **inequivocabile**. Dai test sistematici emerge che:

### Blur leggero (T02, T04, T05)
- PSNR costantemente **superiore a 21 dB**
- L’algoritmo funziona **molto bene**

### Blur moderato (T06, T07)
- Forte **crollo delle performance**
- Motion blur (T06): PSNR = **18.21 dB**
- Gaussian blur (T07): PSNR = **12.71 dB**, risultato quasi inutilizzabile

### Blur intenso (T08–T13)
- Risultati **variabili ma generalmente bassi**
- Caso interessante: motion blur con `len = 30` (T08)  
  - PSNR = **22.13 dB**, sorprendentemente buono  
  - Probabile effetto di una combinazione favorevole di parametri
- In generale, i valori si assestano **sotto i 20 dB**

## 3. Sensibilità al tipo di blur: il gaussiano è più “difficile”

Confrontando test a **parità di livello di blur**, emerge un pattern chiaro:

- **Blur leggero**
  - Motion (T04): PSNR = **22.63**
  - Gaussian (T05): PSNR = **21.67**
  - → Performance simili

- **Blur moderato**
  - Motion (T06): PSNR = **18.21**
  - Gaussian (T07): PSNR = **12.71**
  - → Il blur gaussiano è nettamente più dannoso

Questo suggerisce che l’algoritmo, basato sui **gradienti**, soffre in particolare la distruzione **diffusa e omogenea** dei dettagli tipica del blur gaussiano di media-alta intensità.

## 4. Importanza della dimensione del kernel di stima

Il confronto tra:

- **T09** (`kernel_size = 35`)
- **T12** (`kernel_size = 61`)

a parità di blur (`motion_len = 30`) è particolarmente illuminante.

Aumentare eccessivamente lo spazio di ricerca del kernel:

- peggiora i risultati (**PSNR da 19.30 a 17.62**)
- conferma l’ipotesi che un kernel troppo grande può:
  - introdurre **rumore**
  - portare a una **stima meno precisa**

Questa tabella è uno **strumento di confronto fondamentale** per dimostrare:

- dove
- e quanto

l’approccio **end-to-end supervisionato** supererà (o eventualmente no) il metodo **model-based**.

Secondo la previsione del professore, le reti neurali dovrebbero comportarsi meglio soprattutto nei casi di **blur più intenso**, dove l’algoritmo di Shan ha mostrato le maggiori difficoltà (ad esempio **T07, T11, T13**).
