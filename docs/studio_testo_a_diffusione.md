# Studio di fattibilità: generare testo con un modello di linguaggio a diffusione

Studio del 14 ottobre 2026. **Nessun codice del motore è stato scritto per la
generazione di testo**: senza il checkpoint del modello non c'è modo di
verificarlo, e un codice che non si può verificare non va nel motore.

## In breve

- Il motore contiene già il cuore di un modello di linguaggio a diffusione
  (LLaDA2-MoE), ma lo usa solo come encoder di testo di LLaDA-Image: produce
  stati nascosti, non parole.
- Per scrivere testo mancano sei pezzi (sezione 2). Nessuno è impossibile, ma
  ognuno va confrontato con l'implementazione di riferimento, e serve il
  checkpoint.
- Il modello candidato, LLaDA2.0-mini (16 miliardi di parametri), non sta nei
  6 GB di VRAM di un portatile e in RAM sarebbe lento: l'ordine di grandezza
  stimato è un paio di token al secondo (sezione 3).
- Per una chat o per un agente che controlla il browser conviene oggi un modello
  di linguaggio normale che usa il motore come strumento (sezione 3.6 di
  [integrazione_progetti.md](./integrazione_progetti.md)). Per provare un modello
  a diffusione sulla stessa macchina esiste già `llama-diffusion-cli` di
  llama.cpp, con modelli più piccoli (sezione 4).

## 1. Che cosa c'è già nel motore

| Pezzo | Dove | Stato |
|---|---|---|
| Backbone LLaDA2-MoE: 20 strati, dimensione 2048, 256 esperti con 8 attivi per token più 1 condiviso (primo strato denso), attenzione bidirezionale, vocabolario di 173 568 token | `src/model/te/llm.hpp` (`LLMArch::LLADA2_MOE`) | in uso come encoder di testo ([llada_image.md](./llada_image.md)) |
| Tokenizzatore da `tokenizer.json` esterno; l'interfaccia ha anche `decode()` | `src/tokenizers/`, [tokenizers.md](./tokenizers.md) | in uso per codificare; la decodifica non è stata provata con questo vocabolario |
| Collocazione dei pesi su GPU, RAM e disco, divisione per blocchi su più GPU, esecuzione a segmenti | `src/core/backend_fit.cpp`, [backend.md](./backend.md) | in uso per le immagini |
| Server asincrono che ricarica il modello su un profilo più leggero | `examples/server/` | in uso per le immagini |

Controllo sui numeri: 256 esperti × 3 matrici × 2048 × 512 per 19 strati MoE
fanno circa 15,3 miliardi di parametri, coerenti con i 16 della scheda del
modello.

## 2. Che cosa manca

| # | Pezzo mancante | Perché serve | Come si valida |
|---|---|---|---|
| 1 | **Testa del vocabolario** (`lm_head`) | `llm.hpp` non la costruisce; la mappatura dei nomi dei checkpoint prevede il tensore, quindi il file probabilmente lo contiene e oggi non viene usato | logit del primo passo uguali a quelli di riferimento, entro una tolleranza da fissare confrontando F16 con il riferimento |
| 2 | **Campionatore a blocchi** | il modello scrive a blocchi di 32 token: parte da posizioni "maschera" e le rifinisce in 32 passi, fissando a ogni passo una parte dei token con un criterio di confidenza (da leggere nel codice di riferimento); servono token maschera, criterio di fissaggio, temperatura, arresto | stessi token del riferimento a temperatura 0 su alcuni prompt |
| 3 | **Maschera di attenzione a blocchi e cache** | oggi l'attenzione è bidirezionale su tutta la sequenza (va bene per un encoder); per generare ogni blocco vede il prompt e i blocchi già fissati, che si possono tenere in cache (circa 40 KB per token: 20 strati × 4 teste KV × 128 × 2 × 2 byte, cioè 1,3 GB a 32 768 token). Il disegno esatto va letto nel codice di riferimento del modello | stessi token con e senza cache; tempo per blocco ridotto |
| 4 | **Modello di conversazione** | il motore non interpreta i template chat del tokenizer: ruoli e token speciali vanno scritti a mano per questo modello | risposta sensata su un dialogo di prova, formato verificato con un test |
| 5 | **Superficie pubblica** | un contesto di testo separato da `sd_ctx_t` (il modello è un altro), funzioni C aggiuntive e compatibili, una rotta `/v1/chat/completions` con flusso a blocchi, ricarica e profili anche per questo contesto | test del server con libreria simulata, come `test/server_recovery_test.cpp` |
| 6 | **Validazione sul modello vero** | senza checkpoint nulla di quanto sopra è dimostrato | confronto con l'implementazione ufficiale (Transformers) |

## 3. Il modello e la macchina

Dati dichiarati dagli autori (scheda di
[LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini)): 16 miliardi di
parametri totali, 1,4 miliardi attivi per passo, contesto di 32 768 token,
licenza Apache-2.0, esempio d'uso con `block_length=32`, `steps=32` e temperatura
0. Gli ambienti elencati nella scheda sono Transformers, SGLang, vLLM e Docker
Model Runner.

Stime mie, non misurate (parametri × bit per peso):

| Formato | Dimensione dei pesi |
|---|---|
| F16 | circa 32 GB |
| Q8_0 | circa 17 GB |
| Q4 (famiglia K) | circa 9-10 GB |

Su un portatile con 6 GB di VRAM e circa 15 GiB di RAM, quindi: il modello
quantizzato a 4 bit sta in RAM ma non in VRAM. Si potrebbero tenere in VRAM
attenzione, embedding ed esperto condiviso e portare gli esperti in corsa, ma i
modelli a diffusione sfruttano male questa strategia: in un modello
autoregressivo un token usa 8 esperti su 256, qui un passo elabora 32 posizioni
insieme e le loro scelte coprono gran parte degli esperti (256 scelte su 256
esperti ne toccano il 63% se uniformi; la scelta reale è più sbilanciata, quindi
meno, ma resta una quota grande). Con circa 5-6 GB da far passare per ogni passo,
a circa 12 GB/s di PCIe e 32 passi per blocco, il risultato stimato è dell'ordine
di un quarto di minuto per blocco di 32 token, cioè **un paio di token al
secondo**;
calcolare gli esperti sulla CPU dà un ordine di grandezza simile. Meno passi
accelerano, a un costo di qualità che andrebbe misurato.

Altri punti di riferimento, riferiti da terzi e non verificati qui:

- secondo [un articolo dell'8 agosto 2026](https://openclawdc.com/blog/open-text-diffusion-models-local/),
  llama.cpp esegue Dream, LLaDA e RND1 con `llama-diffusion-cli`; LLaDA-8B e
  Dream-7B sono modelli densi da 8 e 7 miliardi (a 4 bit, per una mia stima,
  circa 4-5 GB). L'articolo avverte che LLaDA-8B rende meglio con tanti passi quanti
  sono i token (perdendo il vantaggio della parallelizzazione), che RND1 è un
  artefatto di ricerca e che Apple Silicon guadagna poco dalla decodifica
  parallela;
- LLaDA2.0-mini gira su SGLang e vLLM; il [GGUF di LLaDA2.2-flash](https://huggingface.co/Akicou/inclusionAI_LLaDA2.2-flash-GGUF)
  (103 miliardi di parametri, 13 attivi) dichiara di **non** funzionare con
  llama.cpp e rimanda a un fork (`diffuse-cpp`), con 59 GB per la quantizzazione
  Q4_K_S. È fuori portata per un portatile.

## 4. Strade possibili

| Strada | Costo | Quando ha senso |
|---|---|---|
| **A. Niente testo nel motore**: la chat usa un modello di linguaggio normale (locale o via API) e il motore è uno strumento | nessuno, già documentato (sezione 3.6 della guida) | chat e agenti che controllano il browser: serve affidabilità nell'uso di strumenti, che per i modelli a diffusione di questa classe non ho trovato dimostrata |
| **B. Processo ausiliario con `llama-diffusion-cli`** (LLaDA-8B o Dream-7B a 4 bit) | un wrapper nell'applicazione ospite; nessuna modifica al fork | per provare davvero la generazione a diffusione sulla stessa macchina. È un programma a riga di comando: un server HTTP per questi modelli non risulta dalle fonti lette |
| **C. Implementarlo nel motore** (fasi sotto) | grande, e servono checkpoint e una macchina con memoria a sufficienza | se si vuole un unico binario che fa immagini e testo, sfrutta `--multi-gpu` e la collocazione dei pesi, e si ha un modo di validarlo |
| **D. SGLang o vLLM** su una GPU grande | un altro servizio da gestire | per usare LLaDA2.0-mini o più grandi con le prestazioni dichiarate dagli autori |

Raccomandazione: **A** per chat e agenti, **B** se interessa sperimentare la
diffusione sul testo con l'hardware attuale, **C** solo se si ottiene una macchina
dove validarlo.

## 5. Piano a fasi per la strada C

Ogni fase è una modifica piccola, con il proprio test e un commit a parte; le
funzioni pubbliche si aggiungono, non cambiano.

| Fase | Contenuto | Accettazione |
|---|---|---|
| 0 | Procurarsi il checkpoint e produrre il riferimento con Transformers: token generati a temperatura 0 e logit del primo passo per 3 prompt, salvati in file | file di riferimento riproducibili da un altro avvio |
| 1 | Testa del vocabolario | logit del primo passo entro la tolleranza fissata, su CPU in F16 |
| 2 | Campionatore a blocchi, senza cache | stessi token del riferimento sui 3 prompt |
| 3 | Maschera a blocchi e cache | stessi token della fase 2, tempo per blocco ridotto e misurato |
| 4 | Modello di conversazione, decodifica e arresto | risposta leggibile; test del formato |
| 5 | Funzioni C e rotta di chat con flusso a blocchi | test del server con libreria simulata, senza modello |
| 6 | Collocazione per la memoria: 1 GPU da 6 GB (streaming) e più GPU (divisione) | token al secondo misurati, riportati nella documentazione |

Una macchina a noleggio con due GPU per un'ora servirebbe anche a provare
`--multi-gpu` su hardware vero ([backend.md](./backend.md)): la stessa
occasione copre le due validazioni, a una spesa da decidere.

## 6. Rischi e limiti

- **Qualità d'uso**: il vantaggio dichiarato dei modelli a diffusione è la
  decodifica parallela, che rende su GPU grandi; su un portatile con poca memoria
  non ci si può contare. Per istruzioni complesse e chiamate di strumenti i
  modelli autoregressivi sono per ora più maturi (valutazione mia, non misurata).
- **Sensibilità numerica**: instradamento degli esperti e quantizzazione possono
  spostare le soglie di confidenza del campionatore; la fase 2 deve confrontare
  token, non solo la verosimiglianza.
- **Manutenzione**: l'architettura evolve (esiste già una serie 2.2); il codice
  scritto per 2.0-mini potrebbe richiedere adattamenti.
- **Tensor parallelism**: ggml, nel sottomodulo di questo fork, ha un "meta
  backend" per dividere i calcoli di un grafo fra più GPU
  (`ggml_backend_meta_device`), che il motore non usa. Sarebbe la strada per far
  correre più veloce una singola generazione su più GPU, per le immagini come
  per il testo. Non è stato valutato qui.
- **Vincoli del progetto**: tutto il lavoro resta nel fork, su `fix_to_claude`;
  nessuna modifica a `ggml/`, a `thirdparty/` né al repository di upstream.

## Fonti

- Scheda del modello [inclusionAI/LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini)
- [Open-Source Text Diffusion Models You Can Run Locally (2026)](https://openclawdc.com/blog/open-text-diffusion-models-local/), 8 agosto 2026
- [LLaDA2.2-flash GGUF](https://huggingface.co/Akicou/inclusionAI_LLaDA2.2-flash-GGUF), pagina di terzi
- Nel repository: [llada_image.md](./llada_image.md), [tokenizers.md](./tokenizers.md),
  [backend.md](./backend.md), `src/model/te/llm.hpp`, `src/name_conversion.cpp`
