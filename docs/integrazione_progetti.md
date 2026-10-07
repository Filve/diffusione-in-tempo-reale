# Integrare diffusione-in-tempo-reale nei propri progetti

Guida in italiano per installare, inizializzare e usare questo fork di
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) dentro
altri progetti (app desktop, server, altre applicazioni AI).

Indice:

1. [Cosa aggiunge questo fork](#1-cosa-aggiunge-questo-fork)
2. [Installazione](#2-installazione) (prerequisiti, build, primo utilizzo, pacchetto portabile)
3. [Come collegarlo a un progetto](#3-come-collegarlo-a-un-progetto) (CMake, API C, FFI, processo, HTTP)
4. [Retrocompatibilità e risorse](#4-retrocompatibilità-e-risorse)
5. [Sicurezza e test](#5-sicurezza-cosa-sapere-e-cosa-fare)
6. [Autori e licenza](#6-autori-e-licenza)
7. [Cronologia delle modifiche del fork](#7-cronologia-delle-modifiche-del-fork)

## 1. Cosa aggiunge questo fork

Il motore resta quello di stable-diffusion.cpp (C++17 + ggml). Le aggiunte
servono a farlo funzionare su più macchine possibile e a fallire in modo
controllato:

| Funzione | Cosa fa | Default |
|---|---|---|
| Rilevamento device | `sd_get_device_count()` / `sd_get_device_info()` elencano CPU, GPU e acceleratori con tipo e memoria, prima di caricare un modello | sempre disponibile |
| Fallback CPU | se il backend richiesto non esiste o non si inizializza, il contesto passa alla CPU con un avviso invece di fallire | **attivo** |
| Memory guard | sopra una soglia di memoria libera pressione (evict dei pesi, prefetch sospeso) prima che l'allocazione fallisca | **disattivo** |
| Server senza modello | `sd-server` parte anche senza `-m`: device e upscale funzionano, la generazione risponde con un errore chiaro | **attivo** |
| Autenticazione server | `--api-key` (Bearer o `X-API-Key`) e allowlist CORS `--cors-origins` | **disattivi** |
| Pacchetto portabile | backend ggml come librerie dinamiche caricate a runtime: un solo pacchetto, ogni macchina usa ciò che ha | opzionale in build |
| Release automatiche | workflow `release-portable.yml`: zip per Linux/macOS/Windows a ogni tag `portable-v*` | su tag |
| Suite di test | `test/`: smoke di sicurezza, robustezza API, fuzzer del caricamento modelli | manuale |

L'unico cambio di comportamento predefinito rispetto a upstream è il fallback
CPU (prima un backend non disponibile causava un errore). Si disattiva con
`disable_backend_fallback = true` (API) o `--disable-backend-fallback` (CLI).

## 2. Installazione

### 2.1 Prerequisiti

- **CMake** >= 3.12 e un compilatore **C++17** (CMake >= 3.15 per usare il
  comando `cmake --install` riportato sotto):
  - macOS: `xcode-select --install` (Apple clang) e `brew install cmake`
  - Linux (Debian/Ubuntu): `sudo apt install build-essential cmake git`
  - Windows: Visual Studio 2022 (workload "Sviluppo di applicazioni desktop con C++") oppure MSYS2; CMake incluso o da cmake.org
- **Python 3** solo per gli script di test e di conversione (facoltativo).
- SDK del backend desiderato, se non CPU: CUDA Toolkit per `-DSD_CUDA=ON`,
  Vulkan SDK per `-DSD_VULKAN=ON`, ecc. (dettagli in [build.md](./build.md)).

### 2.2 Clonazione

```shell
git clone https://github.com/Filve/diffusione-in-tempo-reale.git
cd diffusione-in-tempo-reale
git submodule update --init ggml thirdparty/libwebp thirdparty/libwebm
```

Senza i sottomoduli la configurazione CMake fallisce. Il sottomodulo
`examples/server/frontend` serve solo per l'interfaccia web del server ed è
facoltativo. Eseguire i comandi dalla cartella principale del repository; in
Windows usare PowerShell, il Developer PowerShell/Prompt di Visual Studio o
Git Bash, con `git` e `cmake` disponibili nel `PATH`.

### 2.3 Build e installazione passo-passo

I comandi cambiano leggermente in base alla shell e al generatore CMake.
L'opzione `-DCMAKE_BUILD_TYPE=Release` seleziona la configurazione per
generatori a configurazione singola (per esempio Ninja o Make); con Visual
Studio la configurazione si sceglie invece in fase di build con
`--config Release`.

#### Linux e macOS (Ninja o Make)

```shell
# Dalla radice del repository. Senza backend GPU esplicito si compila la CPU.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# Aggiungere alla configurazione, se richiesto e se l'SDK è installato:
# -DSD_VULKAN=ON oppure -DSD_CUDA=ON

cmake --build build --parallel

# Verifica
./build/bin/sd-cli --list-devices
```

#### Windows con Visual Studio

Aprire un Developer PowerShell/Prompt di Visual Studio con gli strumenti C++
installati, quindi eseguire dalla radice del repository:

```powershell
cmake -S . -B build -A x64
# Aggiungere alla configurazione, se richiesto e se l'SDK è installato:
# -DSD_VULKAN=ON oppure -DSD_CUDA=ON

cmake --build build --config Release --parallel
& ".\build\bin\Release\sd-cli.exe" --list-devices
```

La build Visual Studio è multi-config: se si è compilato `Debug`, l'eseguibile
è invece `.\build\bin\Debug\sd-cli.exe`. Con Ninja su Windows, impostando
`-G Ninja -DCMAKE_BUILD_TYPE=Release` in fase di configurazione, il percorso è
`.\build\bin\sd-cli.exe`. Se il file non viene trovato, verificare la
configurazione selezionata e il percorso stampato da MSBuild, senza spostare
la working directory dentro `build\bin`.

Per installare la libreria e usarla da un altro progetto CMake (facoltativo):

```shell
# Linux/macOS
cmake --install build --prefix "$HOME/.local/sd"
```

```powershell
# Windows; usare un percorso di installazione a cui l'utente può scrivere.
cmake --install build --config Release --prefix "$HOME\sd-install"
```

L'installazione fornisce l'header pubblico `stable-diffusion.h`, le librerie e
i file di configurazione CMake/pkg-config. Non installa gli eseguibili di
esempio: `sd-cli` e `sd-server` restano nella directory di build. Il formato
della libreria dipende dalla piattaforma e dal tipo di build (statica per
default); per usarla da un altro progetto seguire la sezione 3 e mantenere
coerenti architettura, compilatore e configurazione. La directory di
installazione non viene aggiunta automaticamente al `PATH`.

### 2.4 Primo utilizzo: generare un'immagine

Serve un file di pesi (`.safetensors` o `.gguf`), **non incluso** nel
progetto: si scarica da fonti affidabili (es. Hugging Face) scegliendo un
modello adatto alla macchina (vedi [sd.md](./sd.md), [flux.md](./flux.md) e
gli altri documenti per famiglia di modello; per macchine piccole preferire
versioni quantizzate `.gguf`, vedi
[quantization_and_gguf.md](./quantization_and_gguf.md)).

```shell
# generazione base
./build/bin/sd-cli -m modello.safetensors -p "un gatto astronauta" -o gatto.png

# macchina con poca memoria: guardiano + pesi in RAM + budget VRAM prudente
./build/bin/sd-cli -m modello.safetensors -p "un gatto astronauta" -o gatto.png \
    --memory-guard 90 --offload-to-cpu --max-vram -1

# da immagine esistente (img2img)
./build/bin/sd-cli -m modello.safetensors -p "in stile acquerello" \
    -i foto.png --strength 0.6 -o acquerello.png
```

Con Visual Studio su Windows, eseguire dalla radice del repository usando
PowerShell e il percorso della configurazione compilata:

```powershell
& ".\build\bin\Release\sd-cli.exe" -m ".\modello.safetensors" `
    -p "un gatto astronauta" -o ".\gatto.png"
```

Se si è usato Ninja, rimuovere `Release` dal percorso:
`.\build\bin\sd-cli.exe`. I percorsi relativi ai modelli e ai file di output
sono risolti rispetto alla working directory corrente.

Opzioni utili: `--steps`, `--cfg-scale`, `-W`/`-H` (dimensioni), `--seed`,
`-v` (log dettagliati). Elenco completo: `sd-cli --help`.

### 2.5 Primo utilizzo: il server HTTP

```shell
./build/bin/sd-server -m modello.safetensors --api-key mia-chiave
```

Con Visual Studio su Windows, da PowerShell:

```powershell
& ".\build\bin\Release\sd-server.exe" -m ".\modello.safetensors" `
    --api-key "mia-chiave"
```

Con Ninja usare `.\build\bin\sd-server.exe`. Per richieste HTTP da PowerShell
usare `curl.exe` (non l'alias `curl` presente in alcune versioni di
PowerShell), oppure `Invoke-RestMethod`.

Da un client qualsiasi:

```shell
curl -s http://127.0.0.1:1234/sdcpp/v1/capabilities -H "X-API-Key: mia-chiave"

curl -s -X POST http://127.0.0.1:1234/sdcpp/v1/img_gen \
     -H "Content-Type: application/json" -H "X-API-Key: mia-chiave" \
     -d '{"prompt": "un gatto astronauta", "width": 512, "height": 512}'
```

Da una seconda finestra PowerShell su Windows, sostituire `curl` con
`curl.exe`, per esempio:

```powershell
curl.exe -s "http://127.0.0.1:1234/sdcpp/v1/capabilities" `
    -H "X-API-Key: mia-chiave"
```

Sono disponibili anche endpoint compatibili OpenAI (`/v1/images/generations`)
e AUTOMATIC1111 (`/sdapi/v1/txt2img`). Dettagli di sicurezza in sezione 5.

### Pacchetto portabile (consigliato per la distribuzione)
```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
# Per una distribuzione x86 che includa varianti CPU selezionabili a runtime:
# aggiungere -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF.
cmake --build build --config Release --parallel
```

Su Windows il pacchetto richiede una build a configurazione singola per
produrre i file nel percorso atteso dallo script. Da un Developer
PowerShell/Prompt con Ninja installato, usare una directory di build nuova:

```powershell
cmake -S . -B build-portable -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
cmake --build build-portable --parallel
```

Distribuire insieme all'eseguibile i moduli backend dinamici ggml generati
(per esempio `.dll`, `.so` o `.dylib`, a seconda della piattaforma): i backend
non utilizzabili sulla macchina vengono semplicemente ignorati. L'opzione
`SD_BUILD_SHARED_GGML_LIB=ON` è necessaria: senza di essa il CMake del
progetto forza ggml statico.

Per creare lo zip distribuibile (binari + moduli backend + licenze +
LEGGIMI) da una build portabile:

```shell
./scripts/package_portable.sh build    # Linux/macOS/Git Bash; risultato in dist/
```

Lo script di packaging è Bash; su Windows avviarlo da Git Bash o da WSL dalla
radice del repository, passando la directory di build usata sopra:

```bash
./scripts/package_portable.sh build-portable
```

Il workflow usa Ninja proprio per mantenere gli eseguibili in `build/bin`;
con il generatore Visual Studio finiscono invece in `build/bin/Release` e lo
script non li trova. Il pacchetto è rilocabile all'interno della stessa
piattaforma e architettura: si scompatta e si usa, senza installazione.
Il workflow GitHub Actions `release-portable.yml` produce automaticamente gli
zip per Linux x86-64, macOS arm64 e Windows x86-64 (con varianti CPU
SSE/AVX/AVX2/AVX-512 su x86 per le macchine più vecchie) a ogni tag
`portable-v*` o avvio manuale. I pacchetti di release non sono un singolo
binario multipiattaforma: scegliere quello corrispondente al sistema operativo
e all'architettura del computer di destinazione.

## 3. Come collegarlo a un progetto

### 3.1 Libreria C/C++ con CMake

Questo esempio usa la libreria installata al punto 2.3. Sostituire il prefisso
con il percorso effettivamente scelto sulla propria macchina.

```cmake
cmake_minimum_required(VERSION 3.14)
project(mia_app C CXX)

find_package(stable-diffusion REQUIRED)
add_executable(mia_app main.c)
target_link_libraries(mia_app PRIVATE stable-diffusion)
set_target_properties(mia_app PROPERTIES LINKER_LANGUAGE CXX)
```

Configurare passando `CMAKE_PREFIX_PATH` alla directory di installazione:
`-DCMAKE_PREFIX_PATH=/percorso/di/installazione` su Linux/macOS oppure, in
PowerShell, `-DCMAKE_PREFIX_PATH="$HOME\sd-install"` se si è usato il prefisso
Windows dell'esempio. La libreria installata deve essere stata compilata per
la stessa piattaforma, architettura e configurazione ABI dell'applicazione.
Dopo aver aggiornato il fork, ricompilare l'applicazione contro l'header e la
libreria corrispondenti; non mescolare header di una revisione con librerie
precompilate di un'altra.

### 3.2 Sequenza di inizializzazione consigliata

Il ciclo di vita completo, nell'ordine tecnico corretto:

1. **Logging** (prima di tutto): `sd_set_log_callback(cb, user_data)` — senza
   callback la libreria non stampa nulla; ogni messaggio arriva con livello
   (`SD_LOG_DEBUG/INFO/WARN/ERROR`), testo e il puntatore `user_data`.
2. **Rilevamento hardware** (facoltativo ma consigliato):
   `sd_get_device_count()` + `sd_get_device_info(i, &info)` — non caricano
   nulla, si possono chiamare subito e da più thread. Alla prima chiamata
   vengono scoperti i moduli backend dinamici ggml accanto all'eseguibile.
3. **Parametri del contesto**: `sd_ctx_params_init(&cp)` — azzera e imposta i
   default; poi valorizzare solo i campi necessari (tabella sotto).
4. **Creazione del contesto**: `new_sd_ctx(&cp)` — carica il modello, sceglie
   e inizializza i backend (auto-fit + fallback CPU); è l'operazione costosa
   (secondi–minuti). Restituisce `NULL` in caso di errore (motivo nei log).
5. **Avanzamento** (facoltativo): `sd_set_progress_callback(cb, data)` riceve
   `(step, steps, time, data)` a ogni passo di campionamento — è ciò che serve
   per una progress bar.
6. **Parametri di generazione**: `sd_img_gen_params_init(&gp)`; poi prompt e
   dimensioni. Default reali: 512×512, `strength` 0.75, `seed` -1 (= casuale),
   `batch_count` 1, 20 passi.
7. **Generazione**: `generate_image(ctx, &gp, &images, &count)` — bloccante;
   chiamarla da un worker thread se l'app ha una UI.
8. **Annullamento** (facoltativo, da un altro thread):
   `sd_cancel_generation(ctx, SD_CANCEL_ALL)` interrompe appena possibile;
   `SD_CANCEL_NEW_LATENTS` finisce l'immagine in corso e salta le successive.
9. **Lettura del risultato**: ogni `sd_image_t` ha `width`, `height`,
   `channel` (3 = RGB) e `data` — buffer di `width*height*channel` byte, righe
   contigue senza padding, ordine RGB. Copiarlo o convertirlo subito.
10. **Rilascio**: `free_sd_images(images, count)` e poi `free_sd_ctx(ctx)` —
    sempre le funzioni della libreria, mai `free()` del chiamante (su Windows
    CRT diversi causerebbero corruzione).

Campi principali di `sd_ctx_params_t` (tutti gli altri possono restare ai
default):

| Campo | Significato | Default |
|---|---|---|
| `model_path` | checkpoint completo (`.safetensors`/`.gguf`/`.ckpt`) | obbligatorio* |
| `diffusion_model_path` | in alternativa: solo modello di diffusione (componenti separati) | — |
| `n_threads` | thread CPU; `<= 0` = core fisici | auto |
| `backend` | assegnazione backend, es. `"cpu"`, `"vulkan0"`, `"diffusion=cuda0,te=cpu"`; vuoto = migliore disponibile | auto |
| `params_backend` | dove risiedono i pesi: `"cpu"` (RAM), `"disk"`, per modulo | auto-fit |
| `max_vram` | budget per device in GiB, es. `"6"` o `"cuda0=6,vulkan0=4"`; `-1` = riserva 1 GiB | memoria libera |
| `memory_guard` | soglia % del guardiano di memoria (50–99); `0` = spento | `0` |
| `disable_backend_fallback` | `true` = fallire invece di ripiegare sulla CPU | `false` |
| `enable_mmap` | mappa i pesi dal file invece di copiarli in RAM | `false` |
| `wtype` | forza il tipo dei pesi (quantizzazione a caricamento) | tipo del file |

*uno tra `model_path` e `diffusion_model_path` è obbligatorio.

Esempio compilato e collaudato con il percorso `find_package` qui sopra:

```c
#include <stdio.h>
#include "stable-diffusion.h"

int main(int argc, char** argv) {
    // 1. Rilevamento hardware (nessun modello caricato)
    for (size_t i = 0; i < sd_get_device_count(); i++) {
        sd_device_info_t d;
        if (sd_get_device_info(i, &d)) {
            printf("device %zu: %s [%s] %s\n", i, d.name, sd_device_type_name(d.type), d.description);
        }
    }

    // 2. Contesto con memory guard; il fallback CPU è già attivo
    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);          // inizializza SEMPRE prima di impostare i campi
    cp.model_path   = argv[1];
    cp.memory_guard = 90;

    sd_ctx_t* ctx = new_sd_ctx(&cp);
    if (ctx == NULL) {                // modello non valido o backend non disponibile
        return 1;
    }

    // 3. Generazione
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.prompt = "un gatto astronauta";
    gp.width  = 512;
    gp.height = 512;
    gp.seed   = 42;

    sd_image_t* images = NULL;
    int count          = 0;
    int rc             = 1;
    if (generate_image(ctx, &gp, &images, &count) && count > 0) {
        rc = 0;                       // images[i].data / width / height / channel
    }

    free_sd_images(images, count);    // sempre dalla libreria (evita problemi di CRT su Windows)
    free_sd_ctx(ctx);
    return rc;
}
```

Regole importanti per chi integra:

- Chiamare sempre `sd_ctx_params_init()` e `sd_img_gen_params_init()` prima di
  valorizzare i campi: senza init i campi contengono valori casuali.
- Controllare sempre il valore di ritorno di `new_sd_ctx()` e `generate_image()`.
- **Threading**: un contesto non è thread-safe; serializzare gli accessi (come
  fa `sd-server` con un mutex) oppure usare un contesto per thread.
  `sd_cancel_generation()` è pensata per essere chiamata da un altro thread.
  Le funzioni di enumerazione dei device sono state provate da più thread in
  parallelo. I callback (log, progress) arrivano dal thread che sta generando.
- **Durata delle stringhe**: i campi `const char*` dei parametri non vengono
  copiati al momento dell'assegnazione; devono restare validi fino alla
  chiamata (`new_sd_ctx`/`generate_image`) che li consuma. Da linguaggi con
  garbage collector (FFI), ancorare i buffer finché la chiamata non ritorna.
- **Un contesto, molte generazioni**: il modello resta caricato; chiamare
  `generate_image()` più volte sullo stesso contesto è il modo corretto di
  servire più richieste (ricrearlo a ogni richiesta ricarica il modello).
- **Processo e working directory**: con i backend dinamici i moduli ggml
  vengono cercati accanto all'eseguibile, non nella working directory.

### 3.3 Altri linguaggi (TypeScript/Electron, Python, Rust, Go, C#, ...)

L'API è C pura (`include/stable-diffusion.h`), quindi si può usare tramite FFI
o binding. Binding esistenti sono elencati nel [README](../README.md).

Indicazioni tecniche per un binding FFI fatto in casa:

- servono la libreria **dinamica** (`-DSD_BUILD_SHARED_LIBS=ON`) oppure il
  binding N-API/ctypes contro la statica ricompilata PIC;
- dichiarare le strutture esattamente nell'ordine dei campi dell'header (sono
  struct C semplici); dopo ogni aggiornamento del fork rigenerare le
  dichiarazioni, perché i campi possono crescere in coda;
- chiamare sempre `sd_ctx_params_init()` via FFI invece di costruire la struct
  a mano: imposta i default corretti anche per i campi futuri;
- i callback FFI (log/progress) arrivano da thread nativi: nel runtime del
  linguaggio usare il meccanismo apposito (es. `ThreadSafeFunction` in N-API).

Per un'app Electron/Node la via più robusta resta il **processo separato**
(3.4/3.5): isolamento totale dai crash e nessun binding da mantenere.

### 3.4 Come processo: `sd-cli`

Isola il motore dall'applicazione: un crash del motore non abbatte l'app.

- avviare `sd-cli` con gli argomenti della generazione e leggere l'exit code:
  `0` = successo, valore diverso da zero = errore (dettagli su stderr); il
  modo in cui il sistema operativo segnala un crash non è portabile;
- `-o out_%d.png` scrive i file di output; l'app li legge a fine processo;
- `--list-devices` dà l'inventario hardware in formato tabellare
  (`nome<TAB>tipo<TAB>descrizione<TAB>memoria`), facile da parsare;
- opzioni chiave: `--memory-guard 90`, `--backend`, `--max-vram`,
  `--offload-to-cpu`, `--disable-backend-fallback`;
- per annullare: terminare il processo con il meccanismo previsto dal sistema
  operativo e attendere che termini; le modalità di terminazione non sono
  uniformi tra POSIX e Windows.

### 3.5 Come servizio HTTP: `sd-server`

Offre API compatibili OpenAI, `sdapi` e `sdcpp`. L'autenticazione è
disattivata per default: **leggere la sezione 5 prima di esporlo** e usare
`--api-key` / `--cors-origins`.

Il server **parte anche senza modello** (niente loop né uscita silenziosa):

- senza `-m`/`--diffusion-model` si avvia con un avviso in console; gli
  endpoint dei device e di upscale funzionano, quelli di generazione
  rispondono con `{"error": "no model loaded; restart the server with -m/--diffusion-model"}`;
- se il modello è indicato ma non è caricabile, il server **non** parte e
  scrive in console il percorso del modello e il motivo del fallimento
  (exit code 1).

## 4. Retrocompatibilità e risorse

La compatibilità va considerata su più livelli: sistema operativo, architettura
CPU, backend/driver GPU, runtime del compilatore e formato del modello. Un
pacchetto compilato per un sistema operativo non è riutilizzabile direttamente
su un altro; la modalità "portabile" significa rilocabile sulla piattaforma
per cui è stato creato, non universale.

| Piattaforma del pacchetto di release | Architettura prevista | Note |
|---|---|---|
| Linux | x86-64 | La release include varianti CPU x86 selezionabili. Non abilita CUDA o Vulkan. La compatibilità con distribuzioni Linux meno recenti dipende anche dalla versione di glibc e dalle librerie di sistema: verificare sul sistema minimo di destinazione. |
| macOS | arm64 (Apple Silicon) | La release automatica corrente è per macOS arm64; non è un binario Intel/x86-64. Include Metal, che richiede hardware e sistema Apple compatibili. |
| Windows | x86-64 | La release include varianti CPU x86 selezionabili, ma non abilita CUDA o Vulkan. Verificare la presenza dei runtime richiesti dal compilatore; driver GPU e backend GPU vanno predisposti separatamente nelle build personalizzate. |

Questa tabella descrive gli artefatti generati dall'attuale workflow di release,
non una garanzia di compatibilità con tutte le versioni storiche dei sistemi
operativi. Per Windows ARM64, macOS Intel, Linux ARM o Android occorre una
build distinta con toolchain e dipendenze adatte; tali target non fanno parte
dei pacchetti portabili automatici descritti sopra.

- **CPU meno recenti**: per x86, `GGML_CPU_ALL_VARIANTS=ON` con
  `GGML_NATIVE=OFF` produce varianti CPU che il runtime può selezionare in
  base alle istruzioni supportate dal processore. Non permette di eseguire un
  binario x86-64 su CPU a 32 bit né garantisce compatibilità con ogni sistema
  operativo. Evitare `GGML_NATIVE=ON` per pacchetti generici: può ottimizzare
  per il computer di build e renderli incompatibili con CPU più vecchie.
- **Macchine senza GPU o con GPU non funzionante**: il fallback alla CPU è
  attivo per default se il backend richiesto non è disponibile o non si
  inizializza. Per modelli grandi su macchine piccole combinare
  `--offload-to-cpu`, `--params-backend disk` e `--max-vram` (vedi
  [performance.md](./performance.md)).
- **Driver e backend GPU**: i moduli dinamici consentono di includere più
  backend, ma non includono i driver del produttore né rendono compatibili
  GPU non supportate. Installare il runtime/driver richiesto sulla macchina
  finale; se il backend non può avviarsi, il fallback CPU può essere usato,
  con prestazioni potenzialmente molto inferiori.
- **Libreria per integrazioni**: una libreria C/C++ compilata è legata a
  sistema operativo, architettura, ABI e dipendenze usate in compilazione.
  Distribuire build separate per ogni target e non copiare DLL, `.so`, `.dylib`
  o librerie statiche tra sistemi o architetture differenti. Per FFI mantenere
  coerenti dichiarazioni e header della stessa revisione e rilasciare ogni
  risorsa con le funzioni della libreria, non con l'allocatore del linguaggio.
- **Modelli**: i pesi non sono inclusi nel pacchetto portabile; vanno
  distribuiti/scaricati separatamente e la loro licenza va verificata.
  Compatibilità e fabbisogno di memoria dipendono dall'architettura del modello
  e dalle funzionalità supportate dalla build.

- **Memory guard** (`memory_guard` / `--memory-guard <percentuale>`): la
  soglia valida è 50-99 (valori fuori scala vengono riportati nell'intervallo);
  a soglia +5 punti la pressione è "critica". È best effort: non fa fallire
  un grafo che ci sta comunque e non è un limite fisico assoluto, perché le
  allocazioni dei driver fuori dalla gestione del motore non sono visibili.
- **Elenco dispositivi** per scegliere la strategia nell'app: tipo `cpu`,
  `gpu`, `igpu`, `accel`; memoria 0 = non riportata dal backend.
- **Altri motori** (MNN, ncnn, ONNX Runtime): non sono integrati in questo
  repository, perché tutto il motore è basato su ggml. Vanno trattati come
  implementazioni alternative in un livello di astrazione dell'applicazione.

## 5. Sicurezza: cosa sapere e cosa fare

Il motore carica file di modello forniti dall'utente ed `sd-server` accetta
richieste di rete: sono i due punti da proteggere.

### Esiti dei controlli eseguiti su questo fork

- **File di modello ostili**: inesistente, byte casuali, header safetensors con
  lunghezza enorme, header troncato, offset tensori fuori dal file. Tutti
  rifiutati con errore pulito, nessun crash.
- **Argomenti numerici fuori scala** (`--memory-guard 99999999999`,
  `--threads`, `--seed`, `--skip-layers`): trovato un difetto **preesistente**
  nel parser della CLI (eccezione `std::out_of_range` non gestita, terminava
  con SIGABRT). Corretto: ora l'argomento viene rifiutato con errore.
- **API C** (puntatori nulli, indici fuori range, buffer minuscoli, 8 thread
  in parallelo) eseguita con AddressSanitizer e UndefinedBehaviorSanitizer:
  nessun errore.
- **Path traversal sul server**: i LoRA richiesti via HTTP vengono risolti
  solo contro un elenco di file già indicizzati nella cartella configurata,
  quindi un percorso arbitrario inviato dal client viene rifiutato.
- **Server senza modello (test HTTP dal vivo)**: avvio senza `-m`, endpoint
  `capabilities`/`models` funzionanti, generazione rifiutata con errore JSON
  chiaro, body malformati gestiti senza crash.
- **Fuzzing prolungato del caricamento modelli**: 30 068 esecuzioni in 10
  minuti con `test/fuzz_model_loader.py` (seed 7) contro una build
  AddressSanitizer: nessun crash, nessun errore di memoria.

### Punti di attenzione del server

- Ascolta su `127.0.0.1` per impostazione predefinita: **non cambiare
  `--listen-ip` verso `0.0.0.0` senza protezioni**.
- **Autenticazione opzionale** (consigliata): `--api-key <chiave>` richiede su
  ogni richiesta l'header `Authorization: Bearer <chiave>` oppure
  `X-API-Key: <chiave>`; senza chiave valida risponde 401. Il confronto è a
  tempo costante e la chiave non viene mai scritta nei log. Default: nessuna
  autenticazione (comportamento upstream).
- **CORS**: per default riflette qualsiasi `Origin` con
  `Allow-Credentials: true` (comportamento upstream); con
  `--cors-origins "http://localhost:3000,https://mia-app.example"` solo le
  origini elencate ricevono gli header CORS, le altre vengono bloccate dal
  browser.

Esempio di avvio protetto:

```shell
sd-server -m modello.safetensors --api-key "$(openssl rand -hex 24)" \
          --cors-origins "http://localhost:3000"
```

Verifica dal vivo eseguita: 401 senza chiave o con chiave errata, 200 con
`Bearer` o `X-API-Key` corretti, preflight `OPTIONS` libero (204), header CORS
presenti solo per le origini consentite.

Raccomandazioni per chi integra:

1. Usare `sd-cli` o la libreria in-process quando possibile, evitando di
   esporre una porta.
2. Se serve HTTP, tenerlo su `127.0.0.1`, impostare `--api-key` e
   `--cors-origins`, e per esposizioni più ampie aggiungere un proxy con
   limiti di dimensione/frequenza delle richieste.
3. Caricare solo modelli di cui ci si fida (preferire `.safetensors` o `.gguf`;
   il caricatore pickle è scritto per non eseguire il codice contenuto nei
   checkpoint, ma resta una superficie di parsing più ampia) e scaricarli
   da fonti verificate.
4. Eseguire il motore con i permessi minimi (utente dedicato, cartella modelli
   in sola lettura).
5. Limitare risoluzione, numero di step e dimensione dei file in ingresso a
   livello applicativo.

### Limiti dei test (onestà)

Non sono stati eseguiti: generazione completa con un modello reale, prova
della memory guard sotto pressione reale, test su CUDA/Vulkan. Il progetto
non può quindi essere dichiarato "privo di vulnerabilità": questi controlli
riducono il rischio, non lo azzerano.

### Come eseguire i test

I test sono inclusi nella cartella [test/](../test/) del fork (vedi anche
[test/README.md](../test/README.md)):

```shell
# 1. Test rapido (file modello ostili + opzioni estreme), ~1 minuto
./test/security_smoke.sh build/bin/sd-cli

# 2. Robustezza dell'API C con sanitizer (istruzioni di build nel file)
#    test/api_robustness.cpp

# 3. Fuzzing del caricamento modelli (consigliata una build AddressSanitizer)
cmake -B build_asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSD_WEBP=OFF -DSD_WEBM=OFF \
      -DCMAKE_C_FLAGS=-fsanitize=address -DCMAKE_CXX_FLAGS=-fsanitize=address \
      -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address
cmake --build build_asan -j --target sd-cli
python3 test/fuzz_model_loader.py build_asan/bin/sd-cli --seconds 600
```

Criterio di esito: ogni input invalido deve produrre un errore pulito
(exit 1), mai un segnale (exit >= 128) né un report dei sanitizer. Gli input
che causano una scoperta vengono salvati in `test/fuzz_findings/`.

Per la policy di upstream (`CONTRIBUTING.md`) i test non vanno inclusi nelle
PR verso il progetto originale: sono una dotazione di questo fork.

## 6. Autori e licenza

### Autori

- **leejet** e tutti i contributori di
  [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp):
  autori del progetto originale (Copyright (c) 2023 leejet).
- **Gli autori di ggml** e delle librerie in `thirdparty/`, ciascuna con la
  propria licenza (vedere i file di licenza nelle rispettive cartelle).
- **Francesco Simeoni** ([@filve](https://github.com/filve)): autore delle
  modifiche di questo fork e responsabile della loro revisione
  (Copyright (c) 2026, modifiche del fork).

Lo sviluppo delle modifiche è stato assistito da GitHub Copilot (strumento
AI). In linea con `CONTRIBUTING.md` di upstream, lo strumento non è indicato
come co-autore: la responsabilità del codice è dell'autore umano.

### Licenza e modifiche

Il progetto originale è distribuito con **licenza MIT**, che consente di
usare, modificare e distribuire il software mantenendo il copyright e il testo
della licenza. Abbiamo quindi modificato il progetto restando nei termini
della MIT, conservando integralmente il copyright originale in `LICENSE` e
aggiungendo il nostro per le modifiche, e **continuiamo a migliorarlo** con
questi criteri:

- le nuove funzioni sono opt-in, salvo il fallback CPU (sezione 1), così da
  non degradare il comportamento di upstream;
- le modifiche restano piccole, documentate e riconducibili ai punti di
  estensione esistenti, per poter sincronizzare il fork con upstream;
- l'obiettivo è renderlo utilizzabile anche per retrocompatibilità futura,
  su hardware e sistemi operativi diversi.

I **modelli** (pesi `.safetensors`/`.gguf`) hanno licenze proprie,
indipendenti da quella del motore: verificarle prima di ridistribuirli.

## 7. Cronologia delle modifiche del fork

Ogni passaggio svolto su questo fork, nell'ordine in cui è stato fatto, con i
file principali toccati. Utile per capire cosa è diverso da upstream e per
sincronizzazioni future.

1. **Memory guard ("watchguardian")** — componente `sd::MemoryWatchdog`
   (`src/core/memory_watchdog.{h,cpp}`) innestato nel `ModelManager`: sopra la
   soglia riduce la residenza dei pesi (eviction), sospende il prefetch e
   forza l'esecuzione segmentata; best effort, mai fallimenti artificiali.
   API: campo `memory_guard` in `sd_ctx_params_t`; CLI: `--memory-guard`.
   Documentato in [performance.md](./performance.md).
2. **API di rilevamento device** — `sd_get_device_count()`,
   `sd_get_device_info()` (nome, tipo, memoria libera/totale),
   `sd_device_type_name()` in `include/stable-diffusion.h` +
   `src/core/util.cpp`; `--list-devices` arricchito.
3. **Fallback CPU** — `init_backend()` in `src/pipeline/diffusion_engine.cpp`
   ritenta con la CPU se il backend richiesto non esiste o fallisce l'init.
   Campo `disable_backend_fallback` / flag `--disable-backend-fallback`.
   Documentato in [backend.md](./backend.md).
4. **Build portabile** — backend ggml come moduli dinamici
   (`-DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON`);
   verificato il degrado rimuovendo un modulo. Documentata in
   [build.md](./build.md).
5. **Correzione di un bug preesistente della CLI** — valori numerici fuori
   range (`--seed`, `--threads`, ecc.) causavano SIGABRT
   (`std::out_of_range` non gestita in `examples/common/common.cpp`); ora
   errore pulito.
6. **Questa guida** + sezione autori, riga di copyright del fork in `LICENSE`,
   nota "Fork e autori" nel `README.md`.
7. **Server senza modello** — `sd-server` parte anche senza `-m` con log
   chiari (`examples/server/main.cpp`, `runtime.cpp`,
   `examples/common/common.{h,cpp}` con `require_model`); messaggio di errore
   dedicato sugli endpoint di generazione. Testato dal vivo via HTTP.
8. **Suite di test** — `test/security_smoke.sh`, `test/api_robustness.cpp`,
   `test/fuzz_model_loader.py` (+ `.gitignore` aggiornato per tracciarla).
   Fuzzing prolungato eseguito: 30 068 esecuzioni con AddressSanitizer,
   0 scoperte.
9. **Autenticazione e CORS del server** — `--api-key` (Bearer/`X-API-Key`,
   confronto a tempo costante) e `--cors-origins` (allowlist), default
   invariati. Testati dal vivo.
10. **Packaging e release** — `scripts/package_portable.sh` (zip rilocabile
    con binari, moduli backend, licenze, LEGGIMI) e workflow
    `.github/workflows/release-portable.yml` per Linux/macOS/Windows su tag
    `portable-v*`.

Passaggi previsti e non ancora svolti: collaudo del workflow di release sui
runner GitHub, generazione con un modello reale (inclusa la memory guard sotto
pressione vera), test su CUDA/Vulkan, e il livello di astrazione
`ImageRuntime` nell'applicazione ospite (fuori da questo repository).
