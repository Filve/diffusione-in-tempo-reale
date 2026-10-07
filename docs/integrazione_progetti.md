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

- **CMake** >= 3.12 e un compilatore **C++17**:
  - macOS: `xcode-select --install` (Apple clang) e `brew install cmake`
  - Linux (Debian/Ubuntu): `sudo apt install build-essential cmake git`
  - Windows: Visual Studio 2022 (workload "Sviluppo di applicazioni desktop con C++") oppure MSYS2; CMake incluso o da cmake.org
- **Python 3** solo per gli script di test e di conversione (facoltativo).
- SDK del backend desiderato, se non CPU: CUDA Toolkit per `-DSD_CUDA=ON`,
  Vulkan SDK per `-DSD_VULKAN=ON`, ecc. (dettagli in [build.md](./build.md)).

### 2.2 Clonazione

```shell
git clone <URL-del-fork> diffusione-in-tempo-reale
cd diffusione-in-tempo-reale
git submodule update --init ggml thirdparty/libwebp thirdparty/libwebm
```

Senza i sottomoduli la configurazione CMake fallisce. Il sottomodulo
`examples/server/frontend` serve solo per l'interfaccia web del server ed è
facoltativo.

### 2.3 Build e installazione passo-passo

```shell
# 1. Configura (scegli i backend: nessuno = CPU; su macOS Metal è attivo di default)
cmake -B build -DCMAKE_BUILD_TYPE=Release        # es. -DSD_VULKAN=ON, -DSD_CUDA=ON

# 2. Compila (usa tutti i core)
cmake --build build -j

# 3. Verifica subito che funzioni
./build/bin/sd-cli --list-devices

# 4. (Facoltativo) installa in un prefisso per usarlo da altri progetti
cmake --install build --prefix "$HOME/.local/sd"   # o /usr/local con sudo
```

L'installazione contiene `include/stable-diffusion.h`, le librerie
(`libstable-diffusion.a`, `libggml*.a`), la configurazione CMake
(`lib/cmake/stable-diffusion`) e gli eseguibili `sd-cli` e `sd-server`.
Su Windows usare `cmake --build build --config Release` e il prompt
"x64 Native Tools" di Visual Studio.

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

Opzioni utili: `--steps`, `--cfg-scale`, `-W`/`-H` (dimensioni), `--seed`,
`-v` (log dettagliati). Elenco completo: `sd-cli --help`.

### 2.5 Primo utilizzo: il server HTTP

```shell
./build/bin/sd-server -m modello.safetensors --api-key mia-chiave
```

Da un client qualsiasi:

```shell
curl -s http://127.0.0.1:1234/sdcpp/v1/capabilities -H "X-API-Key: mia-chiave"

curl -s -X POST http://127.0.0.1:1234/sdcpp/v1/img_gen \
     -H "Content-Type: application/json" -H "X-API-Key: mia-chiave" \
     -d '{"prompt": "un gatto astronauta", "width": 512, "height": 512}'
```

Sono disponibili anche endpoint compatibili OpenAI (`/v1/images/generations`)
e AUTOMATIC1111 (`/sdapi/v1/txt2img`). Dettagli di sicurezza in sezione 5.

### Pacchetto portabile (consigliato per la distribuzione)
```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
# su x86 aggiungere -DGGML_CPU_ALL_VARIANTS=ON per una build CPU per ogni livello (AVX, AVX2, ...)
cmake --build build -j
```

Distribuire insieme all'eseguibile i moduli `libggml-*` generati: i backend
non utilizzabili sulla macchina vengono semplicemente ignorati. L'opzione
`SD_BUILD_SHARED_GGML_LIB=ON` è necessaria: senza di essa il CMake del
progetto forza ggml statico.

Per creare lo zip distribuibile (binari + moduli backend + licenze +
LEGGIMI) da una build portabile:

```shell
./scripts/package_portable.sh build    # risultato in dist/
```

Il pacchetto è rilocabile: si scompatta e si usa, senza installazione.
Il workflow GitHub Actions `release-portable.yml` produce automaticamente gli
zip per Linux x86-64, macOS arm64 e Windows x86-64 (con varianti CPU
SSE/AVX/AVX2/AVX-512 su x86 per le macchine più vecchie) a ogni tag
`portable-v*` o avvio manuale.

## 3. Come collegarlo a un progetto

### 3.1 Libreria C/C++ con CMake

```cmake
cmake_minimum_required(VERSION 3.14)
project(mia_app C CXX)

find_package(stable-diffusion REQUIRED)
add_executable(mia_app main.c)
target_link_libraries(mia_app PRIVATE stable-diffusion)
set_target_properties(mia_app PROPERTIES LINKER_LANGUAGE CXX)
```

Configurare con `-DCMAKE_PREFIX_PATH=/percorso/di/installazione`.

### 3.2 Sequenza di inizializzazione consigliata

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
  valorizzare i campi.
- Controllare sempre il valore di ritorno di `new_sd_ctx()` e `generate_image()`.
- Un contesto non è thread-safe: serializzare gli accessi (come fa `sd-server`
  con un mutex) oppure usare un contesto per thread. Le funzioni di
  enumerazione dei device sono state provate da più thread in parallelo.
- Le stringhe passate nei parametri (`model_path`, ...) devono restare valide
  finché serve il contesto.

### 3.3 Altri linguaggi (TypeScript/Electron, Python, Rust, Go, C#, ...)

L'API è C pura (`include/stable-diffusion.h`), quindi si può usare tramite FFI
o binding. Binding esistenti sono elencati nel [README](../README.md). Per
un'app Electron/Node: modulo nativo (N-API) oppure processo separato (3.4/3.5).

### 3.4 Come processo: `sd-cli`

Isola il motore dall'applicazione: un crash del motore non abbatte l'app.
Utile: `sd-cli --list-devices`, `--memory-guard 90`, `--backend`,
`--max-vram`, `--disable-backend-fallback`.

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

- **Macchine senza GPU o con GPU non funzionante**: fallback automatico alla
  CPU. Per modelli grandi su macchine piccole combinare `--offload-to-cpu`,
  `--params-backend disk` e `--max-vram` (vedi [performance.md](./performance.md)).
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
