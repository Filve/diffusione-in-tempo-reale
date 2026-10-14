# Test di robustezza e sicurezza (fork)

Script locali di questo fork per verificare che il motore fallisca in modo
pulito con input ostili, senza crash né errori di memoria. Istruzioni complete
in italiano: [docs/integrazione_progetti.md](../docs/integrazione_progetti.md),
sezione "Sicurezza".

| Test | Cosa verifica | Come si esegue |
|---|---|---|
| `security_smoke.sh` | file modello ostili + opzioni CLI estreme → mai crash (exit < 128) | `./test/security_smoke.sh build/bin/sd-cli` |
| `api_robustness.cpp` | API C: puntatori nulli, indici fuori range, buffer piccoli, stress multi-thread (con ASan/UBSan) | vedi intestazione del file |
| `pe_imports_test.cpp` | lettore della tabella di importazione dei file PE (diagnostica delle DLL mancanti su Windows): file validi, troncati e corrotti (con ASan/UBSan) | vedi intestazione del file |
| `file_handle_pool_test.cpp` | pool dei flussi aperti sui file modello: riuso, file modificato o sostituito, flussi in errore, limiti, uso concorrente (con ASan/UBSan e TSan) | vedi intestazione del file |
| `profile_test.cpp` | profili di risorse (`--profile`): scelta automatica da dimensione del modello e memoria (anche sommando la memoria di più GPU), ordine dei livelli, opzioni esplicite che vincono sul profilo | vedi intestazione del file |
| `backend_fit_plan_test.cpp` | pianificatore multi-GPU dell'auto-fit (`--multi-gpu auto\|on\|off`) con GPU simulate: quale modulo va su quale GPU, quando il modello si divide per blocchi, nessun device riceve più pesi del suo budget, i moduli assegnati dall'utente non si muovono (include `src/core/backend_fit.cpp`, quindi serve la libreria compilata; nessuna GPU né modello) | vedi intestazione del file |
| `server_recovery_test.cpp` | custode del contesto del server (`examples/server/recovery.*`): scala dei profili, discesa su memoria insufficiente, risalita dopo una pausa, ricarica dopo un errore, risposte HTTP 503/400/500 con `Retry-After`, uso da più thread. Annullamento di una generazione in corso: lo stop raggiunge solo il job che lo ha chiesto, un job annullato prima di partire non genera nulla, non costa nessuna ricarica del modello e non conta come errore. Non serve nessun modello: libreria e orologio sono simulati (con ASan/UBSan e TSan) | vedi intestazione del file |
| `server_cancel_e2e.py` | annullamento di un job in corso su un vero `sd-server` (serve un modello qualsiasi, su qualunque backend): job fermato, richiesta ripetuta, job in coda, `?wait=0`, richiesta sincrona in attesa che poi completa, job già finito, id sconosciuto, `wait` non valido, contatori di `/sdcpp/v1/health` e log senza righe di errore | `python3 test/server_cancel_e2e.py --server build/bin/sd-server --model <modello>` |
| `fuzz_model_loader.py` | fuzzing a mutazione del caricamento modelli (safetensors/GGUF) | `python3 test/fuzz_model_loader.py <sd-cli> --seconds 300` |

Per il fuzzing è consigliata una build con AddressSanitizer:

```shell
cmake -B build_asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSD_WEBP=OFF -DSD_WEBM=OFF \
      -DCMAKE_C_FLAGS=-fsanitize=address -DCMAKE_CXX_FLAGS=-fsanitize=address \
      -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address
cmake --build build_asan -j --target sd-cli
python3 test/fuzz_model_loader.py build_asan/bin/sd-cli --seconds 600
```

Gli input che causano una scoperta vengono salvati in `test/fuzz_findings/`
(da non committare). Questi test riducono il rischio ma non dimostrano
l'assenza di vulnerabilità.
