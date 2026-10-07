# Test di robustezza e sicurezza (fork)

Script locali di questo fork per verificare che il motore fallisca in modo
pulito con input ostili, senza crash né errori di memoria. Istruzioni complete
in italiano: [docs/integrazione_progetti.md](../docs/integrazione_progetti.md),
sezione "Sicurezza".

| Test | Cosa verifica | Come si esegue |
|---|---|---|
| `security_smoke.sh` | file modello ostili + opzioni CLI estreme → mai crash (exit < 128) | `./test/security_smoke.sh build/bin/sd-cli` |
| `api_robustness.cpp` | API C: puntatori nulli, indici fuori range, buffer piccoli, stress multi-thread (con ASan/UBSan) | vedi intestazione del file |
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
