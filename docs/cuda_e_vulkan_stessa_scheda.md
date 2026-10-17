# Una scheda, due nomi: `CUDA0` e `Vulkan0`

Con i moduli CUDA e Vulkan nella stessa cartella (lo zip `...-bin-win-cuda12-vulkan-x64.zip` della
release, o due zip uniti a mano) una scheda NVIDIA compare **due volte** nell'elenco del motore: è una
scheda sola, raggiunta da due backend. Questa guida spiega cosa fa il motore, perché la scelta del
backend conta molto per i tempi, cosa dice il log e come evitare il caso più costoso: una generazione
fissata su Vulkan sulla scheda che CUDA già serve.

Il resto della convivenza fra applicazioni è in [convivenza_applicazioni.md](./convivenza_applicazioni.md);
le opzioni `--backend` e `--multi-gpu` in [backend.md](./backend.md).

## In breve

- `CUDA0` e `Vulkan0` con la stessa descrizione e lo stesso indirizzo PCI sono **la stessa scheda**. Il
  motore la conta una volta e, senza `--backend`, usa CUDA. Nel piano la riga di `Vulkan0` dice
  `same GPU as CUDA0, not used automatically`.
- Chi nomina `Vulkan0` (`--backend vulkan0`, o un programma che avvia il motore e lo scrive al posto tuo)
  ottiene Vulkan: il motore rispetta sempre un `--backend` esplicito. Sulla stessa scheda Vulkan può essere
  molto più lento di CUDA. **Dalle versioni con questo avviso** il log lo dice, una volta per scheda.
- Un secondo avviso copre l'altro costo che si era visto insieme al primo: pesi che non stanno né sulla
  scheda né nella RAM disponibile restano nel file del modello e vengono riletti a ogni passo.
- Per usare la scheda NVIDIA: **non scrivere `--backend vulkan0`**, oppure scrivi `--backend cuda0`. Per
  un programma che avvia il motore vale lo stesso: non fissare il backend di una scheda che CUDA elenca.

## Come si presenta nel log

L'elenco del motore, ridotto, di un portatile con RTX 3060 e GPU integrata AMD (`sd-cli --list-devices`):

```text
CUDA0   gpu     NVIDIA GeForce RTX 3060 Laptop GPU      5.01 GiB free / 6.00 GiB total
Vulkan0 gpu     NVIDIA GeForce RTX 3060 Laptop GPU      5.01 GiB free / 5.85 GiB total
Vulkan1 igpu    AMD Radeon(TM) Graphics                 7.53 GiB free / 7.92 GiB total
CPU     cpu     (modello del processore)                8.78 GiB free / 15.35 GiB total
```

`CUDA0` e `Vulkan0` hanno la stessa descrizione; i totali differiscono un poco (6,00 e 5,85 GiB), quindi la
memoria non basta a dire che sono la stessa scheda. Il piano di auto-fit la riconosce dall'indirizzo PCI:

```text
auto-fit plan:
  devices:
    CUDA0        NVIDIA GeForce RTX 3060 Laptop GPU free   4938 MiB, budget   4426 MiB
    Vulkan0      NVIDIA GeForce RTX 3060 Laptop GPU free   4938 MiB, budget   4426 MiB (same GPU as CUDA0, not used automatically)
    Vulkan1      AMD Radeon(TM) Graphics          free   7706 MiB, budget   7194 MiB (shares the system memory)
```

Se in un log vedi invece, in fondo al piano, una riga come

```text
DiT          params   4897 MiB, compute reserve  2048 MiB -> compute Vulkan0, params disk
auto-fit: --backend "vulkan0" --params-backend "diffusion=disk,te=disk,vae=disk"
```

la scheda NVIDIA sta lavorando attraverso Vulkan (`compute Vulkan0`), e qualcuno ha chiesto `vulkan0`.

## L'avviso per un backend fissato sul gemello

Quando un modulo gira su un elenco "all'ombra" di un altro, il motore lo lascia fare e scrive, dopo il
piano, una riga per ogni scheda interessata (una sola anche se i moduli sono più d'uno):

```text
[W] auto-fit: Vulkan0 is the same GPU as CUDA0; the automatic choice uses CUDA0 for this card. Backends on a generic graphics API (Vulkan, OpenCL) are usually slower than the one the GPU vendor provides: name CUDA0 in --backend, or leave --backend out, unless Vulkan0 is wanted.
```

La seconda frase compare solo quando il backend che tiene la scheda è del produttore e quello nominato è
Vulkan o OpenCL. Con due elenchi dello stesso backend (`Vulkan1` all'ombra di `Vulkan0`) c'è solo la prima
frase: nessuno dei due è più lento dell'altro. Con un backend che non riporta l'indirizzo PCI il motore non
può sapere che due elenchi sono la stessa scheda, e non avvisa.

L'avviso non cambia nulla del piano: è una riga di log. La scelta resta di chi ha scritto `--backend`.

## L'avviso per i pesi che restano su disco

Il piano elenca per ogni modulo dove stanno i pesi: scheda, RAM, un'altra scheda, oppure `disk`. Con
`params disk` i pesi restano nel file del modello e vengono riletti **ogni volta che il modulo gira**, il
modello di diffusione a ogni passo di campionamento. Dalle versioni con questo avviso il motore lo dice, con le
dimensioni e con la RAM che ha trovato:

```text
[W] auto-fit: the weights of DiT, Conditioner, VAE (6615 MiB) stay on disk and are read again every time they run, the diffusion model at every sampling step: free RAM is 469 MiB and the budget for weights 0 MiB. Close other programs to free RAM, or shrink the weights with --type (for example q8_0) or a quantized model file, to keep them in memory.
```

(I numeri sono quelli del piano di un log reale; la riga stessa è nuova.) Non compare quando la RAM libera è
sconosciuta, quando nessun peso resta su disco, né con `--cpu-only`, che ha già la sua nota.

## Numeri misurati su un portatile

Lavoro: Pony Diffusion V6 XL, 1024×1024, 25 passi, stesso modello e stessa misura. Portatile con RTX 3060
Laptop (6 GiB), GPU integrata AMD e 16 GB di RAM. Sono misure di **un** computer, non promesse.

| Fase           | `CUDA0`, RAM libera (~9 GiB) | `Vulkan0`, RAM quasi piena (469 MiB liberi) |
| -------------- | ---------------------------- | ------------------------------------------- |
| campionamento  | 57 s                         | 899 s                                       |
| decodifica VAE | 5,4 s                        | 120 s                                       |
| totale         | 64 s                         | 1022 s (17 minuti)                          |

La differenza **non è tutta del backend**: nel giro su Vulkan0 la RAM era quasi piena e i pesi venivano riletti
dal disco a ogni passo (circa 6 s su 36 s per passo). Le due cose si sommano, e i due avvisi esistono perché
in quel log nessuna riga nominava né l'una né l'altra. Il rapporto fra le due colonne è circa 16 sul totale: non
è il rapporto fra i due backend a parità di condizioni, che questa prova non misura.

## Cosa fare

**A mano.** Non scrivere `--backend vulkan0` per la scheda NVIDIA. Senza `--backend` il motore sceglie
da solo; per una scheda sola scrivi `--backend cuda0`.

**Da un altro programma.** Chi avvia il motore deve poter nominare un backend solo se sa che quell'elenco è
quello giusto. Un programma che legge `--list-devices` e vede `CUDA0` e `Vulkan0` con la stessa descrizione deve
trattarli come una scheda e tenere CUDA (è quello che fa ora Aletheia: repository Aletheia-Agape, file
`GPU_CUDA_VULKAN.md`). Non fissare il backend quando non serve: con `--auto-fit` il motore sceglie da sé la
scheda dedicata con più memoria libera e, per la scheda doppia, tiene CUDA.

**Prova rapida, senza cambiare il programma che avvia il motore.** Nascondi il modulo Vulkan: il motore elenca
solo `CUDA0` e `CPU` (si perde anche la GPU integrata, che qui non serve). In PowerShell, nella cartella del
motore:

```powershell
Rename-Item ggml-vulkan.dll ggml-vulkan.dll.off     # il motore non carica più Vulkan
# ...prova...
Rename-Item ggml-vulkan.dll.off ggml-vulkan.dll     # rimette tutto a posto
```

**Quanta RAM è libera e chi la occupa**, prima di una generazione grande (PowerShell):

```powershell
$os = Get-CimInstance Win32_OperatingSystem
"RAM libera: {0:N1} GiB su {1:N1} GiB" -f ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB)
Get-Process | Group-Object Name | ForEach-Object { [pscustomobject]@{ Nome = $_.Name; MiB = [int](($_.Group | Measure-Object WorkingSet64 -Sum).Sum / 1MB); Processi = $_.Count } } | Sort-Object MiB -Descending | Select-Object -First 8 | ForEach-Object { "{0,-26} {1,6} MiB  ({2} processi)" -f $_.Nome, $_.MiB, $_.Processi }
```

Se i primi della lista sono browser, chat locali o altri programmi pesanti, chiuderli libera la RAM che il
piano conta nel «budget per i pesi». Se non basta, `--type q8_0` (o un modello già quantizzato) rimpicciolisce i
pesi, vedi [quantization_and_gguf.md](./quantization_and_gguf.md).

## Verificato e non verificato

Verificato:

- Il riconoscimento della scheda doppia su una macchina reale: nel log di un portatile con RTX 3060 il piano
  scrive `same GPU as CUDA0, not used automatically` per `Vulkan0`, e non per la GPU integrata `Vulkan1`.
- Che un `--backend "vulkan0"` fissato da un programma esterno produce l'esecuzione su Vulkan (`compute
  Vulkan0`) e la durata di 17 minuti riportata sopra.
- Le funzioni dei due avvisi con dispositivi simulati, in `test/backend_fit_plan_test.cpp`
  (`check_plan_warnings`): un avviso per scheda, il testo, nessun avviso se la scheda gira sul suo elenco,
  nessun avviso senza indirizzo PCI, due schede uguali elencate due volte, due elenchi dello stesso backend, i
  pesi su disco con la RAM piena, nessun avviso con RAM sconosciuta.

Non verificato:

- La comparsa dei due avvisi in un'esecuzione reale su Windows: servono un binario costruito da questo codice
  e una macchina con CUDA e Vulkan.
- La differenza di velocità fra CUDA e Vulkan **a parità di RAM libera** sulla stessa scheda.
- Backend diversi da CUDA e Vulkan (Metal, SYCL, HIP): la logica dell'ordine si basa su `registry_rank`
  (Vulkan e OpenCL dopo gli altri) ed è provata solo con dispositivi simulati.

## Dove sta nel codice

- `src/core/backend_fit.cpp`: `registry_rank`, `mark_shadowed_devices` (la scheda doppia),
  `shadowed_runtime_warnings` e `disk_streaming_warning` (i due avvisi), `derive_backend_specs` (dove vengono
  scritti).
- `test/backend_fit_plan_test.cpp`: `check_mixed_backends`, `check_plan_warnings`.
- Nel repository Aletheia-Agape: `GPU_CUDA_VULKAN.md`, `src/lib/compute-devices.ts` (`mergeSameCard`).
