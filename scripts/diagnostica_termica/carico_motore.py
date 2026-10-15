#!/usr/bin/env python3
"""Fa generare sd-server di continuo per un tempo stabilito: il carico della fase MOTORE di termico.py.

Manda un job dopo l'altro (sempre lo stesso: stessa immagine, stesso seme, stessi passi), così due prove
fatte in momenti diversi o su PC diversi fanno lo stesso lavoro e si possono confrontare. Alla scadenza del
tempo, o con Ctrl+C, ferma il job in corso con POST /sdcpp/v1/jobs/{id}/cancel e stampa quanti ne ha fatti.

Non salva le immagini e non cambia nessuna impostazione del server. Serve Python 3.8 o più recente.

  python3 carico_motore.py --durata 300
  python3 carico_motore.py --url http://127.0.0.1:1234 --chiave-api LA_TUA_CHIAVE --larghezza 768 --altezza 768
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ATTESA_MASSIMA_SERVER = 120  # secondi di Retry-After che accettiamo di aspettare
ATTESA_SENZA_RISPOSTA = 3.0  # secondi fra due tentativi quando il server non risponde (5 tentativi, poi ci si ferma)


def chiama(url, chiave, metodo, percorso, corpo=None, timeout=60):
    """(codice HTTP, corpo JSON o None, intestazioni); codice None se il server non risponde."""
    dati = None if corpo is None else json.dumps(corpo).encode("utf-8")
    richiesta = urllib.request.Request(url + percorso, data=dati, method=metodo)
    richiesta.add_header("Accept", "application/json")
    if dati is not None:
        richiesta.add_header("Content-Type", "application/json")
    if chiave:
        richiesta.add_header("Authorization", "Bearer " + chiave)
    apri = urllib.request.build_opener(urllib.request.ProxyHandler({})).open  # il motore è di solito su questo PC
    try:
        with apri(richiesta, timeout=timeout) as risposta:
            return risposta.status, json.loads(risposta.read().decode("utf-8", "replace")), risposta.headers
    except urllib.error.HTTPError as errore:
        try:
            return errore.code, json.loads(errore.read().decode("utf-8", "replace")), errore.headers
        except (ValueError, OSError):
            return errore.code, None, errore.headers
    except (urllib.error.URLError, OSError, ValueError):
        return None, None, None


def messaggio_errore(corpo):
    errore = corpo.get("error") if isinstance(corpo, dict) else None
    if isinstance(errore, dict):
        return errore.get("message") or str(errore)
    return str(errore) if errore else "nessun dettaglio"


class Carico:
    def __init__(self, args):
        self.args = args
        self.url = args.url.strip().rstrip("/")
        if "://" not in self.url:
            self.url = "http://" + self.url
        self.fine = time.monotonic() + args.durata
        self.job_corrente = None
        self.completati = []   # secondi di ogni job finito
        self.interrotti = 0
        self.falliti = 0

    def richiesta(self):
        return {"prompt": self.args.prompt, "width": self.args.larghezza, "height": self.args.altezza, "seed": self.args.seme,
                "batch_count": 1, "sample_params": {"sample_steps": self.args.passi}}

    def attendi(self, secondi):
        """Dorme fino a `secondi`, ma non oltre la scadenza. False se la scadenza è arrivata."""
        fine_attesa = min(time.monotonic() + secondi, self.fine)
        while time.monotonic() < fine_attesa:
            time.sleep(min(0.2, fine_attesa - time.monotonic()))
        return time.monotonic() < self.fine

    def ferma_job_corrente(self):
        if not self.job_corrente:
            return False
        codice, corpo, _ = chiama(self.url, self.args.chiave_api, "POST", f"/sdcpp/v1/jobs/{self.job_corrente}/cancel?wait=10")
        self.job_corrente = None
        return codice in (200, 202)

    def un_job(self):
        """Fa generare un job. Restituisce 'completato', 'interrotto', 'fallito' o 'scadenza' (non è partito)."""
        senza_risposta = 0
        while True:
            if time.monotonic() >= self.fine:
                return "scadenza"
            codice, corpo, intestazioni = chiama(self.url, self.args.chiave_api, "POST", "/sdcpp/v1/img_gen", self.richiesta())
            if codice == 202 and isinstance(corpo, dict) and corpo.get("id"):
                break
            if codice is None:
                senza_risposta += 1
                print(f"  il server non risponde ({senza_risposta}/5)", flush=True)
                if senza_risposta >= 5:
                    sys.exit("Il server non risponde: controlla che sd-server sia acceso e l'indirizzo giusto.")
                self.attendi(ATTESA_SENZA_RISPOSTA)
                continue
            if codice == 401:
                sys.exit("Il server vuole la chiave API: passala con --chiave-api (o la variabile TERMICO_CHIAVE_API).")
            if codice in (429, 503):
                try:
                    attesa = min(float(intestazioni.get("Retry-After", 10)), ATTESA_MASSIMA_SERVER)
                except (TypeError, ValueError):
                    attesa = 10.0
                print(f"  il server chiede di aspettare {attesa:.0f} s (HTTP {codice})", flush=True)
                self.attendi(attesa)
                continue
            sys.exit(f"Il server ha rifiutato la richiesta (HTTP {codice}): {messaggio_errore(corpo)}")
        self.job_corrente = corpo["id"]
        percorso = corpo.get("poll_url") or f"/sdcpp/v1/jobs/{corpo['id']}"
        avvio = time.monotonic()
        while True:
            if time.monotonic() >= self.fine:
                self.ferma_job_corrente()
                self.interrotti += 1
                return "interrotto"
            codice, job, _ = chiama(self.url, self.args.chiave_api, "GET", percorso)
            stato = job.get("status") if isinstance(job, dict) else None
            if stato == "completed":
                self.job_corrente = None
                self.completati.append(time.monotonic() - avvio)
                return "completato"
            if stato in ("failed", "cancelled"):
                self.job_corrente = None
                self.falliti += 1
                errore = job.get("error") or {}
                print(f"  job {stato}: {errore.get('message', 'senza dettagli')}", flush=True)
                return "fallito"
            time.sleep(1.0)

    def esegui(self):
        print(f"Carico su {self.url}: {self.args.larghezza}x{self.args.altezza}, {self.args.passi} passi, "
              f"seme {self.args.seme}, per {self.args.durata} secondi. Ctrl+C per fermare.", flush=True)
        inizio = time.monotonic()
        numero = 0
        try:
            while True:
                esito = self.un_job()
                if esito == "scadenza":
                    break
                numero += 1
                if esito == "completato":
                    print(f"job {numero}: completato in {self.completati[-1]:.1f} s", flush=True)
                elif esito == "interrotto":
                    print(f"job {numero}: fermato alla scadenza del tempo", flush=True)
                if self.falliti >= 3:
                    print("Tre job falliti: mi fermo.", flush=True)
                    break
        except KeyboardInterrupt:
            print("\nInterrotto: fermo il job in corso.", flush=True)
            if self.ferma_job_corrente():
                self.interrotti += 1
        finally:
            durata = time.monotonic() - inizio
            medio = f"{sum(self.completati) / len(self.completati):.1f} s" if self.completati else "n/d"
            print(f"\nFinito dopo {durata:.0f} s: {len(self.completati)} job completati (media {medio}), "
                  f"{self.interrotti} fermati, {self.falliti} falliti.", flush=True)
        return 0 if self.completati and not self.falliti else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Non salva")[0].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:1234", help="indirizzo di sd-server (default http://127.0.0.1:1234)")
    ap.add_argument("--chiave-api", dest="chiave_api", default=os.environ.get("TERMICO_CHIAVE_API"),
                    help="chiave API di sd-server (oppure variabile TERMICO_CHIAVE_API)")
    ap.add_argument("--durata", type=int, default=180, help="secondi di carico (default 180)")
    ap.add_argument("--larghezza", type=int, default=512, help="larghezza dell'immagine (default 512)")
    ap.add_argument("--altezza", type=int, default=512, help="altezza dell'immagine (default 512)")
    ap.add_argument("--passi", type=int, default=20, help="passi di campionamento (default 20)")
    ap.add_argument("--seme", type=int, default=1, help="seme: lo stesso a ogni job, così il lavoro è sempre uguale (default 1)")
    ap.add_argument("--prompt", default="a lighthouse on a cliff at sunset, detailed oil painting", help="testo della richiesta")
    args = ap.parse_args(argv)
    if args.durata < 1 or min(args.larghezza, args.altezza) < 64 or args.passi < 1:
        ap.error("durata, passi e dimensioni (almeno 64) devono avere senso")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    return Carico(args).esegui()


if __name__ == "__main__":
    sys.exit(main())
