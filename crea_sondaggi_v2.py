"""
Crea nell'app "Sondaggi tesi" i dieci sondaggi di valutazione sulle immagini
prodotte dalla pipeline (Risultati/<Soggetto>/16colori), uno per soggetto.

Differenze rispetto alla creazione dall'interfaccia dell'app:
  - le immagini restano PNG (l'app le convertirebbe in JPEG, che altera i
    bordi netti e i colori della pixel art);
  - l'immagine mostrata e' ingrandita x4 per replicazione (512x512), cosi' il
    browser la visualizza senza ridimensionarla; toccandola si apre la
    versione x8 (1024x1024), identica a quella fornita al modello
    visione-linguaggio;
  - sono incluse soltanto le immagini ammesse dalla validazione dello sfondo;
  - l'ordine delle immagini e' casuale per ogni partecipante.

Il programma:
  1. verifica che il server dell'app sia spento (altrimenti le modifiche
     verrebbero sovrascritte);
  2. salva una copia di data/surveys.json;
  3. chiude i sondaggi precedenti (restano consultabili, dati intatti);
  4. crea i dieci sondaggi e scrive sondaggi_v2.json con la corrispondenza
     fra domande e immagini, usata da esperimento_strategie.py.

Uso:  python crea_sondaggi_v2.py
      python crea_sondaggi_v2.py --ricrea     (elimina e ricrea i sondaggi v2)
"""

import os
import argparse
import csv
import json
import random
import shutil
import string
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
PIPELINE = Path(__file__).resolve().parent
RISULTATI = PIPELINE / "Risultati"
APP = Path(os.environ.get("SONDAGGI_TESI", Path(__file__).resolve().parent.parent / "Sondaggi tesi"))
DB = APP / "data" / "surveys.json"
UPLOADS = APP / "uploads"
MAPPA = PIPELINE / "sondaggi_v2.json"

NOMI = {
    "Astronauta": "Astronauta",
    "Ballerina": "Ballerina",
    "Cane": "Golden Retriever",
    "Cat": "Gatto arancione",
    "Cowboy": "Cowboy",
    "Cuoco": "Cuoco",
    "Gnomo": "Gnomo",
    "Pirata": "Pirata",
    "Principessa": "Principessa medievale",
    "Samurai": "Samurai",
}

DESCRIZIONE = ('Date un voto da 1 a 5 a ciascuna immagine in base a quanto è fedele alla seguente '
               'descrizione: "{desiderata}" E in base alla qualità della pixel art, a quanto è fatta '
               'bene e se la utilizzereste per un progetto personale (videogioco, animazione ecc.). '
               'Toccando un\'immagine la si vede ingrandita.')


def nuovo_id():
    """Stesso formato di store.newId(): 6 caratteri casuali + 4 dal tempo."""
    alfabeto = string.digits + string.ascii_lowercase
    casuale = "".join(random.choice(alfabeto) for _ in range(6))
    t, s = int(time.time() * 1000), ""
    while t:
        t, r = divmod(t, 36)
        s = alfabeto[r] + s
    return casuale + s[-4:]


def server_attivo():
    try:
        urllib.request.urlopen("http://127.0.0.1:3000/api/config", timeout=2)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Crea i sondaggi v2 nell'app Sondaggi tesi.")
    ap.add_argument("--ricrea", action="store_true", help="elimina i sondaggi v2 esistenti e li ricrea")
    args = ap.parse_args()

    if server_attivo():
        sys.exit("Il server dell'app e' acceso: chiudi la finestra nera di AVVIA.bat e rilancia.")

    db = json.loads(DB.read_text(encoding="utf-8"))
    copia = DB.with_name(f"surveys.backup-{datetime.now():%Y%m%d-%H%M%S}.json")
    shutil.copy2(DB, copia)
    print(f"Copia di sicurezza: {copia.name}")

    if MAPPA.exists():
        vecchia = json.loads(MAPPA.read_text(encoding="utf-8"))
        ids_v2 = {v["id"] for v in vecchia.values()}
        presenti = [s for s in db["surveys"] if s["id"] in ids_v2]
        if presenti and not args.ricrea:
            sys.exit("I sondaggi v2 esistono gia'. Usa --ricrea per eliminarli e ricrearli "
                     "(le eventuali risposte gia' raccolte andrebbero perse).")
        for s in presenti:
            shutil.rmtree(UPLOADS / s["id"], ignore_errors=True)
            (APP / "data" / "responses" / f"{s['id']}.jsonl").unlink(missing_ok=True)
        db["surveys"] = [s for s in db["surveys"] if s["id"] not in ids_v2]

    for s in db["surveys"]:
        if not s.get("closed"):
            s["closed"] = True
            print(f"Chiuso il sondaggio precedente: {s['title']}")

    desiderata = json.loads((PIPELINE / "desiderata.json").read_text(encoding="utf-8"))
    mappa = {}
    adesso = datetime.now(timezone.utc)
    for k, (soggetto, nome) in enumerate(NOMI.items()):
        cartella = RISULTATI / soggetto
        post = list(csv.DictReader(open(cartella / "03_post_elaborazione.csv", encoding="utf-8"), delimiter=";"))
        ammesse = [r["file"] for r in post if r["ammessa"] == "1"]

        sid = nuovo_id()
        base_dir = UPLOADS / sid
        for sotto in ("orig", "view", "thumb"):
            (base_dir / sotto).mkdir(parents=True, exist_ok=True)

        domande, corrispondenza = [], {}
        for f in ammesse:
            qid = nuovo_id()
            nome_file = f"{qid}_{Path(f).stem}.png"
            logica = Image.open(cartella / "16colori" / f).convert("RGB")
            w, h = logica.size
            logica.resize((w * 8, h * 8), Image.NEAREST).save(base_dir / "orig" / nome_file)
            logica.resize((w * 4, h * 4), Image.NEAREST).save(base_dir / "view" / nome_file)
            logica.resize((w * 2, h * 2), Image.NEAREST).save(base_dir / "thumb" / nome_file)
            url = lambda sotto: f"/uploads/{sid}/{sotto}/{quote(nome_file)}"
            domande.append({"id": qid, "label": Path(f).stem, "file": f,
                            "orig": url("orig"), "view": url("view"), "thumb": url("thumb"),
                            "width": w * 8, "height": h * 8})
            corrispondenza[qid] = f

        sondaggio = {
            "id": sid,
            "title": f"Valutazione sprite: {nome}",
            "description": DESCRIZIONE.format(desiderata=desiderata[soggetto])[:1000],
            # istanti distinti, cosi' l'elenco dell'app li mostra in ordine fisso
            "createdAt": adesso.replace(microsecond=0).isoformat().replace("+00:00", f".{k:03d}Z"),
            "scaleMin": 1, "scaleMax": 5,
            "labelMin": "Per niente", "labelMax": "Moltissimo",
            "askName": True, "allowComments": False, "shuffle": True, "closed": False,
            "questions": domande,
        }
        db["surveys"].append(sondaggio)
        mappa[soggetto] = {"id": sid, "titolo": sondaggio["title"], "domande": corrispondenza}
        print(f"Creato: {sondaggio['title']:<40} {len(domande)} immagini  (id {sid})")

    tmp = DB.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DB)
    MAPPA.write_text(json.dumps(mappa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCorrispondenza domande-immagini salvata in {MAPPA.name}")
    print("Avvia l'app con AVVIA.bat: i nuovi sondaggi compaiono nell'elenco.")


if __name__ == "__main__":
    main()
