"""
Crea nell'app "Sondaggi tesi" i dieci questionari sulle animazioni (Capitolo 5).

Per ogni soggetto un questionario con sei animazioni (livelli L1-L3, versioni A e B)
in ordine casuale per valutatore. Ogni animazione e' mostrata accanto allo sprite di
partenza, con il movimento richiesto, e viene votata su cinque aspetti con una
scala da 1 a 5 il cui significato e' descritto livello per livello.
I nomi dei file sono neutri: il valutatore non puo' sapere quale versione stia
guardando.

Prima dell'esecuzione il server dell'app deve essere spento.
Uso:  python crea_sondaggi_video.py            (--ricrea per rifarli)
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from PIL import Image

import crea_sondaggi_v2 as cs

sys.stdout.reconfigure(encoding="utf-8")
PIPELINE = Path(__file__).resolve().parent
VIDEO = PIPELINE / "Risultati_video"
MAPPA_VIDEO = PIPELINE / "sondaggi_video.json"

CRITERI = [
    {"id": "C1", "label": "Identità del personaggio",
     "question": "Il personaggio resta lo stesso dello sprite di partenza?",
     "levels": ["Diventa un altro personaggio o è irriconoscibile",
                "Riconoscibile, ma con cambiamenti evidenti (colori, abiti, proporzioni)",
                "Stesso personaggio, ma alcuni dettagli cambiano o scompaiono",
                "Stesso personaggio, con solo piccole variazioni occasionali",
                "Identico allo sprite di partenza in tutti i fotogrammi"]},
    {"id": "C2", "label": "Movimento richiesto",
     "question": "L'animazione esegue il movimento indicato sopra?",
     "levels": ["Nessun movimento, oppure un movimento del tutto diverso",
                "Un movimento solo vagamente collegato a quello richiesto",
                "Il movimento si riconosce, ma è incompleto o impreciso",
                "Movimento corretto, con piccole imprecisioni",
                "Movimento eseguito esattamente come descritto"]},
    {"id": "C3", "label": "Fluidità e leggibilità",
     "question": "Il movimento è comprensibile e privo di deformazioni?",
     "levels": ["Caotico: deformazioni, parti del corpo che compaiono o spariscono",
                "Scatti o deformazioni frequenti",
                "Comprensibile, ma con alcuni scatti o deformazioni",
                "Fluido e leggibile, con rari difetti",
                "Fluido, naturale e perfettamente leggibile"]},
    {"id": "C4", "label": "Stabilità dell'immagine",
     "question": "Ci sono sfarfallii, pixel che cambiano da soli o uno sfondo instabile?",
     "levels": ["Sfarfallio continuo su tutta l'immagine",
                "Sfarfallio forte su ampie zone",
                "Sfarfallio visibile in alcune zone",
                "Lievi instabilità, notate solo guardando con attenzione",
                "Immagine perfettamente stabile: cambia solo ciò che si muove"]},
    {"id": "C5", "label": "Pixel art e utilizzabilità",
     "question": "È un'animazione in pixel art che useresti in un gioco?",
     "levels": ["Non sembra pixel art (sfocata, effetto 3D); inutilizzabile",
                "Sembra pixel art solo a tratti; andrebbe rifatta",
                "Pixel art accettabile, ma servirebbero ritocchi importanti",
                "Buona pixel art, utilizzabile con piccoli ritocchi",
                "Animazione in pixel art pronta per essere usata in un gioco"]},
]

DESCRIZIONE = ('Guarda ogni animazione (si ripete in loop) confrontandola con lo sprite di partenza e '
               'vota i cinque aspetti indicati. Il movimento richiesto è scritto sopra ogni animazione. '
               'Il personaggio era stato richiesto così: "{desiderata}"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ricrea", action="store_true")
    args = ap.parse_args()
    if cs.server_attivo():
        sys.exit("Il server dell'app e' acceso: chiudilo e rilancia.")

    db = json.loads(cs.DB.read_text(encoding="utf-8"))
    copia = cs.DB.with_name(f"surveys.backup-{datetime.now():%Y%m%d-%H%M%S}.json")
    shutil.copy2(cs.DB, copia)
    print(f"Copia di sicurezza: {copia.name}")

    if MAPPA_VIDEO.exists():
        ids = {v["id"] for v in json.loads(MAPPA_VIDEO.read_text(encoding="utf-8")).values()}
        presenti = [s for s in db["surveys"] if s["id"] in ids]
        if presenti and not args.ricrea:
            sys.exit("I questionari sulle animazioni esistono gia'. Usa --ricrea per rifarli.")
        for s in presenti:
            shutil.rmtree(cs.UPLOADS / s["id"], ignore_errors=True)
            (cs.APP / "data" / "responses" / f"{s['id']}.jsonl").unlink(missing_ok=True)
        db["surveys"] = [s for s in db["surveys"] if s["id"] not in ids]

    for s in db["surveys"]:
        if not s.get("closed"):
            s["closed"] = True
            print(f"Chiuso: {s['title']}")

    sprite = json.loads((VIDEO / "sprite_scelti.json").read_text(encoding="utf-8"))
    desiderata = json.loads((PIPELINE / "desiderata.json").read_text(encoding="utf-8"))
    rng = random.Random(2026)
    mappa, adesso = {}, datetime.now(timezone.utc)
    for k, (soggetto, nome) in enumerate(cs.NOMI.items()):
        sid = cs.nuovo_id()
        base = cs.UPLOADS / sid
        for sotto in ("orig", "view", "thumb"):
            (base / sotto).mkdir(parents=True, exist_ok=True)
        ref_nome = "sprite_partenza.png"
        ref = Image.open(PIPELINE / "Risultati" / soggetto / "16colori" / sprite[soggetto]["file"]).convert("RGB")
        ref.resize((ref.width * 4, ref.height * 4), Image.NEAREST).save(base / "view" / ref_nome)

        domande, corrisp = [], {}
        codici = rng.sample(range(100, 1000), 6)
        voci = [(liv, ver) for liv in ("L1", "L2", "L3") for ver in ("A", "B")]
        for (liv, ver), codice in zip(voci, codici):
            cartella = VIDEO / soggetto / liv
            azione = json.loads((cartella / "01_prompt.json").read_text(encoding="utf-8"))["azione"]
            qid = cs.nuovo_id()
            file = f"animazione_{codice}.webp"
            shutil.copy2(cartella / f"{ver}.webp", base / "view" / f"{qid}_{file}")
            # versione x8 per l'ingrandimento
            anim = Image.open(cartella / f"{ver}.webp")
            fot = []
            for i in range(anim.n_frames):
                anim.seek(i)
                f = anim.convert("RGB")
                fot.append(f.resize((f.width * 2, f.height * 2), Image.NEAREST))
            fot[0].save(base / "orig" / f"{qid}_{file}", save_all=True, append_images=fot[1:],
                        duration=anim.info.get("duration", 62), loop=0, lossless=True, quality=100, method=4)
            url = lambda sotto, n: f"/uploads/{sid}/{sotto}/{quote(n)}"
            domande.append({"id": qid, "label": "Animazione", "file": file,
                            "orig": url("orig", f"{qid}_{file}"), "view": url("view", f"{qid}_{file}"),
                            "thumb": url("view", f"{qid}_{file}"), "width": 1024, "height": 1024,
                            "description": f"Movimento richiesto: {azione}",
                            "reference": url("view", ref_nome), "referenceLabel": "sprite di partenza"})
            corrisp[qid] = {"livello": liv, "versione": ver, "file": file}
        sondaggio = {
            "id": sid, "title": f"Valutazione animazioni: {nome}",
            "description": DESCRIZIONE.format(desiderata=desiderata[soggetto])[:1000],
            "createdAt": adesso.replace(microsecond=0).isoformat().replace("+00:00", f".{k:03d}Z"),
            "scaleMin": 1, "scaleMax": 5, "labelMin": "", "labelMax": "",
            "askName": True, "allowComments": False, "shuffle": True, "closed": False,
            "criteria": CRITERI, "questions": domande,
        }
        db["surveys"].append(sondaggio)
        mappa[soggetto] = {"id": sid, "titolo": sondaggio["title"], "sprite": sprite[soggetto]["file"],
                           "domande": corrisp}
        print(f"Creato: {sondaggio['title']:<45} 6 animazioni (id {sid})")

    tmp = cs.DB.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cs.DB)
    MAPPA_VIDEO.write_text(json.dumps(mappa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Corrispondenza salvata in {MAPPA_VIDEO.name}")


if __name__ == "__main__":
    main()
