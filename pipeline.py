"""
Pipeline completa: dalla desiderata alle immagini selezionate.

Per ogni soggetto elencato in desiderata.json esegue in sequenza:

  Fase 1  Front-end semantico
          La desiderata viene inserita nel meta-prompt e inviata a Gemma 4
          (Ollama). La risposta JSON contiene prompt positivo, prompt negativo
          e 15 quesiti classificati in fondamentale / importante / utile.

  Fase 2  Generazione (ComfyUI)
          N immagini 1024x1024 con checkpoint, LoRA e campionatore fissati;
          il seme di ogni immagine e' casuale e viene registrato.

  Fase 3  Post-elaborazione deterministica (Python)
          a) discretizzazione: nearest neighbour 1024 -> 128 (fattore 8);
          b) validazione dello sfondo sull'immagine 128x128;
          c) solo per le ammesse, riduzione della palette: il colore di sfondo
             e' una voce riservata e i pixel del personaggio vengono ridotti a
             COLORI-1 colori (median cut + k-means percettivo);
          d) ingrandimento nearest neighbour x8 per la valutazione.

  Fase 4  Selezione (torneo a gironi da 8, oracolo visione-linguaggio)
          Le immagini ammesse, nell'ordine di generazione, sono divise in
          gironi da 8; ogni girone e' un torneo a eliminazione diretta.
          Ogni confronto viene registrato con la motivazione del modello.

L'esecuzione e' riprendibile: rilanciando lo stesso comando le fasi e le
immagini gia' completate vengono saltate.

USO
  python pipeline.py                               tutti i soggetti, tutte le fasi
  python pipeline.py --soggetti Pirata,Cowboy
  python pipeline.py --fasi front-end,generazione  solo alcune fasi
  python pipeline.py --soggetti Gnomo --n 4        prova veloce

STRUTTURA DI USCITA  (cartella Risultati/<Soggetto>/)
  01_front_end.json          desiderata, meta-prompt, prompt, negativo, quesiti
  02_generazione.csv         indice, seme, durata
  1024/NNN.png               immagini generate (con il workflow nei metadati)
  128/NNN.png                immagini discretizzate
  16colori/NNN.png           immagini ammesse con palette ridotta (128x128)
  ingrandite/NNN.png         le stesse ingrandite x8 (1024x1024), viste dal VLM
  03_post_elaborazione.csv   colori per stadio, sfondo (P, esito), colori finali
  04_torneo_log.jsonl        ogni confronto: coppia, scelta, motivazione, durata
  04_torneo.csv              girone e turno raggiunto da ogni immagine
  vincitrici/                l'immagine vincente di ogni girone

Dipendenze: pip install pillow numpy requests
Richiede ComfyUI aperto (porta 8000) e Ollama con gemma4:31b-cloud.
"""

import argparse
import base64
import csv
import json
import random
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from riduci_palette import quantizza

sys.stdout.reconfigure(encoding="utf-8")
CARTELLA = Path(__file__).resolve().parent

# ==========================================================================
#  CONFIGURAZIONE
# ==========================================================================
OLLAMA_URL = "http://localhost:11434"
MODELLO_LLM = "gemma4:31b-cloud"
COMFYUI_URL = "http://127.0.0.1:8000"

CHECKPOINT = "waiIllustriousSDXL_v170.safetensors"
LORA = "Ill\\ElinSpriteNoobLocon_byKonan.safetensors"
FORZA_LORA = 1.0
PASSI = 30
CFG = 3.5
CAMPIONATORE = "dpmpp_2m"
SCHEDULAZIONE = "karras"
LATO = 1024

N_IMMAGINI = 80
FATTORE = 8                # 1024 / 8 = 128
EPS = 10.0                 # tolleranza sul colore di sfondo
SOGLIA = 0.50              # quota minima di pixel di sfondo
CORNICE = 2                # spessore in pixel logici della cornice
COLORI = 16                # colori totali, sfondo compreso
DIM_GIRONE = 8
TENTATIVI = 3

LIVELLI = ("fondamentale", "importante", "utile")

META_PROMPT = """Scrivimi un prompt dettagliato utilizzando i booru tag per la generazione di un'immagine in base a questa desiderata:
"{DESIDERATA}"

Dopo scrivimi 15 domande da imporre ad un VLM che devono avere una risposta si o no che serviranno per far valutare dal VLM se l'immagine generata è coerente alla desiderata. Voglio inoltre che classifichi le domande che generi in base a 3 livelli: fondamentale, importante, utile.

Scrivimi inoltre un prompt negativo con i booru tag delle caratteristiche da evitare.
Restituisci la risposta esclusivamente in formato JSON con questa struttura:
{"prompt": "...", "prompt_negativo": "...", "domande": [{"testo": "...", "livello": "fondamentale|importante|utile"}]}"""

PROMPT_ORACOLO = """Ti vengono fornite due immagini, etichettate A e B, generate a partire dalla stessa richiesta.

RICHIESTA ORIGINALE:
{{"{DESIDERATA}"}}

REQUISITI DA VALUTARE (in ordine di importanza):
{{{REQUISITI}}}

ISTRUZIONI CRITICHE:
1. Confronta l'immagine A e l'immagine B rispetto ai requisiti elencati.
2. Scegli quale delle due soddisfa MEGLIO l'insieme dei requisiti, dando priorità ai requisiti fondamentali.
3. Se sono equivalenti sui requisiti fondamentali, usa la qualità estetica generale come spareggio.
4. NON scrivere testo introduttivo o di cortesia.
5. L'output DEVE iniziare con {{ e finire con }}.

FORMATO OUTPUT:
{{
"vincitore": "A" oppure "B",
"motivazione": "<Riassunto conciso ad alto livello, max 1-2 righe>"
}}"""


def adesso():
    return datetime.now().isoformat(timespec="seconds")


def nome(i):
    return f"{i:03d}.png"


def scrivi_csv(percorso, campi, righe):
    with open(percorso, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campi, delimiter=";")
        w.writeheader()
        w.writerows(righe)


def leggi_csv(percorso):
    if not percorso.exists():
        return []
    with open(percorso, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def estrai_json(testo):
    m = re.search(r"\{.*\}", testo, re.DOTALL)
    return json.loads(m.group(0) if m else testo)


# ==========================================================================
#  FASE 1 - FRONT-END SEMANTICO
# ==========================================================================
def valida_front_end(r):
    if not isinstance(r.get("prompt"), str) or not r["prompt"].strip():
        raise ValueError("prompt mancante")
    if not isinstance(r.get("prompt_negativo"), str) or not r["prompt_negativo"].strip():
        raise ValueError("prompt_negativo mancante")
    domande = r.get("domande")
    if not isinstance(domande, list) or len(domande) != 15:
        raise ValueError(f"attese 15 domande, ricevute {len(domande) if isinstance(domande, list) else 0}")
    for d in domande:
        d["livello"] = str(d.get("livello", "")).strip().lower()
        if d["livello"] not in LIVELLI or not str(d.get("testo", "")).strip():
            raise ValueError(f"domanda non valida: {d}")


def fase_front_end(soggetto, desiderata, cartella):
    file = cartella / "01_front_end.json"
    if file.exists():
        print("  [1] front-end gia' eseguito")
        return json.loads(file.read_text(encoding="utf-8"))

    meta = META_PROMPT.replace("{DESIDERATA}", desiderata)
    for tentativo in range(1, TENTATIVI + 1):
        print(f"  [1] interrogazione di {MODELLO_LLM} (tentativo {tentativo}) ... ", end="", flush=True)
        inizio = time.time()
        try:
            risp = requests.post(f"{OLLAMA_URL}/api/generate", timeout=900, json={
                "model": MODELLO_LLM, "prompt": meta, "stream": False, "format": "json"})
            risp.raise_for_status()
            grezza = risp.json().get("response", "")
            r = estrai_json(grezza)
            valida_front_end(r)
        except Exception as e:
            print(f"errore: {e}")
            continue
        conteggio = {l: sum(d["livello"] == l for d in r["domande"]) for l in LIVELLI}
        dati = {"soggetto": soggetto, "desiderata": desiderata, "modello": MODELLO_LLM,
                "meta_prompt": meta, "prompt": r["prompt"].strip(),
                "prompt_negativo": r["prompt_negativo"].strip(), "domande": r["domande"],
                "ripartizione_livelli": conteggio, "tentativi": tentativo,
                "durata_s": round(time.time() - inizio, 1), "data": adesso(), "risposta_grezza": grezza}
        file.write_text(json.dumps(dati, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ok in {dati['durata_s']:.0f} s, livelli {conteggio}")
        return dati
    sys.exit(f"  Front-end fallito per {soggetto} dopo {TENTATIVI} tentativi.")


# ==========================================================================
#  FASE 2 - GENERAZIONE CON COMFYUI
# ==========================================================================
def workflow_generazione(positivo, negativo, seme, prefisso):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": FORZA_LORA}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positivo}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negativo}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": LATO, "height": LATO, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
            "seed": seme, "steps": PASSI, "cfg": CFG, "sampler_name": CAMPIONATORE,
            "scheduler": SCHEDULAZIONE, "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefisso}},
    }


def esegui_comfyui(workflow, timeout=1800):
    cid = str(uuid.uuid4())
    r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": cid}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI ha rifiutato il workflow: {r.text[:500]}")
    pid = r.json()["prompt_id"]
    inizio = time.time()
    while True:
        storia = requests.get(f"{COMFYUI_URL}/history/{pid}", timeout=60).json()
        if pid in storia:
            voce = storia[pid]
            stato = voce.get("status", {})
            if stato.get("status_str") == "error":
                raise RuntimeError(f"esecuzione fallita: {json.dumps(stato.get('messages', []))[:500]}")
            if stato.get("completed", True):
                break
        if time.time() - inizio > timeout:
            raise TimeoutError("ComfyUI non ha restituito risultati")
        time.sleep(1.0)
    for uscita in voce["outputs"].values():
        for info in uscita.get("images", []):
            if info.get("type") == "output":
                q = {"filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": "output"}
                return requests.get(f"{COMFYUI_URL}/view", params=q, timeout=300).content
    raise RuntimeError("nessuna immagine prodotta")


def fase_generazione(soggetto, fe, cartella, n):
    dir_1024 = cartella / "1024"
    dir_1024.mkdir(exist_ok=True)
    registro = cartella / "02_generazione.csv"
    righe = {int(r["indice"]): r for r in leggi_csv(registro)}
    mancanti = [i for i in range(1, n + 1) if not (dir_1024 / nome(i)).exists()]
    if not mancanti:
        print(f"  [2] generazione gia' completa ({n} immagini)")
        return
    print(f"  [2] generazione di {len(mancanti)} immagini con ComfyUI")
    for k, i in enumerate(mancanti, 1):
        seme = random.randint(0, 2**64 - 1)
        wf = workflow_generazione(fe["prompt"], fe["prompt_negativo"], seme, f"Pipeline/{soggetto}/img")
        inizio = time.time()
        for tentativo in range(1, TENTATIVI + 1):
            try:
                dati = esegui_comfyui(wf)
                break
            except Exception as e:
                print(f"      {nome(i)} tentativo {tentativo} fallito: {e}")
                if tentativo == TENTATIVI:
                    sys.exit("  Generazione interrotta: controlla che ComfyUI sia aperto.")
                time.sleep(5)
        (dir_1024 / nome(i)).write_bytes(dati)
        durata = time.time() - inizio
        righe[i] = {"indice": i, "file": nome(i), "seme": seme, "durata_s": f"{durata:.1f}", "data": adesso()}
        scrivi_csv(registro, ["indice", "file", "seme", "durata_s", "data"],
                   [righe[j] for j in sorted(righe)])
        print(f"      [{k}/{len(mancanti)}] {nome(i)}  seme {seme}  {durata:.1f} s")


# ==========================================================================
#  FASE 3 - POST-ELABORAZIONE DETERMINISTICA
# ==========================================================================
def discretizza(img):
    w, h = img.size
    if w % FATTORE or h % FATTORE:
        raise ValueError(f"dimensioni {w}x{h} non multiple di {FATTORE}")
    return img.resize((w // FATTORE, h // FATTORE), Image.NEAREST)


def pixel_di_cornice(a, t):
    t = max(1, min(t, a.shape[0] // 2, a.shape[1] // 2))
    return np.concatenate([a[:t].reshape(-1, 3), a[-t:].reshape(-1, 3),
                           a[t:-t, :t].reshape(-1, 3), a[t:-t, -t:].reshape(-1, 3)])


def connessi_ai_bordi(maschera):
    raggiunti = np.zeros_like(maschera)
    raggiunti[0, :] = maschera[0, :]
    raggiunti[-1, :] = maschera[-1, :]
    raggiunti[:, 0] |= maschera[:, 0]
    raggiunti[:, -1] |= maschera[:, -1]
    while True:
        nuovo = raggiunti.copy()
        nuovo[1:, :] |= raggiunti[:-1, :]
        nuovo[:-1, :] |= raggiunti[1:, :]
        nuovo[:, 1:] |= raggiunti[:, :-1]
        nuovo[:, :-1] |= raggiunti[:, 1:]
        nuovo &= maschera
        if np.array_equal(nuovo, raggiunti):
            return raggiunti
        raggiunti = nuovo


def valida_sfondo(a):
    """Colore dominante della cornice, quota P dei pixel entro EPS, maschera
    dello sfondo connesso ai bordi."""
    bordo = pixel_di_cornice(a, CORNICE)
    vb, cb = np.unique(bordo, axis=0, return_counts=True)
    c_dom = vb[int(cb.argmax())].astype(np.int64)
    d2 = ((a.astype(np.int64) - c_dom) ** 2).sum(axis=2)
    vicini = d2 <= EPS * EPS
    connessi = connessi_ai_bordi(vicini)
    n = a.shape[0] * a.shape[1]
    P = float(vicini.sum() / n)
    return {"c_dom": c_dom, "P": P, "P_connesso": float(connessi.sum() / n),
            "ammessa": P >= SOGLIA, "maschera": connessi}


def riduci_palette(a, esito):
    """Sfondo come voce riservata, personaggio ridotto a COLORI-1 colori."""
    rgba = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
    rgba[esito["maschera"], 3] = 0                      # lo sfondo non partecipa
    ridotta, _ = quantizza(rgba, COLORI - 1)
    uscita = ridotta[..., :3].copy()
    uscita[esito["maschera"]] = esito["c_dom"].astype(np.uint8)
    return uscita


def conta_colori(a):
    return int(len(np.unique(a.reshape(-1, a.shape[-1]), axis=0)))


def esadecimale(c):
    return "#{:02X}{:02X}{:02X}".format(*[int(x) for x in c])


def fase_post(soggetto, cartella, n):
    dirs = {k: cartella / k for k in ("128", "16colori", "ingrandite")}
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    righe = []
    for i in range(1, n + 1):
        src = cartella / "1024" / nome(i)
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        piccola = discretizza(img)
        piccola.save(dirs["128"] / nome(i))
        a = np.asarray(piccola)
        esito = valida_sfondo(a)
        riga = {"indice": i, "file": nome(i), "colori_1024": conta_colori(np.asarray(img)),
                "colori_128": conta_colori(a), "colore_sfondo": esadecimale(esito["c_dom"]),
                "P": f"{esito['P']:.4f}", "P_connesso": f"{esito['P_connesso']:.4f}",
                "ammessa": int(esito["ammessa"]), "colori_finali": ""}
        if esito["ammessa"]:
            finale = riduci_palette(a, esito)
            Image.fromarray(finale).save(dirs["16colori"] / nome(i))
            Image.fromarray(finale).resize((a.shape[1] * FATTORE, a.shape[0] * FATTORE),
                                           Image.NEAREST).save(dirs["ingrandite"] / nome(i))
            riga["colori_finali"] = conta_colori(finale)
        else:
            for k in ("16colori", "ingrandite"):
                (dirs[k] / nome(i)).unlink(missing_ok=True)
        righe.append(riga)
    scrivi_csv(cartella / "03_post_elaborazione.csv", list(righe[0].keys()) if righe else ["indice"], righe)
    m = sum(r["ammessa"] for r in righe)
    print(f"  [3] post-elaborazione: {len(righe)} immagini, {m} ammesse, {len(righe) - m} scartate")
    return [r["file"] for r in righe if r["ammessa"]]


# ==========================================================================
#  FASE 4 - SELEZIONE A TORNEO
# ==========================================================================
def requisiti_testo(domande):
    ordinate = sorted(domande, key=lambda d: LIVELLI.index(d["livello"]))  # sort stabile
    return ",\n".join(f'"{k}. {d["testo"].strip()} [{d["livello"]}]"' for k, d in enumerate(ordinate, 1))


def b64(percorso):
    return base64.b64encode(percorso.read_bytes()).decode("utf-8")


def confronta(prompt, img_a, img_b):
    """Restituisce (scelta, motivazione, tentativi, risposta grezza); A vince per
    default se il modello non risponde in modo valido."""
    ultima = ""
    for tentativo in range(1, TENTATIVI + 1):
        try:
            risp = requests.post(f"{OLLAMA_URL}/api/generate", timeout=900, json={
                "model": MODELLO_LLM, "prompt": prompt, "images": [b64(img_a), b64(img_b)],
                "stream": False, "format": "json", "options": {"temperature": 0.0, "top_p": 0.1}})
            risp.raise_for_status()
            ultima = risp.json().get("response", "")
            r = estrai_json(ultima)
            scelta = str(r.get("vincitore", "")).strip().upper()
            if scelta in ("A", "B"):
                return scelta, r.get("motivazione", ""), tentativo, ultima
        except (json.JSONDecodeError, ValueError, AttributeError):
            time.sleep(1)
        except Exception as e:
            ultima = f"errore API: {e}"
            break
    return "A", "DEFAULT: nessuna risposta valida, avanza A", TENTATIVI, ultima


def fase_torneo(soggetto, fe, cartella, ammesse):
    if not ammesse:
        print("  [4] nessuna immagine ammessa: occorre rigenerare l'intero insieme di candidati")
        return
    prompt = PROMPT_ORACOLO.replace("{DESIDERATA}", fe["desiderata"]) \
                           .replace("{REQUISITI}", requisiti_testo(fe["domande"])) \
                           .replace("{{", "{").replace("}}", "}")
    log_path = cartella / "04_torneo_log.jsonl"
    cache = {}
    if log_path.exists():
        for riga in log_path.read_text(encoding="utf-8").splitlines():
            if riga.strip():
                v = json.loads(riga)
                cache[(v["girone"], v["A"], v["B"])] = v

    dir_img = cartella / "ingrandite"
    dir_vinc = cartella / "vincitrici"
    dir_vinc.mkdir(exist_ok=True)
    gironi = [ammesse[k:k + DIM_GIRONE] for k in range(0, len(ammesse), DIM_GIRONE)]
    turno_raggiunto = {}
    print(f"  [4] torneo: {len(ammesse)} immagini ammesse in {len(gironi)} gironi")

    for g, girone in enumerate(gironi, 1):
        partecipanti = list(girone)
        turno = 1
        for f in partecipanti:
            turno_raggiunto[f] = (g, 1)
        while len(partecipanti) > 1:
            vincitori = []
            for k in range(0, len(partecipanti), 2):
                if k + 1 >= len(partecipanti):              # numero dispari: passa il turno
                    vincitori.append(partecipanti[k])
                    continue
                a, b = partecipanti[k], partecipanti[k + 1]
                if (g, a, b) in cache:
                    v = cache[(g, a, b)]
                else:
                    inizio = time.time()
                    scelta, motivazione, tent, grezza = confronta(prompt, dir_img / a, dir_img / b)
                    v = {"soggetto": soggetto, "girone": g, "turno": turno, "A": a, "B": b,
                         "scelta": scelta, "vincitore": a if scelta == "A" else b,
                         "motivazione": motivazione, "tentativi": tent,
                         "durata_s": round(time.time() - inizio, 1), "data": adesso(),
                         "risposta_grezza": grezza}
                    with open(log_path, "a", encoding="utf-8") as fl:
                        fl.write(json.dumps(v, ensure_ascii=False) + "\n")
                    print(f"      girone {g} turno {turno}: {a} vs {b} -> {v['vincitore']} "
                          f"({v['durata_s']:.0f} s)")
                vincitori.append(v["vincitore"])
            turno += 1
            partecipanti = vincitori
            for f in partecipanti:
                turno_raggiunto[f] = (g, turno)
        vincitrice = partecipanti[0]
        shutil.copy(dir_img / vincitrice, dir_vinc / f"girone_{g:02d}_{vincitrice}")
        print(f"      girone {g}: vince {vincitrice}")

    scrivi_csv(cartella / "04_torneo.csv", ["file", "girone", "turno_raggiunto"],
               [{"file": f, "girone": gt[0], "turno_raggiunto": gt[1]}
                for f, gt in sorted(turno_raggiunto.items())])


# ==========================================================================
def main():
    global COMFYUI_URL, OLLAMA_URL
    fasi_tutte = ["front-end", "generazione", "post", "torneo"]
    ap = argparse.ArgumentParser(description="Pipeline completa dalla desiderata alle immagini selezionate.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--soggetti", help="soggetti separati da virgola (predefinito: tutti)")
    ap.add_argument("--fasi", default=",".join(fasi_tutte), help="fasi da eseguire, separate da virgola")
    ap.add_argument("--n", type=int, default=N_IMMAGINI, help="immagini per soggetto (predefinito 80)")
    ap.add_argument("--desiderata", default=str(CARTELLA / "desiderata.json"))
    ap.add_argument("--uscita", default=str(CARTELLA / "Risultati"))
    ap.add_argument("--comfyui", default=COMFYUI_URL)
    ap.add_argument("--ollama", default=OLLAMA_URL)
    args = ap.parse_args()
    COMFYUI_URL, OLLAMA_URL = args.comfyui.rstrip("/"), args.ollama.rstrip("/")

    fasi = [f.strip() for f in args.fasi.split(",")]
    for f in fasi:
        if f not in fasi_tutte:
            sys.exit(f"Fase sconosciuta: {f} (valide: {', '.join(fasi_tutte)})")
    desiderata = json.loads(Path(args.desiderata).read_text(encoding="utf-8"))
    soggetti = [s.strip() for s in args.soggetti.split(",")] if args.soggetti else list(desiderata)
    for s in soggetti:
        if s not in desiderata:
            sys.exit(f"Soggetto {s} non presente in {args.desiderata}")

    if "generazione" in fasi:
        try:
            requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).raise_for_status()
        except Exception:
            sys.exit(f"ComfyUI non raggiungibile su {COMFYUI_URL}: aprilo prima di lanciare la pipeline.")
    if "front-end" in fasi or "torneo" in fasi:
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
        except Exception:
            sys.exit(f"Ollama non raggiungibile su {OLLAMA_URL}.")

    inizio_tot = time.time()
    for s in soggetti:
        print(f"\n=== {s} ===")
        cartella = Path(args.uscita) / s
        cartella.mkdir(parents=True, exist_ok=True)
        fe = fase_front_end(s, desiderata[s], cartella) if "front-end" in fasi else \
            json.loads((cartella / "01_front_end.json").read_text(encoding="utf-8"))
        if "generazione" in fasi:
            fase_generazione(s, fe, cartella, args.n)
        ammesse = fase_post(s, cartella, args.n) if "post" in fasi else \
            [r["file"] for r in leggi_csv(cartella / "03_post_elaborazione.csv") if r["ammessa"] == "1"]
        if "torneo" in fasi:
            fase_torneo(s, fe, cartella, ammesse)
    print(f"\nPipeline completata in {(time.time() - inizio_tot) / 60:.1f} minuti. Risultati in {args.uscita}")


if __name__ == "__main__":
    main()
