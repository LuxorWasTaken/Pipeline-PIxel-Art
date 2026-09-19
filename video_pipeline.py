"""
Estensione temporale della pipeline (Capitolo 5): da uno sprite a tre animazioni.

Per ogni soggetto si usa lo sprite con il voto medio umano piu' alto (a parita',
quello con voti piu' concordi) e si generano tre animazioni, una per livello:
  L1  micro-movimento sul posto (respiro, ondeggiare)
  L2  movimento medio ciclico di tutto il corpo (es. camminata sul posto)
  L3  azione ampia caratteristica del soggetto

Fasi (ciascuna riprendibile):
  prompt  Gemma 4 riceve lo sprite, la desiderata e il livello e restituisce in JSON
          l'azione (in italiano, per i valutatori), il prompt e il prompt negativo
          per il modello video
  video   Wan 2.2 I2V A14B in ComfyUI, 512x512 (sprite ingrandito x4), 81 fotogrammi,
          primo e ultimo fotogramma = sprite (animazione in loop), due modelli
          (rumore alto / basso) con adattatori lightx2v a 4+4 passi, NAG per rendere
          efficace il prompt negativo con CFG = 1
  post    riduzione nearest neighbour a 128x128 e due versioni:
            A  palette indipendente per fotogramma (procedura del Capitolo 3)
            B  palette vincolata: tutti i fotogrammi sulla palette dello sprite
          animazioni WebP senza perdita, ingrandite x4, a 16 fps

USO
  python video_pipeline.py --prova                      Gnomo, livello L1
  python video_pipeline.py                              tutti i soggetti e livelli
  python video_pipeline.py --soggetti Cat --livelli L3
  python video_pipeline.py --fasi post                  rifa' solo la post-elaborazione

Uscita: Risultati_video/<Soggetto>/<Livello>/
"""

import os
import argparse
import base64
import csv
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import pipeline as pl
from riduci_palette import distanze

sys.stdout.reconfigure(encoding="utf-8")
PIPELINE = Path(__file__).resolve().parent
RISULTATI = PIPELINE / "Risultati"
USCITA = PIPELINE / "Risultati_video"
SONDAGGI = Path(os.environ.get("SONDAGGI_TESI", Path(__file__).resolve().parent.parent / "Sondaggi tesi"))
MAPPA = PIPELINE / "sondaggi_v2.json"

# --------------------------------------------------------------- configurazione
MODELLO_ALTO = "Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
MODELLO_BASSO = "Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"
LORA_ALTO = "WAN\\wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
LORA_BASSO = "WAN\\wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
CODIFICATORE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
LATO = 512                 # 128 x 4
INGRANDIMENTO = 4
FOTOGRAMMI = 81
FPS = 16
PASSI, PASSO_CAMBIO = 8, 4
SHIFT = 8.0
NAG = {"nag_scale": 11.0, "nag_alpha": 0.25, "nag_tau": 2.5}

LIVELLI = {
    "L1": "micro-movimento: un movimento minimo e ciclico sul posto, come il respiro, un leggero "
          "ondeggiare del corpo o il battito delle palpebre; il personaggio non cambia posa.",
    "L2": "movimento medio: un ciclo di movimento ben visibile che coinvolge tutto il corpo o gli arti, "
          "eseguito sul posto, come una camminata senza spostarsi nell'inquadratura; per un soggetto "
          "sdraiato o seduto, un cambiamento visibile della posizione del corpo o degli arti. Un semplice "
          "respiro o un leggero dondolio non sono sufficienti.",
    "L3": "azione ampia: un'azione espressiva e caratteristica del soggetto, come un attacco, un "
          "salto o un gesto tipico, che coinvolge tutto il corpo.",
}

META_PROMPT_VIDEO = """Ti viene fornito uno sprite in pixel art generato a partire da questa desiderata:
"{DESIDERATA}"

Lo sprite deve diventare un'animazione in loop generata da un modello image-to-video. Il primo e l'ultimo fotogramma dell'animazione coincidono con lo sprite fornito.
Livello di movimento richiesto: {LIVELLO}

Scegli un'azione adatta al personaggio raffigurato e coerente con il livello richiesto, poi scrivi il prompt per il modello video rispettando questi vincoli:
- la camera è fissa: nessun movimento di camera, zoom o rotazione dell'inquadratura;
- lo sfondo verde a tinta unita resta identico e immobile;
- il personaggio resta al centro dell'inquadratura e non esce dall'immagine;
- il movimento parte dalla posa dello sprite e ritorna alla stessa posa alla fine;
- il personaggio mantiene aspetto, colori e proporzioni dello sprite;
- lo stile resta pixel art bidimensionale.
Il prompt va scritto in inglese, in linguaggio naturale, in una o due frasi descrittive.
Scrivi inoltre un prompt negativo in inglese con le caratteristiche da evitare.
Restituisci la risposta esclusivamente in formato JSON con questa struttura:
{"azione": "<breve descrizione in italiano dell'azione scelta>", "prompt": "...", "prompt_negativo": "..."}"""

NEGATIVO_FISSO = ("camera movement, camera pan, zoom in, zoom out, camera rotation, 3d render, "
                  "realistic, blurry, motion blur, background change, scene change, extra characters, "
                  "text, watermark")


def adesso():
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------- sprite
def scegli_sprite():
    mappa = json.loads(MAPPA.read_text(encoding="utf-8"))
    scelti = {}
    for s, info in mappa.items():
        risp = [json.loads(l) for l in open(SONDAGGI / "data" / "responses" / f"{info['id']}.jsonl",
                                            encoding="utf-8") if l.strip()]
        voti = {f: [r["answers"][q]["rating"] for r in risp if q in r["answers"]]
                for q, f in info["domande"].items()}
        f = sorted(voti, key=lambda x: (-np.mean(voti[x]), np.std(voti[x]), x))[0]
        scelti[s] = {"file": f, "media": round(float(np.mean(voti[f])), 3),
                     "dev_std": round(float(np.std(voti[f])), 3), "voti": voti[f]}
    return scelti


# --------------------------------------------------------------- fase prompt
def fase_prompt(s, liv, sprite, cartella):
    file = cartella / "01_prompt.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))
    desiderata = json.loads((PIPELINE / "desiderata.json").read_text(encoding="utf-8"))[s]
    meta = META_PROMPT_VIDEO.replace("{DESIDERATA}", desiderata).replace("{LIVELLO}", LIVELLI[liv])
    img = base64.b64encode((RISULTATI / s / "ingrandite" / sprite["file"]).read_bytes()).decode()
    for tentativo in range(1, pl.TENTATIVI + 1):
        inizio = time.time()
        try:
            r = requests.post(f"{pl.OLLAMA_URL}/api/generate", timeout=900, json={
                "model": pl.MODELLO_LLM, "prompt": meta, "images": [img], "stream": False, "format": "json"})
            r.raise_for_status()
            grezza = r.json().get("response", "")
            j = pl.estrai_json(grezza)
            if not all(str(j.get(k, "")).strip() for k in ("azione", "prompt", "prompt_negativo")):
                raise ValueError("campi mancanti")
        except Exception as e:
            print(f"      prompt, tentativo {tentativo}: {e}")
            continue
        dati = {"soggetto": s, "livello": liv, "descrizione_livello": LIVELLI[liv], "sprite": sprite,
                "desiderata": desiderata, "meta_prompt": meta, "azione": j["azione"].strip(),
                "prompt": j["prompt"].strip(),
                "prompt_negativo": j["prompt_negativo"].strip().rstrip(",. ") + ", " + NEGATIVO_FISSO,
                "prompt_negativo_modello": j["prompt_negativo"].strip(), "tentativi": tentativo,
                "durata_s": round(time.time() - inizio, 1), "data": adesso(), "risposta_grezza": grezza}
        file.write_text(json.dumps(dati, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"      azione: {dati['azione']}")
        return dati
    sys.exit(f"Prompt non ottenuto per {s} {liv}")


# --------------------------------------------------------------- fase video
def workflow_video(immagine, positivo, negativo, seme, prefisso):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELLO_ALTO, "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELLO_BASSO, "weight_dtype": "default"}},
        "3": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["1", 0], "sage_attention": "auto"}},
        "4": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["2", 0], "sage_attention": "auto"}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": SHIFT}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": SHIFT}},
        "7": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["5", 0], "lora_name": LORA_ALTO, "strength_model": 1.0}},
        "8": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["6", 0], "lora_name": LORA_BASSO, "strength_model": 1.0}},
        "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": CODIFICATORE, "type": "wan"}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": positivo}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": negativo}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "13": {"class_type": "LoadImage", "inputs": {"image": immagine}},
        "14": {"class_type": "WanFirstLastFrameToVideo", "inputs": {
            "positive": ["10", 0], "negative": ["11", 0], "vae": ["12", 0], "width": LATO, "height": LATO,
            "length": FOTOGRAMMI, "batch_size": 1, "start_image": ["13", 0], "end_image": ["13", 0]}},
        "15": {"class_type": "WanVideoNAG", "inputs": {"model": ["7", 0], "conditioning": ["14", 1], **NAG}},
        "16": {"class_type": "WanVideoNAG", "inputs": {"model": ["8", 0], "conditioning": ["14", 1], **NAG}},
        "17": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["15", 0], "add_noise": "enable", "noise_seed": seme, "steps": PASSI, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "beta", "positive": ["14", 0], "negative": ["14", 1],
            "latent_image": ["14", 2], "start_at_step": 0, "end_at_step": PASSO_CAMBIO,
            "return_with_leftover_noise": "enable"}},
        "18": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["16", 0], "add_noise": "disable", "noise_seed": seme, "steps": PASSI, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "beta", "positive": ["14", 0], "negative": ["14", 1],
            "latent_image": ["17", 0], "start_at_step": PASSO_CAMBIO, "end_at_step": 10000,
            "return_with_leftover_noise": "disable"}},
        "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["12", 0]}},
        "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": prefisso}},
    }


def carica_su_comfyui(img, nome):
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = requests.post(f"{pl.COMFYUI_URL}/upload/image", timeout=120,
                      files={"image": (nome, buf.getvalue(), "image/png")},
                      data={"overwrite": "true", "type": "input", "subfolder": "video_tesi"})
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def esegui(workflow, timeout=7200):
    r = requests.post(f"{pl.COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"workflow rifiutato: {r.text[:800]}")
    pid, inizio = r.json()["prompt_id"], time.time()
    while True:
        storia = requests.get(f"{pl.COMFYUI_URL}/history/{pid}", timeout=60).json()
        if pid in storia:
            voce = storia[pid]
            stato = voce.get("status", {})
            if stato.get("status_str") == "error":
                raise RuntimeError(f"esecuzione fallita: {json.dumps(stato.get('messages', []))[:800]}")
            if stato.get("completed", True):
                return voce
        if time.time() - inizio > timeout:
            raise TimeoutError("nessun risultato")
        time.sleep(3)


def fase_video(s, liv, dati, cartella):
    dir_grezzi = cartella / "grezzi_512"
    if dir_grezzi.exists() and len(list(dir_grezzi.glob("*.png"))) == FOTOGRAMMI:
        return
    dir_grezzi.mkdir(exist_ok=True)
    sprite = Image.open(RISULTATI / s / "16colori" / dati["sprite"]["file"]).convert("RGB")
    ingresso = sprite.resize((LATO, LATO), Image.NEAREST)
    nome = carica_su_comfyui(ingresso, f"{s}_{dati['sprite']['file']}")
    seme = random.randint(0, 2**53)
    inizio = time.time()
    voce = esegui(workflow_video(nome, dati["prompt"], dati["prompt_negativo"], seme, f"Video_tesi/{s}/{liv}/f"))
    durata = time.time() - inizio
    immagini = [i for u in voce["outputs"].values() for i in u.get("images", []) if i.get("type") == "output"]
    if len(immagini) != FOTOGRAMMI:
        raise RuntimeError(f"attesi {FOTOGRAMMI} fotogrammi, ricevuti {len(immagini)}")
    for k, info in enumerate(immagini):
        q = {"filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": "output"}
        (dir_grezzi / f"{k:03d}.png").write_bytes(requests.get(f"{pl.COMFYUI_URL}/view", params=q, timeout=120).content)
    (cartella / "02_generazione.json").write_text(json.dumps({
        "seme": seme, "durata_s": round(durata, 1), "data": adesso(), "lato": LATO, "fotogrammi": FOTOGRAMMI,
        "fps": FPS, "passi": PASSI, "passo_cambio": PASSO_CAMBIO, "shift": SHIFT, "nag": NAG,
        "modelli": [MODELLO_ALTO, MODELLO_BASSO], "lora": [LORA_ALTO, LORA_BASSO],
        "codificatore": CODIFICATORE, "vae": VAE}, indent=2), encoding="utf-8")
    print(f"      video generato in {durata / 60:.1f} min (seme {seme})")


# --------------------------------------------------------------- fase post
def palette_vincolata(a, sprite):
    """Versione B: sfondo -> colore riservato dello sprite, resto -> colore piu' vicino
    fra i 15 colori del personaggio dello sprite."""
    es_sprite = pl.valida_sfondo(sprite)
    c_sfondo = es_sprite["c_dom"].astype(np.uint8)
    colori = np.unique(sprite.reshape(-1, 3), axis=0)
    personaggio = np.array([c for c in colori if not np.array_equal(c, c_sfondo)], dtype=np.int64)
    es = pl.valida_sfondo(a)              # sfondo del fotogramma: connesso ai bordi
    pixel = a.reshape(-1, 3).astype(np.int64)
    rgba = np.column_stack([pixel, np.full(len(pixel), 255)])
    pal = np.column_stack([personaggio, np.full(len(personaggio), 255)]).tolist()
    distinti, inversa = np.unique(rgba, axis=0, return_inverse=True)
    vicino = distanze(distinti, pal).argmin(axis=1)
    uscita = personaggio[vicino[inversa.reshape(-1)]].reshape(a.shape).astype(np.uint8)
    uscita[es["maschera"]] = c_sfondo
    return uscita


def salva_animazione(fotogrammi, percorso, scala):
    imgs = [Image.fromarray(f).resize((f.shape[1] * scala, f.shape[0] * scala), Image.NEAREST) for f in fotogrammi]
    imgs[0].save(percorso, save_all=True, append_images=imgs[1:], duration=round(1000 / FPS), loop=0,
                 lossless=True, quality=100, method=4)


def fase_post(s, liv, dati, cartella):
    grezzi = sorted((cartella / "grezzi_512").glob("*.png"))
    if len(grezzi) != FOTOGRAMMI:
        print("      fotogrammi mancanti, post-elaborazione saltata")
        return
    sprite = np.asarray(Image.open(RISULTATI / s / "16colori" / dati["sprite"]["file"]).convert("RGB"))
    # l'ultimo fotogramma coincide con il primo: lo si esclude per un loop senza ripetizioni
    grezzi = grezzi[:-1]
    logici, A, B, righe = [], [], [], []
    for k, f in enumerate(grezzi):
        img = Image.open(f).convert("RGB")
        a = np.asarray(img.resize((img.width // INGRANDIMENTO, img.height // INGRANDIMENTO), Image.NEAREST))
        es = pl.valida_sfondo(a)
        fa = pl.riduci_palette(a, es) if es["ammessa"] else a.copy()
        fb = palette_vincolata(a, sprite)
        logici.append(a); A.append(fa); B.append(fb)
        righe.append({"fotogramma": k, "colori_512": pl.conta_colori(np.asarray(img)), "colori_128": pl.conta_colori(a),
                      "P": round(es["P"], 4), "sfondo_ammesso": int(es["ammessa"]),
                      "colori_A": pl.conta_colori(fa), "colori_B": pl.conta_colori(fb)})
    for nome, seq in (("logici_128", logici), ("A_128", A), ("B_128", B)):
        d = cartella / nome
        d.mkdir(exist_ok=True)
        for k, f in enumerate(seq):
            Image.fromarray(f).save(d / f"{k:03d}.png")
    salva_animazione([np.asarray(Image.open(f).convert("RGB")) for f in grezzi], cartella / "grezzo.webp", 1)
    salva_animazione(A, cartella / "A.webp", INGRANDIMENTO)
    salva_animazione(B, cartella / "B.webp", INGRANDIMENTO)
    with open(cartella / "03_post.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(righe[0]), delimiter=";")
        w.writeheader()
        w.writerows(righe)
    print(f"      post: {len(A)} fotogrammi, colori A max {max(r['colori_A'] for r in righe)}, "
          f"B max {max(r['colori_B'] for r in righe)}, fotogrammi con sfondo non ammesso "
          f"{sum(1 - r['sfondo_ammesso'] for r in righe)}")


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generazione e post-elaborazione delle animazioni (Capitolo 5).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--soggetti")
    ap.add_argument("--livelli", default="L1,L2,L3")
    ap.add_argument("--fasi", default="prompt,video,post")
    ap.add_argument("--prova", action="store_true", help="solo Gnomo, livello L1")
    args = ap.parse_args()

    sprite = scegli_sprite()
    USCITA.mkdir(exist_ok=True)
    (USCITA / "sprite_scelti.json").write_text(json.dumps(sprite, indent=2), encoding="utf-8")
    soggetti = ["Gnomo"] if args.prova else ([s.strip() for s in args.soggetti.split(",")] if args.soggetti else list(sprite))
    livelli = ["L1"] if args.prova else [l.strip() for l in args.livelli.split(",")]
    fasi = args.fasi.split(",")
    if "video" in fasi:
        requests.get(f"{pl.COMFYUI_URL}/system_stats", timeout=5).raise_for_status()
    inizio = time.time()
    for s in soggetti:
        for liv in livelli:
            cartella = USCITA / s / liv
            cartella.mkdir(parents=True, exist_ok=True)
            print(f"=== {s} {liv} (sprite {sprite[s]['file']}, voto {sprite[s]['media']})")
            dati = fase_prompt(s, liv, sprite[s], cartella) if "prompt" in fasi else \
                json.loads((cartella / "01_prompt.json").read_text(encoding="utf-8"))
            if "video" in fasi:
                fase_video(s, liv, dati, cartella)
            if "post" in fasi:
                fase_post(s, liv, dati, cartella)
    print(f"Completato in {(time.time() - inizio) / 60:.1f} minuti. Risultati in {USCITA}")


if __name__ == "__main__":
    main()
