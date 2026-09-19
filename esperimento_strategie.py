"""
Confronto sperimentale delle quattro strategie di selezione (Capitolo 4).

DISEGNO
  Le immagini ammesse di ciascun soggetto vengono divise, nell'ordine di
  generazione, in gruppi da 20 (001-020, 021-040, 041-060, 061-080): con 10
  soggetti si ottengono 40 gruppi, ciascuno dei quali e' un problema di
  ordinamento indipendente. Su ogni gruppo:
    - le quattro strategie producono un ordinamento dei candidati
        confronto esaustivo  (190 confronti)
        ricerca lineare      (19 confronti, sottoinsieme dei precedenti)
        torneo               (19 confronti, sottoinsieme dei precedenti)
        valutazione globale  (1 interrogazione con le 20 immagini)
    - il panel umano fornisce la rilevanza di ogni immagine: voto medio - 1
  e si calcolano NDCG@k, tau-b di Kendall e la presenza del vincitore fra i
  primi tre umani. Le 40 misure per strategia danno media, intervallo di
  confidenza bootstrap e test di Wilcoxon appaiati fra strategie.

RIUSO DELLE INTERROGAZIONI
  Ogni confronto e' posto con in posizione A il candidato generato prima.
  Con questa convenzione i confronti della ricerca lineare e del torneo sono
  anche confronti del confronto esaustivo: 190 interrogazioni per gruppo
  bastano alle tre strategie a coppie. L'oracolo e' quello di pipeline.py
  (stessa direttiva, stesse immagini ingrandite, stessi parametri), e i
  confronti gia' eseguiti dal torneo della pipeline vengono riutilizzati.
  Tutte le risposte sono salvate: il programma si puo' interrompere e
  riprendere senza ripetere chiamate.

USO
  python esperimento_strategie.py --piano              gruppi, costi, stato del panel
  python esperimento_strategie.py --simulato           prova completa senza modello ne' voti reali
  python esperimento_strategie.py                      esegue le interrogazioni mancanti,
                                                       poi l'analisi se i voti sono disponibili
  python esperimento_strategie.py --solo-analisi       solo analisi dai dati salvati
  opzioni: --paralleli 2   --simmetrico   --soggetti Pirata,Cane   --limite 10

Dipendenze: pip install numpy scipy requests pillow
"""

import os
import argparse
import csv
import json
import math
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import pipeline as pl

sys.stdout.reconfigure(encoding="utf-8")
PIPELINE = Path(__file__).resolve().parent
RISULTATI = PIPELINE / "Risultati"
SONDAGGI = Path(os.environ.get("SONDAGGI_TESI", Path(__file__).resolve().parent.parent / "Sondaggi tesi"))
MAPPA = PIPELINE / "sondaggi_v2.json"

DIM_GRUPPO = 20
KS = (1, 3, 5, 10)
SEME = 2026
MIN_SECONDI_PER_IMMAGINE = 1.5     # compilazioni piu' rapide vengono escluse
MIN_COMPLETEZZA = 0.9              # quota minima di immagini votate

NOMI_STRATEGIE = ["confronto esaustivo", "ricerca lineare", "torneo", "valutazione globale"]

PROMPT_GLOBALE = """Ti vengono fornite {n} immagini, numerate da 1 a {n} nell'ordine in cui sono allegate, generate a partire dalla stessa richiesta.

RICHIESTA ORIGINALE:
{{"{desiderata}"}}

REQUISITI DA VALUTARE (in ordine di importanza):
{{{requisiti}}}

ISTRUZIONI CRITICHE:
1. Confronta tutte le immagini rispetto ai requisiti elencati.
2. Ordinale dalla migliore alla peggiore rispetto all'insieme dei requisiti, dando priorità ai requisiti fondamentali.
3. A parità di requisiti fondamentali, usa la qualità estetica generale come spareggio.
4. L'ordinamento deve contenere ciascun numero da 1 a {n} esattamente una volta.
5. NON scrivere testo introduttivo o di cortesia.
6. L'output DEVE iniziare con {{ e finire con }}.

FORMATO OUTPUT:
{{
"classifica": [numeri dal migliore al peggiore],
"motivazione": "<Riassunto conciso ad alto livello, max 1-2 righe>"
}}"""


# ==========================================================================
#  Dati
# ==========================================================================
def indice(f):
    return int(Path(f).stem)


def carica_soggetto(s):
    cartella = RISULTATI / s
    fe = json.loads((cartella / "01_front_end.json").read_text(encoding="utf-8"))
    post = list(csv.DictReader(open(cartella / "03_post_elaborazione.csv", encoding="utf-8"), delimiter=";"))
    ammesse = sorted((r["file"] for r in post if r["ammessa"] == "1"), key=indice)
    gruppi = {}
    for f in ammesse:
        g = (indice(f) - 1) // DIM_GRUPPO + 1
        gruppi.setdefault(f"{s}-{g}", []).append(f)
    prompt = pl.PROMPT_ORACOLO.replace("{DESIDERATA}", fe["desiderata"]) \
                              .replace("{REQUISITI}", pl.requisiti_testo(fe["domande"])) \
                              .replace("{{", "{").replace("}}", "}")
    prompt_glob = PROMPT_GLOBALE.format(n="{n}", desiderata=fe["desiderata"],
                                        requisiti=pl.requisiti_testo(fe["domande"]))
    return {"cartella": cartella, "desiderata": fe["desiderata"], "gruppi": gruppi,
            "prompt": prompt, "prompt_globale": prompt_glob}


def carica_panel(soggetti, simulato=False):
    """Restituisce {soggetto: {"voti": {file: [voto per valutatore]}, "valutatori": [...],
    "esclusi": [...]}} oppure None per i soggetti senza voti."""
    if simulato:
        return None
    if not MAPPA.exists():
        return {s: None for s in soggetti}
    mappa = json.loads(MAPPA.read_text(encoding="utf-8"))
    panel = {}
    for s in soggetti:
        info = mappa.get(s)
        f_risp = SONDAGGI / "data" / "responses" / f"{info['id']}.jsonl" if info else None
        if not info or not f_risp.exists():
            panel[s] = None
            continue
        risposte = [json.loads(l) for l in open(f_risp, encoding="utf-8") if l.strip()]
        n_imm = len(info["domande"])
        valide, esclusi = [], []
        for r in risposte:
            n = len(r.get("answers", {}))
            sec = (r.get("durationMs") or 0) / 1000 / max(n, 1)
            if n < MIN_COMPLETEZZA * n_imm:
                esclusi.append(f"{r['participant']} (incompleto: {n}/{n_imm})")
            elif sec < MIN_SECONDI_PER_IMMAGINE:
                esclusi.append(f"{r['participant']} (troppo rapido: {sec:.1f} s/immagine)")
            else:
                valide.append(r)
        voti = {f: [r["answers"].get(qid, {}).get("rating") for r in valide]
                for qid, f in info["domande"].items()}
        panel[s] = {"voti": voti, "valutatori": [r["participant"] for r in valide],
                    "esclusi": esclusi,
                    "durate_min": [round((r.get("durationMs") or 0) / 60000, 1) for r in valide]}
        if not valide:
            panel[s] = None
    return panel


def panel_simulato(dati, rng):
    """Voti fittizi per provare il programma: qualita' latente + rumore per valutatore."""
    panel = {}
    for s, d in dati.items():
        files = [f for g in d["gruppi"].values() for f in g]
        qualita = {f: rng.normal(3.0, 0.9) for f in files}
        voti = {f: [int(np.clip(round(qualita[f] + rng.normal(0, 0.8)), 1, 5)) for _ in range(6)]
                for f in files}
        panel[s] = {"voti": voti, "valutatori": [f"V{i}" for i in range(1, 7)], "esclusi": [],
                    "durate_min": [], "qualita": qualita}
    return panel


# ==========================================================================
#  Oracolo con memoria delle risposte
# ==========================================================================
class Archivio:
    def __init__(self, percorso):
        self.percorso = percorso
        self.dati = {}
        self.lock = threading.Lock()
        if percorso.exists():
            for l in open(percorso, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    self.dati[r["chiave"]] = r

    def get(self, chiave):
        return self.dati.get(chiave)

    def put(self, r):
        with self.lock:
            self.dati[r["chiave"]] = r
            with open(self.percorso, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chiave_coppia(s, a, b):
    return f"{s}|coppia|{a}|{b}"


def importa_torneo_pipeline(archivio, soggetti):
    nuovi = 0
    for s in soggetti:
        log = RISULTATI / s / "04_torneo_log.jsonl"
        if not log.exists():
            continue
        for l in open(log, encoding="utf-8"):
            if not l.strip():
                continue
            v = json.loads(l)
            k = chiave_coppia(s, v["A"], v["B"])
            if archivio.get(k) is None and not v["motivazione"].startswith("DEFAULT"):
                archivio.put({"chiave": k, "soggetto": s, "tipo": "coppia", "A": v["A"], "B": v["B"],
                              "scelta": v["scelta"], "motivazione": v["motivazione"], "fallito": False,
                              "secondi": v["durata_s"], "origine": "torneo della pipeline"})
                nuovi += 1
    return nuovi


class ErroreServizio(RuntimeError):
    """Ollama o il servizio remoto non hanno risposto."""


class Oracolo:
    def __init__(self, s, dati, archivio, simulato=None):
        self.s, self.d, self.archivio, self.simulato = s, dati, archivio, simulato
        self.rng = random.Random(f"{SEME}-{s}")

    def confronto(self, a, b, esegui=True):
        k = chiave_coppia(self.s, a, b)
        r = self.archivio.get(k)
        if r is not None or not esegui:
            return r
        inizio = time.time()
        if self.simulato is not None:
            q = self.simulato
            p_a = 1 / (1 + math.exp(-2.0 * (q[a] - q[b] + 0.2)))
            scelta, motiv, fallito = ("A" if self.rng.random() < p_a else "B"), "simulato", False
        else:
            img = self.d["cartella"] / "ingrandite"
            scelta, motiv, _, grezza = pl.confronta(self.d["prompt"], img / a, img / b)
            if str(grezza).startswith("errore API"):
                # errore di rete o del servizio: non e' una risposta del modello e non
                # va archiviata, altrimenti al rilancio il confronto verrebbe saltato
                raise ErroreServizio(grezza)
            fallito = motiv.startswith("DEFAULT")
        r = {"chiave": k, "soggetto": self.s, "tipo": "coppia", "A": a, "B": b, "scelta": scelta,
             "motivazione": motiv, "fallito": fallito, "secondi": round(time.time() - inizio, 1),
             "origine": "esperimento"}
        self.archivio.put(r)
        return r

    def vince(self, a, b):
        r = self.confronto(a, b, esegui=False)
        if r is None:
            raise KeyError((a, b))
        return a if r["scelta"] == "A" else b

    def globale(self, gruppo, esegui=True):
        k = f"{self.s}|globale|{'|'.join(gruppo)}"
        r = self.archivio.get(k)
        if r is not None or not esegui:
            return r
        n, inizio = len(gruppo), time.time()
        classifica, motiv = None, ""
        if self.simulato is not None:
            punt = {f: self.simulato[f] + self.rng.gauss(0, 1.0) for f in gruppo}
            classifica = [gruppo.index(f) + 1 for f in sorted(gruppo, key=lambda f: -punt[f])]
            motiv = "simulato"
        else:
            import requests
            import base64
            prompt = self.d["prompt_globale"].replace("{n}", str(n))
            imgs = [base64.b64encode((self.d["cartella"] / "ingrandite" / f).read_bytes()).decode()
                    for f in gruppo]
            errore_servizio = False
            for _ in range(pl.TENTATIVI):
                try:
                    risp = requests.post(f"{pl.OLLAMA_URL}/api/generate", timeout=1800, json={
                        "model": pl.MODELLO_LLM, "prompt": prompt, "images": imgs, "stream": False,
                        "format": "json", "options": {"temperature": 0.0, "top_p": 0.1}})
                    risp.raise_for_status()
                    errore_servizio = False
                    j = pl.estrai_json(risp.json().get("response", ""))
                    c = [int(x) for x in j.get("classifica", [])]
                    if sorted(c) == list(range(1, n + 1)):
                        classifica, motiv = c, j.get("motivazione", "")
                        break
                    motiv = f"classifica non valida: {c}"
                except requests.exceptions.RequestException as e:
                    errore_servizio, motiv = True, f"errore del servizio: {e}"
                    time.sleep(5)
                except Exception as e:
                    motiv = f"risposta non valida: {e}"
                    time.sleep(2)
            if classifica is None and errore_servizio:
                raise ErroreServizio(motiv)
        r = {"chiave": k, "soggetto": self.s, "tipo": "globale", "gruppo": gruppo,
             "classifica": classifica, "motivazione": motiv, "fallito": classifica is None,
             "secondi": round(time.time() - inizio, 1), "origine": "esperimento"}
        self.archivio.put(r)
        return r


# ==========================================================================
#  Strategie: punteggio per candidato (piu' alto = migliore, uguali = pari merito)
# ==========================================================================
def coppie_esaustivo(g):
    return [(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))]


def coppie_lineare_torneo(orc, g):
    """Confronti richiesti da ricerca lineare e torneo, calcolabili solo man mano
    che gli esiti sono noti: restituisce i confronti mancanti del passo corrente."""
    mancanti = []
    campione = g[0]
    for f in g[1:]:
        r = orc.confronto(campione, f, esegui=False)
        if r is None:
            mancanti.append((campione, f))
            break
        campione = campione if r["scelta"] == "A" else f
    corrente = list(g)
    while len(corrente) > 1:
        prossimo, bloccato = [], False
        for i in range(0, len(corrente), 2):
            if i + 1 >= len(corrente):
                prossimo.append(corrente[i])
                continue
            r = orc.confronto(corrente[i], corrente[i + 1], esegui=False)
            if r is None:
                mancanti.append((corrente[i], corrente[i + 1]))
                bloccato = True
            else:
                prossimo.append(corrente[i] if r["scelta"] == "A" else corrente[i + 1])
        if bloccato:
            break
        corrente = prossimo
    return list(dict.fromkeys(mancanti))


def esaustivo(orc, g):
    v = {f: 0 for f in g}
    for a, b in coppie_esaustivo(g):
        v[orc.vince(a, b)] += 1
    return v


def lineare(orc, g):
    campione = g[0]
    for f in g[1:]:
        campione = orc.vince(campione, f)
    return {f: int(f == campione) for f in g}


def torneo(orc, g):
    turno = {f: 1 for f in g}
    corrente, k = list(g), 1
    while len(corrente) > 1:
        k += 1
        prossimo = []
        for i in range(0, len(corrente), 2):
            prossimo.append(corrente[i] if i + 1 >= len(corrente) else orc.vince(corrente[i], corrente[i + 1]))
        for f in prossimo:
            turno[f] = k
        corrente = prossimo
    return turno


def globale(orc, g):
    r = orc.globale(g, esegui=False)
    if r is None or r["fallito"]:
        return None
    return {g[idx - 1]: len(g) - pos for pos, idx in enumerate(r["classifica"])}


FUNZIONI = dict(zip(NOMI_STRATEGIE, [esaustivo, lineare, torneo, globale]))


# ==========================================================================
#  Misure
# ==========================================================================
def dcg(punteggi, guadagno, k):
    """DCG@k atteso rispetto a un ordinamento casuale dei pari merito."""
    gruppi = {}
    for f, s in punteggi.items():
        gruppi.setdefault(s, []).append(f)
    pos, tot = 0, 0.0
    for s in sorted(gruppi, reverse=True):
        membri = gruppi[s]
        sconti = [1 / math.log2(p + 2) if p < k else 0.0 for p in range(pos, pos + len(membri))]
        tot += sum(guadagno[f] for f in membri) * sum(sconti) / len(sconti)
        pos += len(membri)
    return tot


def ndcg(punteggi, guadagno, k):
    ideale = dcg(dict(guadagno), guadagno, k)
    return dcg(punteggi, guadagno, k) / ideale if ideale > 0 else float("nan")


def tau_b(x, y):
    from scipy.stats import kendalltau
    t = kendalltau(x, y, variant="b").statistic
    return float(t) if t == t else float("nan")


def top1_in_top3(punteggi, medie):
    """Probabilita' che il primo scelto dalla strategia (sorteggio fra i pari
    merito) sia fra i primi tre umani (pari merito compresi)."""
    soglia = sorted(medie.values(), reverse=True)[min(2, len(medie) - 1)]
    migliori = [f for f, s in punteggi.items() if s == max(punteggi.values())]
    return sum(medie[f] >= soglia for f in migliori) / len(migliori)


def misure(punteggi, medie):
    guadagno = {f: medie[f] - 1.0 for f in medie}
    m = {f"ndcg@{k}": ndcg(punteggi, guadagno, k) for k in KS}
    m["ndcg"] = ndcg(punteggi, guadagno, len(medie))
    files = list(medie)
    m["tau_b"] = tau_b([punteggi[f] for f in files], [medie[f] for f in files])
    m["top1_in_top3"] = top1_in_top3(punteggi, medie)
    return m


def riferimento_casuale(medie, prove, rng):
    files, acc = list(medie), {}
    for _ in range(prove):
        rng.shuffle(files)
        p = {f: len(files) - i for i, f in enumerate(files)}
        for k, v in misure(p, medie).items():
            acc[k] = acc.get(k, 0.0) + (0.0 if k == "tau_b" else v)
    return {k: v / prove for k, v in acc.items()}


def riferimento_umano(voti_gruppo):
    """Accordo fra valutatori: per ciascun valutatore, l'ordinamento dato dalla
    media degli altri e' valutato rispetto ai suoi voti; media sui valutatori."""
    n_val = len(next(iter(voti_gruppo.values())))
    risultati = []
    for r in range(n_val):
        propri = {f: v[r] for f, v in voti_gruppo.items() if v[r] is not None}
        if len(propri) < len(voti_gruppo):
            continue
        altri = {f: np.mean([x for i, x in enumerate(v) if i != r and x is not None]) for f, v in voti_gruppo.items()}
        risultati.append(misure(altri, {f: float(x) for f, x in propri.items()}))
    if not risultati:
        return None
    return {k: float(np.nanmean([x[k] for x in risultati])) for k in risultati[0]}


def contro_singoli(punteggi, voti_gruppo):
    """Stesse misure rispetto a ciascun valutatore, mediate: confrontabili con il
    riferimento umano."""
    n_val = len(next(iter(voti_gruppo.values())))
    ris = []
    for r in range(n_val):
        propri = {f: float(v[r]) for f, v in voti_gruppo.items() if v[r] is not None}
        if len(propri) == len(voti_gruppo):
            ris.append(misure(punteggi, propri))
    return {k: float(np.nanmean([x[k] for x in ris])) for k in ris[0]} if ris else None


def alfa_krippendorff(matrice):
    """Alfa intervallare; matrice elementi x valutatori con None per i mancanti."""
    gruppi, valori = [], []
    for riga in matrice:
        v = [x for x in riga if x is not None]
        if len(v) >= 2:
            gruppi.append(v)
            valori.extend(v)
    n = len(valori)
    if n < 2:
        return float("nan")
    d_oss = sum(sum((a - b) ** 2 for i, a in enumerate(g) for j, b in enumerate(g) if i != j) / (len(g) - 1)
                for g in gruppi) / n
    t = np.array(valori, dtype=float)
    d_att = ((t[:, None] - t[None, :]) ** 2).sum() / (n * (n - 1))
    return 1 - d_oss / d_att if d_att > 0 else float("nan")


def bootstrap_ic(valori, rng, repliche=10000):
    v = np.array([x for x in valori if x == x], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    medie = rng.choice(v, size=(repliche, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(medie, 2.5)), float(np.percentile(medie, 97.5))


def holm(pvalori):
    ordine = sorted(range(len(pvalori)), key=lambda i: pvalori[i])
    corretti, massimo = [0.0] * len(pvalori), 0.0
    for rango, i in enumerate(ordine):
        massimo = max(massimo, min(1.0, (len(pvalori) - rango) * pvalori[i]))
        corretti[i] = massimo
    return corretti


# ==========================================================================
#  Esecuzione delle interrogazioni
# ==========================================================================
def esegui_interrogazioni(dati, oracoli, args):
    compiti_prioritari, compiti_esaustivo, compiti_globali = [], [], []
    for s, d in dati.items():
        for gid, g in d["gruppi"].items():
            if oracoli[s].globale(g, esegui=False) is None:
                compiti_globali.append((s, g))
            for a, b in coppie_esaustivo(g):
                if oracoli[s].confronto(a, b, esegui=False) is None:
                    compiti_esaustivo.append((s, a, b))
                if args.simmetrico and oracoli[s].confronto(b, a, esegui=False) is None:
                    compiti_esaustivo.append((s, b, a))
    totale = len(compiti_globali) + len(compiti_esaustivo)
    if totale == 0:
        print("Tutte le interrogazioni sono gia' state eseguite.")
        return
    tempi = [r["secondi"] for r in next(iter(oracoli.values())).archivio.dati.values() if r["tipo"] == "coppia"]
    unit = float(np.median(tempi)) if tempi else 3.0
    print(f"Interrogazioni da eseguire: {totale} (stima {totale * unit / 3600 / max(args.paralleli, 1):.1f} ore "
          f"a {unit:.1f} s l'una con {args.paralleli} in parallelo)")
    limite = args.limite if args.limite else totale
    fatte = [0]
    t0 = time.time()

    def avanza(msg):
        fatte[0] += 1
        trascorso = time.time() - t0
        resto = trascorso / fatte[0] * (min(limite, totale) - fatte[0])
        print(f"  [{fatte[0]}/{min(limite, totale)}] {msg}   restano circa {resto / 60:.0f} min", flush=True)

    # 1) valutazioni globali (poche)
    for s, g in compiti_globali:
        if fatte[0] >= limite:
            return
        r = oracoli[s].globale(g)
        avanza(f"globale {s} {g[0]}-{g[-1]}: {'ok' if not r['fallito'] else 'FALLITA'} ({r['secondi']:.0f} s)")

    # 2) confronti di ricerca lineare e torneo, in ordine, gruppo per gruppo
    for s, d in dati.items():
        for gid, g in d["gruppi"].items():
            while fatte[0] < limite:
                mancanti = coppie_lineare_torneo(oracoli[s], g)
                if not mancanti:
                    break
                for a, b in mancanti:
                    if fatte[0] >= limite:
                        return
                    r = oracoli[s].confronto(a, b)
                    avanza(f"{gid} {a} vs {b} -> {r['scelta']} ({r['secondi']:.0f} s)")

    # 3) restanti confronti del confronto esaustivo, in parallelo
    da_fare = [(s, a, b) for s, a, b in compiti_esaustivo if oracoli[s].confronto(a, b, esegui=False) is None]
    da_fare = da_fare[:max(0, limite - fatte[0])]

    def lavoro(c):
        s, a, b = c
        r = oracoli[s].confronto(a, b)
        avanza(f"{s} {a} vs {b} -> {r['scelta']} ({r['secondi']:.0f} s)")

    with ThreadPoolExecutor(max_workers=max(1, args.paralleli)) as ex:
        list(ex.map(lavoro, da_fare))


# ==========================================================================
#  Analisi
# ==========================================================================
def analizza(dati, oracoli, panel, cartella, args):
    rng = np.random.default_rng(SEME)
    righe, righe_ordini = [], []
    print("\n=== Panel umano ===")
    accordo = []
    for s in dati:
        p = panel.get(s)
        if p is None:
            print(f"  {s:<12} nessun voto disponibile")
            continue
        alfa = alfa_krippendorff(list(p["voti"].values()))
        accordo.append({"soggetto": s, "valutatori": len(p["valutatori"]), "alfa": alfa,
                        "esclusi": "; ".join(p["esclusi"])})
        print(f"  {s:<12} valutatori {len(p['valutatori'])}  alfa di Krippendorff {alfa:.3f}"
              + (f"  esclusi: {'; '.join(p['esclusi'])}" if p["esclusi"] else ""))

    for s, d in dati.items():
        p = panel.get(s)
        if p is None:
            continue
        for gid, g in d["gruppi"].items():
            voti_g = {f: p["voti"][f] for f in g}
            medie = {f: float(np.mean([x for x in v if x is not None])) for f, v in voti_g.items()}
            base = {"gruppo": gid, "soggetto": s, "n": len(g)}
            for nome in NOMI_STRATEGIE:
                try:
                    punt = FUNZIONI[nome](oracoli[s], g)
                except KeyError:
                    punt = None
                if punt is None:
                    righe.append({**base, "strategia": nome, "stato": "incompleto"})
                    continue
                m = misure(punt, medie)
                singoli = contro_singoli(punt, voti_g) or {}
                righe.append({**base, "strategia": nome, "stato": "ok", **m,
                              **{f"singolo_{k}": v for k, v in singoli.items()}})
                for f in g:
                    righe_ordini.append({"gruppo": gid, "strategia": nome, "file": f,
                                         "punteggio": punt[f], "media_umana": round(medie[f], 3)})
            righe.append({**base, "strategia": "casuale", "stato": "ok",
                          **riferimento_casuale(medie, args.prove_casuali, random.Random(f"{SEME}-{gid}"))})
            ru = riferimento_umano(voti_g)
            if ru:
                righe.append({**base, "strategia": "accordo fra valutatori", "stato": "ok",
                              **{f"singolo_{k}": v for k, v in ru.items()}})

    if not righe:
        print("\nNessun gruppo analizzabile: servono i voti del panel.")
        return

    metriche = [f"ndcg@{k}" for k in KS] + ["ndcg", "tau_b", "top1_in_top3"]
    campi = ["gruppo", "soggetto", "n", "strategia", "stato"] + metriche + [f"singolo_{m}" for m in metriche]
    with open(cartella / "misure_per_gruppo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campi, delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(righe)
    with open(cartella / "ordinamenti.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gruppo", "strategia", "file", "punteggio", "media_umana"], delimiter=";")
        w.writeheader()
        w.writerows(righe_ordini)

    # --- riepilogo con intervalli di confidenza
    nomi = NOMI_STRATEGIE + ["casuale"]
    riepilogo = []
    print("\n=== Accordo con il panel (media sui gruppi, IC 95% bootstrap) ===")
    print(f"  {'strategia':<22}" + "".join(f"{m:>22}" for m in metriche))
    for nome in nomi:
        rr = [r for r in righe if r["strategia"] == nome and r["stato"] == "ok"]
        riga = {"strategia": nome, "gruppi": len(rr)}
        testo = f"  {nome:<22}"
        for m in metriche:
            v = [r[m] for r in rr if m in r and r[m] == r[m]]
            media = float(np.mean(v)) if v else float("nan")
            lo, hi = bootstrap_ic(v, rng) if nome != "casuale" else (float("nan"), float("nan"))
            riga.update({m: media, f"{m}_ic_basso": lo, f"{m}_ic_alto": hi})
            testo += f"{media:>8.3f} [{lo:.2f},{hi:.2f}]" if lo == lo else f"{media:>22.3f}"
        riepilogo.append(riga)
        print(testo)

    print("\n=== Rispetto ai singoli valutatori (confrontabile con l'accordo fra valutatori) ===")
    for nome in NOMI_STRATEGIE + ["accordo fra valutatori"]:
        rr = [r for r in righe if r["strategia"] == nome and f"singolo_ndcg@3" in r]
        if not rr:
            continue
        valori = {m: float(np.nanmean([r[f"singolo_{m}"] for r in rr])) for m in metriche}
        print(f"  {nome:<24}" + "  ".join(f"{m} {v:.3f}" for m, v in valori.items()))
        for rg in riepilogo:
            if rg["strategia"] == nome:
                rg.update({f"singolo_{m}": v for m, v in valori.items()})
        if nome == "accordo fra valutatori":
            riepilogo.append({"strategia": nome, "gruppi": len(rr), **{f"singolo_{m}": v for m, v in valori.items()}})

    # --- test di Wilcoxon appaiati
    from scipy.stats import wilcoxon
    test = []
    coppie = [(a, b) for i, a in enumerate(nomi) for b in nomi[i + 1:]]
    for m in metriche:
        pv, blocco = [], []
        for a, b in coppie:
            va = {r["gruppo"]: r[m] for r in righe if r["strategia"] == a and r["stato"] == "ok"}
            vb = {r["gruppo"]: r[m] for r in righe if r["strategia"] == b and r["stato"] == "ok"}
            comuni = [gid for gid in va if gid in vb and va[gid] == va[gid] and vb[gid] == vb[gid]]
            diff = [va[gid] - vb[gid] for gid in comuni]
            if len(comuni) < 5 or all(abs(x) < 1e-12 for x in diff):
                p = float("nan")
            else:
                p = float(wilcoxon([va[g] for g in comuni], [vb[g] for g in comuni]).pvalue)
            blocco.append({"metrica": m, "strategia_a": a, "strategia_b": b, "gruppi": len(comuni),
                           "differenza_media": float(np.mean(diff)) if diff else float("nan"), "p": p})
            pv.append(p if p == p else 1.0)
        for riga, ph in zip(blocco, holm(pv)):
            riga["p_holm"] = ph
        test.extend(blocco)

    print("\n=== Test di Wilcoxon appaiati su NDCG@3 (correzione di Holm) ===")
    for t in test:
        if t["metrica"] == "ndcg@3":
            print(f"  {t['strategia_a']:<20} vs {t['strategia_b']:<20} diff {t['differenza_media']:+.3f}  "
                  f"p {t['p']:.4f}  p Holm {t['p_holm']:.4f}")

    # --- costi e dipendenza dall'ordine
    coppie_arch = [r for r in next(iter(oracoli.values())).archivio.dati.values() if r["tipo"] == "coppia"]
    glob_arch = [r for r in next(iter(oracoli.values())).archivio.dati.values() if r["tipo"] == "globale"]
    esec = [r for r in coppie_arch if r.get("origine") == "esperimento"]
    print("\n=== Oracolo ===")
    print(f"  confronti archiviati {len(coppie_arch)} (di cui riusati dal torneo della pipeline "
          f"{len(coppie_arch) - len(esec)}), falliti {sum(r['fallito'] for r in coppie_arch)}")
    if coppie_arch:
        print(f"  vittorie in posizione A: {100 * np.mean([r['scelta'] == 'A' for r in coppie_arch]):.1f}%")
    idx = {(r["soggetto"], r["A"], r["B"]): r["scelta"] for r in coppie_arch}
    doppie = [(s, a, b) for (s, a, b) in idx if indice(a) < indice(b) and (s, b, a) in idx]
    if doppie:
        coerenti = sum((idx[(s, a, b)] == "A") == (idx[(s, b, a)] == "B") for s, a, b in doppie)
        print(f"  coppie ripetute in ordine inverso {len(doppie)}, esito coerente {100 * coerenti / len(doppie):.1f}%")
    print(f"  valutazioni globali {len(glob_arch)}, fallite {sum(r['fallito'] for r in glob_arch)}")
    n_g = sum(len(d["gruppi"]) for d in dati.values())
    costi = {"confronto esaustivo": sum(len(g) * (len(g) - 1) // 2 for d in dati.values() for g in d["gruppi"].values()),
             "ricerca lineare": sum(len(g) - 1 for d in dati.values() for g in d["gruppi"].values()),
             "torneo": sum(len(g) - 1 for d in dati.values() for g in d["gruppi"].values()),
             "valutazione globale": n_g}
    unit_c = float(np.median([r["secondi"] for r in coppie_arch])) if coppie_arch else float("nan")
    unit_g = float(np.median([r["secondi"] for r in glob_arch])) if glob_arch else float("nan")
    for rg in riepilogo:
        if rg["strategia"] in costi:
            rg["interrogazioni"] = costi[rg["strategia"]]
            rg["secondi_per_interrogazione"] = unit_g if rg["strategia"] == "valutazione globale" else unit_c

    campi_r = sorted({k for r in riepilogo for k in r}, key=lambda k: (k != "strategia", k))
    with open(cartella / "riepilogo_strategie.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campi_r, delimiter=";")
        w.writeheader()
        w.writerows(riepilogo)
    with open(cartella / "test_wilcoxon.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metrica", "strategia_a", "strategia_b", "gruppi",
                                          "differenza_media", "p", "p_holm"], delimiter=";")
        w.writeheader()
        w.writerows(test)
    if accordo:
        with open(cartella / "accordo_valutatori.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["soggetto", "valutatori", "alfa", "esclusi"], delimiter=";")
            w.writeheader()
            w.writerows(accordo)
    print(f"\nFile scritti in {cartella}")


def diagnostica_valutatori(dati, cartella, args):
    """Per ogni compilazione (senza filtri): completezza, tempo per immagine, media e
    dispersione dei voti, e correlazione di Spearman con la media degli altri
    valutatori del medesimo soggetto."""
    from scipy.stats import spearmanr
    if not MAPPA.exists():
        return
    mappa = json.loads(MAPPA.read_text(encoding="utf-8"))
    righe = []
    for s in dati:
        info = mappa[s]
        risposte = [json.loads(l) for l in open(SONDAGGI / "data" / "responses" / f"{info['id']}.jsonl",
                                                  encoding="utf-8") if l.strip()]
        files = list(info["domande"].values())
        matrice = [[r["answers"].get(q, {}).get("rating") for q in info["domande"]] for r in risposte]
        for i, r in enumerate(risposte):
            propri = matrice[i]
            altri = [np.mean([matrice[j][k] for j in range(len(risposte)) if j != i and matrice[j][k] is not None])
                     for k in range(len(files))]
            coppie = [(a, b) for a, b in zip(propri, altri) if a is not None]
            rho = spearmanr([a for a, _ in coppie], [b for _, b in coppie]).statistic if len(coppie) > 2 else float("nan")
            n = len(r["answers"])
            voti = [a["rating"] for a in r["answers"].values()]
            righe.append({"soggetto": s, "valutatore": r["participant"].strip().lower(),
                          "votate": n, "totale": len(files),
                          "secondi_per_immagine": round((r.get("durationMs") or 0) / 1000 / max(n, 1), 2),
                          "media": round(float(np.mean(voti)), 3), "dev_std": round(float(np.std(voti)), 3),
                          "rho_con_altri": round(float(rho), 3)})
    with open(cartella / "valutatori.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0]), delimiter=";")
        w.writeheader()
        w.writerows(righe)
    print("\n=== Valutatori (tutte le compilazioni): tempo medio per immagine e rho con gli altri ===")
    for v in sorted({r["valutatore"] for r in righe}):
        rr = [r for r in righe if r["valutatore"] == v]
        print(f"  {v:<18} compilazioni {len(rr):>2}  s/immagine {np.median([r['secondi_per_immagine'] for r in rr]):.1f}"
              f"  rho mediana {np.nanmedian([r['rho_con_altri'] for r in rr]):.3f}"
              f"  (min {np.nanmin([r['rho_con_altri'] for r in rr]):.3f})")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Confronto delle strategie di selezione su gruppi da 20.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--piano", action="store_true", help="mostra gruppi, costi e stato del panel")
    ap.add_argument("--simulato", action="store_true", help="oracolo e voti simulati (prova del programma)")
    ap.add_argument("--solo-analisi", action="store_true", help="nessuna nuova interrogazione")
    ap.add_argument("--simmetrico", action="store_true", help="ripete ogni confronto anche in ordine B/A")
    ap.add_argument("--soggetti", help="sottoinsieme di soggetti, separati da virgola")
    ap.add_argument("--paralleli", type=int, default=1, help="interrogazioni concorrenti (predefinito 1)")
    ap.add_argument("--limite", type=int, default=0, help="numero massimo di nuove interrogazioni (per prove)")
    ap.add_argument("--prove-casuali", type=int, default=2000, help="ordinamenti casuali per il riferimento")
    ap.add_argument("--min-secondi", type=float, default=MIN_SECONDI_PER_IMMAGINE,
                    help="tempo medio minimo per immagine di una compilazione valida (0 = nessun filtro)")
    ap.add_argument("--min-completezza", type=float, default=MIN_COMPLETEZZA,
                    help="quota minima di immagini votate di una compilazione valida")
    ap.add_argument("--nome-analisi", default="", help="sottocartella per i risultati dell'analisi (varianti)")
    args = ap.parse_args()
    globals()["MIN_SECONDI_PER_IMMAGINE"] = args.min_secondi
    globals()["MIN_COMPLETEZZA"] = args.min_completezza

    soggetti = [s.strip() for s in args.soggetti.split(",")] if args.soggetti else \
        sorted(p.name for p in RISULTATI.iterdir() if (p / "03_post_elaborazione.csv").exists())
    dati = {s: carica_soggetto(s) for s in soggetti}

    cartella = PIPELINE / ("Esperimento_strategie_simulato" if args.simulato else "Esperimento_strategie")
    cartella.mkdir(exist_ok=True)
    archivio = Archivio(cartella / "archivio_interrogazioni.jsonl")
    json.dump({s: d["gruppi"] for s, d in dati.items()},
              open(cartella / "gruppi.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    rng = np.random.default_rng(SEME)
    panel = panel_simulato(dati, rng) if args.simulato else carica_panel(soggetti)
    if not args.simulato:
        riusati = importa_torneo_pipeline(archivio, soggetti)
        if riusati:
            print(f"Riutilizzati {riusati} confronti gia' eseguiti dal torneo della pipeline.")
    oracoli = {s: Oracolo(s, dati[s], archivio, simulato=panel[s]["qualita"] if args.simulato else None)
               for s in soggetti}

    n_gruppi = sum(len(d["gruppi"]) for d in dati.values())
    n_coppie = sum(len(g) * (len(g) - 1) // 2 for d in dati.values() for g in d["gruppi"].values())
    presenti = sum(1 for r in archivio.dati.values() if r["tipo"] == "coppia")
    print(f"Soggetti {len(soggetti)}, gruppi {n_gruppi}, confronti necessari {n_coppie * (2 if args.simmetrico else 1)} "
          f"(gia' archiviati {presenti}), valutazioni globali {n_gruppi}")
    if args.piano:
        for s, d in dati.items():
            stato = "voti presenti" if panel.get(s) else "voti non ancora disponibili"
            print(f"  {s:<12} " + "  ".join(f"{gid.split('-')[1]}:{len(g)}" for gid, g in d["gruppi"].items())
                  + f"   ({stato})")
        return

    if not args.solo_analisi:
        if not args.simulato:
            try:
                import requests
                requests.get(f"{pl.OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
            except Exception:
                sys.exit(f"Ollama non raggiungibile su {pl.OLLAMA_URL}: avvialo e rilancia il comando.")
        try:
            esegui_interrogazioni(dati, oracoli, args)
        except ErroreServizio as e:
            sys.exit(f"\nInterrotto per un errore del servizio del modello:\n  {str(e)[:300]}\n"
                     "Le risposte gia' ottenute sono salvate: rilancia lo stesso comando per riprendere.")

    if all(panel.get(s) is None for s in soggetti):
        print("\nVoti del panel non ancora disponibili: analisi rimandata. "
              "Quando i sondaggi sono completati: python esperimento_strategie.py --solo-analisi")
        return
    cartella_analisi = cartella / args.nome_analisi if args.nome_analisi else cartella
    cartella_analisi.mkdir(exist_ok=True)
    diagnostica_valutatori(dati, cartella_analisi, args)
    analizza(dati, oracoli, panel, cartella_analisi, args)


if __name__ == "__main__":
    main()
