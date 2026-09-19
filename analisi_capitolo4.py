"""
Analisi di approfondimento per le Sezioni 4.5-4.8 e figure del Capitolo 4.

Usa l'archivio di esperimento_strategie.py e i voti dei sondaggi (tutti i
valutatori con compilazione completa; nessun filtro sul tempo).

Produce in Esperimento_strategie/capitolo4/:
  ndcg_k.pdf                NDCG@k (k = 1..10) per strategia, con IC 95% e riferimento casuale
  costo_accordo.pdf         NDCG@1 e NDCG@3 contro interrogazioni per gruppo
  accuratezza_coppie.pdf    accordo dell'oracolo con la preferenza umana in funzione del divario
  esempi_disaccordo.png     primo scelto dal sistema e dagli umani in due gruppi
  report.txt                tutti i numeri citati nel testo

    python analisi_capitolo4.py
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

import esperimento_strategie as es

sys.stdout.reconfigure(encoding="utf-8")
es.MIN_SECONDI_PER_IMMAGINE = 0.0
OUT = es.PIPELINE / "Esperimento_strategie" / "capitolo4"
OUT.mkdir(parents=True, exist_ok=True)
righe_report = []


def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    righe_report.append(s)


soggetti = sorted(p.name for p in es.RISULTATI.iterdir() if (p / "03_post_elaborazione.csv").exists())
dati = {s: es.carica_soggetto(s) for s in soggetti}
panel = es.carica_panel(soggetti)
archivio = es.Archivio(es.PIPELINE / "Esperimento_strategie" / "archivio_interrogazioni.jsonl")
oracoli = {s: es.Oracolo(s, dati[s], archivio) for s in soggetti}
rng = np.random.default_rng(es.SEME)
NOMI = es.NOMI_STRATEGIE
ETICHETTE = {"confronto esaustivo": "Confronto esaustivo", "ricerca lineare": "Ricerca lineare",
             "torneo": "Torneo", "valutazione globale": "Valutazione globale"}

# ---------------------------------------------------------------- per gruppo
gruppi = []
for s in soggetti:
    for gid, g in dati[s]["gruppi"].items():
        voti = {f: panel[s]["voti"][f] for f in g}
        medie = {f: float(np.mean([x for x in v if x is not None])) for f, v in voti.items()}
        punt = {n: es.FUNZIONI[n](oracoli[s], g) for n in NOMI}
        gruppi.append({"s": s, "gid": gid, "g": g, "medie": medie, "voti": voti, "punt": punt})
rep(f"Gruppi {len(gruppi)}, candidati {sum(len(x['g']) for x in gruppi)}")

# ------------------------------------------------------------ panel e accordo
rep("\n== PANEL")
tutti_item, valutatori = [], sorted({v.strip().lower() for s in soggetti for v in panel[s]["valutatori"]})
for s in soggetti:
    nomi = [v.strip().lower() for v in panel[s]["valutatori"]]
    for f, vs in panel[s]["voti"].items():
        riga = [None] * len(valutatori)
        for n, x in zip(nomi, vs):
            riga[valutatori.index(n)] = x
        tutti_item.append(riga)
rep(f"valutatori {len(valutatori)}: {valutatori}")
rep(f"alfa complessivo (794 candidati x 7 valutatori): {es.alfa_krippendorff(tutti_item):.3f}")
for s in soggetti:
    a = es.alfa_krippendorff(list(panel[s]["voti"].values()))
    tutti_voti = [x for v in panel[s]["voti"].values() for x in v if x is not None]
    medie_s = [np.mean([x for x in v if x is not None]) for v in panel[s]["voti"].values()]
    rep(f"  {s:<12} alfa {a:.3f}  voto medio {np.mean(tutti_voti):.2f}  dev.std {np.std(tutti_voti):.2f}  "
        f"medie per immagine: min {min(medie_s):.2f} max {max(medie_s):.2f}  durate(min) {panel[s]['durate_min']}")
medie_val = {}
for i, v in enumerate(valutatori):
    xs = [r[i] for r in tutti_item if r[i] is not None]
    medie_val[v] = (np.mean(xs), np.std(xs))
rep("media e dev.std dei voti per valutatore: " + ", ".join(f"{v} {m:.2f}/{d:.2f}" for v, (m, d) in medie_val.items()))
# correlazioni di Spearman fra coppie di valutatori, per soggetto
rho_coppie = []
for s in soggetti:
    mat = np.array([[x if x is not None else np.nan for x in v] for v in panel[s]["voti"].values()], dtype=float)
    for i in range(mat.shape[1]):
        for j in range(i + 1, mat.shape[1]):
            ok = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
            rho_coppie.append(spearmanr(mat[ok, i], mat[ok, j]).statistic)
rep(f"rho di Spearman fra coppie di valutatori: mediana {np.nanmedian(rho_coppie):.3f}, "
    f"IQR {np.nanpercentile(rho_coppie, 25):.3f}-{np.nanpercentile(rho_coppie, 75):.3f}, n {len(rho_coppie)}")
# alfa dopo standardizzazione per valutatore (effetto della severita' individuale)
z_item = []
for riga in tutti_item:
    z_item.append([None if x is None else (x - medie_val[valutatori[i]][0]) / medie_val[valutatori[i]][1]
                   for i, x in enumerate(riga)])
rep(f"alfa complessivo su voti standardizzati per valutatore: {es.alfa_krippendorff(z_item):.3f}")

# ------------------------------------------------------------ NDCG@k 1..10
rep("\n== NDCG@k")
K = list(range(1, 11))
curve = {n: {k: [] for k in K} for n in NOMI}
casuale = {k: [] for k in K}
for x in gruppi:
    guad = {f: m - 1 for f, m in x["medie"].items()}
    media_rel = np.mean(list(guad.values()))
    for k in K:
        for n in NOMI:
            curve[n][k].append(es.ndcg(x["punt"][n], guad, k))
        ideale = es.dcg(dict(guad), guad, k)
        casuale[k].append(media_rel * sum(1 / math.log2(p + 1) for p in range(1, k + 1)) / ideale)
for n in NOMI:
    rep(f"  {n:<20} " + " ".join(f"@{k}:{np.mean(curve[n][k]):.3f}" for k in K))
rep(f"  {'casuale (esatto)':<20} " + " ".join(f"@{k}:{np.mean(casuale[k]):.3f}" for k in K))

# ------------------------------------------------------------ per soggetto
rep("\n== NDCG@3 PER SOGGETTO (media dei gruppi)")
for s in soggetti:
    gg = [x for x in gruppi if x["s"] == s]
    rep(f"  {s:<12} " + "  ".join(f"{n[:10]} {np.mean([es.ndcg(x['punt'][n], {f: m - 1 for f, m in x['medie'].items()}, 3) for x in gg]):.3f}"
                              for n in NOMI)
        + f"  casuale {np.mean([casuale[3][i] for i, x in enumerate(gruppi) if x['s'] == s]):.3f}")

# ------------------------------------------------------------ vincitori
rep("\n== PRIMO SCELTO")
stessi, rango_umano = 0, {n: [] for n in NOMI}
for x in gruppi:
    ve = [f for f, p in x["punt"]["confronto esaustivo"].items() if p == max(x["punt"]["confronto esaustivo"].values())]
    vt = max(x["punt"]["torneo"], key=x["punt"]["torneo"].get)
    stessi += vt in ve
    ordinati = sorted(x["medie"].values(), reverse=True)
    for n in NOMI:
        top = [f for f, p in x["punt"][n].items() if p == max(x["punt"][n].values())]
        rango_umano[n].append(np.mean([1 + sum(m > x["medie"][f] for m in ordinati) for f in top]))
rep(f"vincitore del torneo fra i primi a pari merito del confronto esaustivo: {stessi}/{len(gruppi)}")
for n in NOMI:
    rep(f"  {n:<20} rango umano del primo scelto: mediana {np.median(rango_umano[n]):.1f}, media {np.mean(rango_umano[n]):.2f}")
rep(f"  casuale: rango atteso {np.mean([(len(x['g']) + 1) / 2 for x in gruppi]):.2f}")
pari = [sum(p == max(x["punt"]["confronto esaustivo"].values()) for p in x["punt"]["confronto esaustivo"].values()) for x in gruppi]
rep(f"pari merito al primo posto nel confronto esaustivo: gruppi con piu' di uno {sum(p > 1 for p in pari)}, massimo {max(pari)}")

# ------------------------------------------------------------ coppie
rep("\n== ORACOLO SULLE COPPIE")
coppie = []
for x in gruppi:
    for a, b in es.coppie_esaustivo(x["g"]):
        r = oracoli[x["s"]].confronto(a, b, esegui=False)
        d = x["medie"][a] - x["medie"][b]
        coppie.append({"d": d, "scelta": r["scelta"], "s": x["s"]})
n_a = sum(c["scelta"] == "A" for c in coppie)
rep(f"coppie interne ai gruppi {len(coppie)}, vince A {n_a} ({100 * n_a / len(coppie):.1f}%), "
    f"binomiale p = {binomtest(n_a, len(coppie)).pvalue:.2e}")
umani_a = sum(c["d"] > 0 for c in coppie); umani_pari = sum(c["d"] == 0 for c in coppie)
rep(f"secondo gli umani A e' migliore in {umani_a} coppie ({100 * umani_a / len(coppie):.1f}%), pari {umani_pari}")
non_pari = [c for c in coppie if c["d"] != 0]
acc = np.mean([(c["d"] > 0) == (c["scelta"] == "A") for c in non_pari])
rep(f"accordo dell'oracolo con la preferenza umana (coppie non pari): {100 * acc:.1f}% su {len(non_pari)}")
fasce = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 5)]
tab_fasce = []
for lo, hi in fasce:
    cc = [c for c in non_pari if lo < abs(c["d"]) <= hi] if lo == 0 else [c for c in non_pari if lo < abs(c["d"]) <= hi]
    if not cc:
        continue
    a_ = np.mean([(c["d"] > 0) == (c["scelta"] == "A") for c in cc])
    a_meglio = [c for c in cc if c["d"] > 0]; b_meglio = [c for c in cc if c["d"] < 0]
    acc_a = np.mean([c["scelta"] == "A" for c in a_meglio]); acc_b = np.mean([c["scelta"] == "B" for c in b_meglio])
    lo_ic, hi_ic = binomtest(int(round(a_ * len(cc))), len(cc)).proportion_ci()
    tab_fasce.append((lo, hi, len(cc), a_, lo_ic, hi_ic, acc_a, acc_b))
    rep(f"  |d| in ({lo},{hi}]: n {len(cc)}, accordo {100 * a_:.1f}% [{100 * lo_ic:.1f},{100 * hi_ic:.1f}]  "
        f"se A migliore {100 * acc_a:.1f}%  se B migliore {100 * acc_b:.1f}%")

# ------------------------------------------------------------ torneo della pipeline
rep("\n== VINCITORI DEL TORNEO DELLA PIPELINE")
pct, quartile, medie_v, medie_tutte = [], 0, [], []
for s in soggetti:
    medie_s = {f: float(np.mean([x for x in v if x is not None])) for f, v in panel[s]["voti"].items()}
    valori = sorted(medie_s.values())
    medie_tutte += valori
    for fv in (es.RISULTATI / s / "vincitrici").iterdir():
        f = fv.name.split("_")[-1]
        p = (sum(v < medie_s[f] for v in valori) + 0.5 * (sum(v == medie_s[f] for v in valori) - 1)) / (len(valori) - 1)
        pct.append(p); quartile += p >= 0.75; medie_v.append(medie_s[f])
rep(f"vincitori {len(pct)}: voto medio {np.mean(medie_v):.2f} contro {np.mean(medie_tutte):.2f} di tutti; "
    f"percentile medio {100 * np.mean(pct):.1f}; nel quartile superiore {quartile}/{len(pct)}")

# ------------------------------------------------------------ casi
rep("\n== CASI")
x = next(x for x in gruppi if x["s"] == "Pirata" and "026.png" in x["g"])
pe = x["punt"]["confronto esaustivo"]
rep(f"Pirata 026: media umana {x['medie']['026.png']:.2f} (rango {1 + sum(m > x['medie']['026.png'] for m in x['medie'].values())}/{len(x['g'])}), "
    f"vittorie {pe['026.png']}/{len(x['g']) - 1} (rango sistema {1 + sum(v > pe['026.png'] for v in pe.values())}), "
    f"turno torneo {x['punt']['torneo']['026.png']}")
peggiori = sorted(gruppi, key=lambda x: es.ndcg(x["punt"]["confronto esaustivo"], {f: m - 1 for f, m in x["medie"].items()}, 1))[:2]
esempi = []
for x in peggiori:
    pe = x["punt"]["confronto esaustivo"]
    fs = max(x["g"], key=lambda f: (pe[f], x["medie"][f]))
    fu = max(x["g"], key=lambda f: (x["medie"][f], pe[f]))
    mot = [r["motivazione"] for r in archivio.dati.values() if r["tipo"] == "coppia" and r["soggetto"] == x["s"]
           and {r["A"], r["B"]} == {fs, fu}]
    rep(f"{x['gid']}: sistema {fs} (vittorie {pe[fs]}, media umana {x['medie'][fs]:.2f}); "
        f"umani {fu} (media {x['medie'][fu]:.2f}, vittorie {pe[fu]}); motivazione del confronto diretto: {mot}")
    esempi.append((x, fs, fu))

# ------------------------------------------------------------ costi
rep("\n== COSTI")
co = [r for r in archivio.dati.values() if r["tipo"] == "coppia" and r.get("origine") == "esperimento"]
gl = [r for r in archivio.dati.values() if r["tipo"] == "globale"]
n_es = sum(len(x["g"]) * (len(x["g"]) - 1) // 2 for x in gruppi)
n_lin = sum(len(x["g"]) - 1 for x in gruppi)
rep(f"interrogazioni: esaustivo {n_es} ({n_es / len(gruppi):.1f} per gruppo), lineare e torneo {n_lin} "
    f"({n_lin / len(gruppi):.2f}), globale {len(gruppi)}")
rep(f"tempo coppia: mediana {np.median([r['secondi'] for r in co]):.1f} s, p90 {np.percentile([r['secondi'] for r in co], 90):.1f} s, "
    f"totale eseguite {len(co)} in {sum(r['secondi'] for r in co) / 3600:.1f} h (somma dei tempi)")
rep(f"tempo globale: mediana {np.median([r['secondi'] for r in gl]):.1f} s, min {min(r['secondi'] for r in gl):.1f}, max {max(r['secondi'] for r in gl):.1f}")
tempo_coppia = float(np.median([r['secondi'] for r in co]))
tempo_glob = float(np.median([r['secondi'] for r in gl]))
rep(f"tempo per gruppo stimato: esaustivo {n_es / len(gruppi) * tempo_coppia / 60:.1f} min, lineare/torneo "
    f"{n_lin / len(gruppi) * tempo_coppia / 60:.1f} min, globale {tempo_glob / 60:.2f} min")

(OUT / "report.txt").write_text("\n".join(righe_report), encoding="utf-8")

# ================================================================ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "DejaVu Sans"
COL = {"confronto esaustivo": "#2f5d7c", "ricerca lineare": "#c4843a", "torneo": "#5d8f4e",
       "valutazione globale": "#9a4f7a"}
MARK = {"confronto esaustivo": "o", "ricerca lineare": "s", "torneo": "^", "valutazione globale": "D"}
virgola = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

# figura 1: NDCG@k
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for n in NOMI:
    m = [np.mean(curve[n][k]) for k in K]
    ic = [es.bootstrap_ic(curve[n][k], rng, 5000) for k in K]
    ax.fill_between(K, [a for a, _ in ic], [b for _, b in ic], color=COL[n], alpha=0.15, linewidth=0)
    ax.plot(K, m, marker=MARK[n], color=COL[n], label=ETICHETTE[n], linewidth=1.6, markersize=5)
ax.plot(K, [np.mean(casuale[k]) for k in K], color="#555555", linestyle=(0, (4, 2.4)), linewidth=1.3,
        label="Ordinamento casuale")
ax.set_xticks(K)
ax.set_xlabel("$k$", fontsize=10)
ax.set_ylabel("NDCG@$k$ (media sui 40 gruppi)", fontsize=10)
ax.yaxis.set_major_formatter(virgola)
ax.grid(axis="y", color="#dddddd", linewidth=0.6)
for lato in ("top", "right"):
    ax.spines[lato].set_visible(False)
ax.legend(frameon=False, fontsize=8.8, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "ndcg_k.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "ndcg_k.png", dpi=180, bbox_inches="tight", pad_inches=0.03)

# figura 2: costo e accordo
costo = {"confronto esaustivo": n_es / len(gruppi), "ricerca lineare": n_lin / len(gruppi),
         "torneo": n_lin / len(gruppi), "valutazione globale": 1}
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
for ax, k in zip(axs, (1, 3)):
    for n in NOMI:
        m = np.mean(curve[n][k]); lo, hi = es.bootstrap_ic(curve[n][k], rng, 5000)
        dx = {"ricerca lineare": 0.93, "torneo": 1.07}.get(n, 1.0)
        ax.errorbar(costo[n] * dx, m, yerr=[[m - lo], [hi - m]], fmt=MARK[n], color=COL[n], capsize=3,
                    markersize=6, label=ETICHETTE[n])
    ax.axhline(np.mean(casuale[k]), color="#555555", linestyle=(0, (4, 2.4)), linewidth=1.1,
               label="Ordinamento casuale")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("interrogazioni per gruppo (scala log.)", fontsize=9.5)
    ax.set_title(f"NDCG@{k}", fontsize=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(virgola)
    ax.grid(color="#e3e3e3", linewidth=0.6)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
axs[0].legend(frameon=False, fontsize=7.8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "costo_accordo.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "costo_accordo.png", dpi=180, bbox_inches="tight", pad_inches=0.03)

# figura 3: accordo sulle coppie per divario umano
fig, ax = plt.subplots(figsize=(8.2, 3.6))
xs = np.arange(len(tab_fasce))
etich = [f"{str(lo).replace('.', ',')}–{str(hi).replace('.', ',')}" if hi < 5 else f"> {str(lo).replace('.', ',')}"
         for lo, hi, *_ in tab_fasce]
acc_v = [t[3] for t in tab_fasce]
ax.bar(xs, acc_v, color="#5b7f95", width=0.55, label="accordo complessivo")
ax.errorbar(xs, acc_v, yerr=[[a - t[4] for a, t in zip(acc_v, tab_fasce)], [t[5] - a for a, t in zip(acc_v, tab_fasce)]],
            fmt="none", color="#1b1b1b", capsize=3, linewidth=1)
ax.plot(xs - 0.12, [t[6] for t in tab_fasce], "o", color="#2f5d7c", markersize=5, label="umani preferiscono A")
ax.plot(xs + 0.12, [t[7] for t in tab_fasce], "s", color="#b5543c", markersize=5, label="umani preferiscono B")
for xi, t in zip(xs, tab_fasce):
    ax.text(xi, 0.03, f"n = {t[2]}", ha="center", va="bottom", fontsize=8, color="white")
ax.axhline(0.5, color="#555555", linestyle=(0, (4, 2.4)), linewidth=1)
ax.set_xticks(xs)
ax.set_xticklabels(etich)
ax.set_ylim(0, 1)
ax.set_xlabel("differenza fra i voti medi umani dei due candidati", fontsize=9.5)
ax.set_ylabel("confronti concordi con gli umani", fontsize=9.5)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
for lato in ("top", "right"):
    ax.spines[lato].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "accuratezza_coppie.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "accuratezza_coppie.png", dpi=180, bbox_inches="tight", pad_inches=0.03)

# figura 4: esempi di disaccordo
L, T, M = 360, 58, 18
tela = Image.new("RGB", (2 * L + M, len(esempi) * (L + T) + (len(esempi) - 1) * M), "white")
from PIL import ImageDraw, ImageFont
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
except OSError:
    font = ImageFont.truetype("arial.ttf", 17)
d = ImageDraw.Draw(tela)
for i, (x, fs, fu) in enumerate(esempi):
    y = i * (L + T + M)
    pe = x["punt"]["confronto esaustivo"]
    for j, (f, tit) in enumerate(((fs, "primo per il sistema"), (fu, "primo per gli umani"))):
        img = Image.open(es.RISULTATI / x["s"] / "ingrandite" / f).convert("RGB").resize((L, L), Image.NEAREST)
        tela.paste(img, (j * (L + M), y + T))
        d.text((j * (L + M) + 2, y + 2), f"{x['gid']} {f[:3]} - {tit}", fill="black", font=font)
        d.text((j * (L + M) + 2, y + 24),
               f"vittorie {pe[f]}/{len(x['g']) - 1}, voto medio {x['medie'][f]:.2f}".replace(".", ","), fill="#444444", font=font)
tela.save(OUT / "esempi_disaccordo.png")
print(f"\nFigure e report in {OUT}")
