"""
Analisi del questionario sulle animazioni (Capitolo 5, Sezione 5.8).

Legge sondaggi_video.json, le risposte dell'app e Risultati_video/metriche_video.csv.
Produce in Risultati_video/analisi/:
  voti_per_animazione.csv   media di ciascun aspetto per animazione e versione
  report.txt                tutti i numeri citati nel testo
  voti_livelli.pdf          medie per aspetto, livello e versione con IC 95%
  stabilita_sfarfallio.pdf  voto di stabilita' contro sfarfallio misurato
  modi_fallimento.png       fotogrammi di esempio dei modi di fallimento

    python analisi_video.py
"""

import os
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr

import esperimento_strategie as es

sys.stdout.reconfigure(encoding="utf-8")
PIPELINE = Path(__file__).resolve().parent
VIDEO = PIPELINE / "Risultati_video"
OUT = VIDEO / "analisi"
OUT.mkdir(exist_ok=True)
SONDAGGI = Path(os.environ.get("SONDAGGI_TESI", Path(__file__).resolve().parent.parent / "Sondaggi tesi"))
CRIT = ["C1", "C2", "C3", "C4", "C5"]
NOMI_C = {"C1": "Identità", "C2": "Movimento richiesto", "C3": "Fluidità", "C4": "Stabilità", "C5": "Pixel art e utilizzo"}
LIV = ["L1", "L2", "L3"]
rep_righe = []


def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    rep_righe.append(s)


# Alias facoltativi per unificare lo stesso valutatore firmato in modi diversi:
# file alias_valutatori.json accanto allo script, {"firma": "nome_unificato"}.
# Non incluso nel repository per non pubblicare i nomi dei partecipanti.
_ALIAS_FILE = Path(__file__).resolve().parent / "alias_valutatori.json"
ALIAS = json.loads(_ALIAS_FILE.read_text(encoding="utf-8")) if _ALIAS_FILE.exists() else {}


def valutatore(nome):
    n = nome.strip().lower()
    return ALIAS.get(n, n)


def holm(p):
    o = sorted(range(len(p)), key=lambda i: p[i]); out, m = [0.0] * len(p), 0.0
    for r, i in enumerate(o):
        m = max(m, min(1.0, (len(p) - r) * p[i])); out[i] = m
    return out


# ------------------------------------------------------------------ dati
mappa = json.loads((PIPELINE / "sondaggi_video.json").read_text(encoding="utf-8"))
voti = []          # (soggetto, livello, versione, valutatore, {C: voto}, secondi compilazione)
for s, info in mappa.items():
    for l in open(SONDAGGI / "data" / "responses" / f"{info['id']}.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        for qid, a in r["answers"].items():
            d = info["domande"][qid]
            voti.append((s, d["livello"], d["versione"], valutatore(r["participant"]), a["ratings"],
                         (r.get("durationMs") or 0) / 1000))
valutatori = sorted({v[3] for v in voti})
soggetti = sorted(mappa)
rep(f"voti registrati: {len(voti)} animazioni-valutatore, valutatori {len(valutatori)}: {valutatori}")

# tempi per valutatore e coerenza con gli altri (come nel Capitolo 4)
rep("\n== VALUTATORI")
durate = {}
for s, info in mappa.items():
    for l in open(SONDAGGI / "data" / "responses" / f"{info['id']}.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            durate.setdefault(valutatore(r["participant"]), []).append((r.get("durationMs") or 0) / 60000)
tutte = [d for v in durate.values() for d in v]
rep(f"compilazioni {len(tutte)}, durata mediana {np.median(tutte):.1f} min (min {min(tutte):.1f}, max {max(tutte):.1f}), "
    f"totale {sum(tutte) / 60:.1f} h; secondi per voto (mediana) {np.median(tutte) * 60 / 30:.1f}")
chiave = lambda v: (v[0], v[1], v[2])
for v in valutatori:
    rho = []
    for c in CRIT:
        propri, altri = [], []
        for k in {chiave(x) for x in voti}:
            mine = [x[4][c] for x in voti if chiave(x) == k and x[3] == v]
            others = [x[4][c] for x in voti if chiave(x) == k and x[3] != v]
            if mine and others:
                propri.append(mine[0]); altri.append(np.mean(others))
        rho.append(spearmanr(propri, altri).statistic)
    rep(f"  {v:<18} durata mediana {np.median(durate[v]):5.1f} min  rho con gli altri per aspetto "
        + " ".join(f"{c} {r:.2f}" for c, r in zip(CRIT, rho)) + f"  media {np.nanmean(rho):.2f}")

# ------------------------------------------------------------------ medie per animazione
chiavi = sorted({chiave(x) for x in voti})
medie = {k: {c: float(np.mean([x[4][c] for x in voti if chiave(x) == k])) for c in CRIT} for k in chiavi}
with open(OUT / "voti_per_animazione.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["soggetto", "livello", "versione"] + CRIT)
    for k in chiavi:
        w.writerow(list(k) + [f"{medie[k][c]:.3f}" for c in CRIT])

rep("\n== MEDIE PER LIVELLO E VERSIONE (media delle medie per animazione, n = 10)")
tab = {}
for l in LIV:
    for ver in ("A", "B"):
        vals = {c: [medie[(s, l, ver)][c] for s in soggetti] for c in CRIT}
        tab[(l, ver)] = vals
        rep(f"  {l} {ver}: " + "  ".join(f"{c} {np.mean(vals[c]):.2f}" for c in CRIT))
for ver in ("A", "B"):
    rep(f"  tutti {ver}: " + "  ".join(f"{c} {np.mean([medie[(s, l, ver)][c] for s in soggetti for l in LIV]):.2f}" for c in CRIT))
rep("  complessivo (A e B): " + "  ".join(f"{c} {np.mean([medie[k][c] for k in chiavi]):.2f}" for c in CRIT))

# ------------------------------------------------------------------ A contro B
rep("\n== A CONTRO B (30 coppie di animazioni, Wilcoxon, Holm sui 5 aspetti)")
pv, righe_ab = [], []
for c in CRIT:
    a = np.array([medie[(s, l, "A")][c] for s in soggetti for l in LIV])
    b = np.array([medie[(s, l, "B")][c] for s in soggetti for l in LIV])
    d = b - a
    p = float(wilcoxon(b, a).pvalue) if np.any(d != 0) else 1.0
    pv.append(p)
    righe_ab.append((c, a.mean(), b.mean(), d.mean(), int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()), p))
for (c, ma, mb, dm, piu, meno, pari, p), ph in zip(righe_ab, holm(pv)):
    rep(f"  {c} {NOMI_C[c]:<22} A {ma:.2f}  B {mb:.2f}  diff {dm:+.3f}  B meglio {piu}, A meglio {meno}, pari {pari}  p {p:.3f}  p Holm {ph:.3f}")
# per singolo valutatore: quante volte preferisce B nella stabilita'
rep("  stabilita' (C4), confronto per valutatore sulle 30 coppie:")
for v in valutatori:
    diff = []
    for s in soggetti:
        for l in LIV:
            va = [x[4]["C4"] for x in voti if x[:3] == (s, l, "A") and x[3] == v]
            vb = [x[4]["C4"] for x in voti if x[:3] == (s, l, "B") and x[3] == v]
            if va and vb:
                diff.append(vb[0] - va[0])
    diff = np.array(diff)
    rep(f"    {v:<18} B>A {int((diff > 0).sum()):2d}  A>B {int((diff < 0).sum()):2d}  uguali {int((diff == 0).sum()):2d}")

# ------------------------------------------------------------------ livelli
rep("\n== EFFETTO DEL LIVELLO (media di A e B per soggetto; Friedman su 10 soggetti; Wilcoxon a coppie con Holm)")
for c in CRIT:
    x = {l: np.array([(medie[(s, l, "A")][c] + medie[(s, l, "B")][c]) / 2 for s in soggetti]) for l in LIV}
    fr = friedmanchisquare(x["L1"], x["L2"], x["L3"]).pvalue
    coppie = [("L1", "L2"), ("L1", "L3"), ("L2", "L3")]
    pp = [float(wilcoxon(x[a], x[b]).pvalue) for a, b in coppie]
    rep(f"  {c} {NOMI_C[c]:<22} L1 {x['L1'].mean():.2f}  L2 {x['L2'].mean():.2f}  L3 {x['L3'].mean():.2f}  Friedman p {fr:.4f}  "
        + "  ".join(f"{a}-{b} pH {h:.3f}" for (a, b), h in zip(coppie, holm(pp))))

# ------------------------------------------------------------------ accordo
rep("\n== ACCORDO FRA VALUTATORI (alfa di Krippendorff, intervallare, 60 animazioni)")
for c in CRIT:
    mat = [[next((x[4][c] for x in voti if chiave(x) == k and x[3] == v), None) for v in valutatori] for k in chiavi]
    rep(f"  {c} {NOMI_C[c]:<22} alfa {es.alfa_krippendorff(mat):.3f}")
mat_all = [[next((x[4][c] for x in voti if chiave(x) == k and x[3] == v), None) for v in valutatori] for k in chiavi for c in CRIT]
rep(f"  tutti gli aspetti insieme: alfa {es.alfa_krippendorff(mat_all):.3f}")
# correlazioni fra aspetti
rep("  correlazioni di Spearman fra aspetti (60 animazioni):")
for i, c1 in enumerate(CRIT):
    rep("    " + c1 + " " + " ".join(f"{spearmanr([medie[k][c1] for k in chiavi], [medie[k][c2] for k in chiavi]).statistic:5.2f}" for c2 in CRIT))

# ------------------------------------------------------------------ metriche
rep("\n== MISURE AUTOMATICHE E VOTI (Spearman sulle 60 animazioni)")
met = {(r["soggetto"], r["livello"], r["versione"]): r for r in csv.DictReader(open(VIDEO / "metriche_video.csv", encoding="utf-8"), delimiter=";")}
grandezze = ["sfarf_sfondo", "sfarf_sogg", "deriva", "iou_media", "iou_min", "chiusura"]
for g in grandezze:
    xs = [float(met[k][g]) for k in chiavi]
    rep(f"  {g:<13} " + "  ".join(f"{c} {spearmanr(xs, [medie[k][c] for k in chiavi]).statistic:+.2f}" for c in CRIT))
rep("  solo versione B (30 animazioni), dove lo sfarfallio dello sfondo e' nullo:")
chiavi_b = [k for k in chiavi if k[2] == "B"]
for g in ("sfarf_sogg", "iou_media", "iou_min", "chiusura"):
    xs = [float(met[k][g]) for k in chiavi_b]
    rep(f"  {g:<13} " + "  ".join(f"{c} {spearmanr(xs, [medie[k][c] for k in chiavi_b]).statistic:+.2f}" for c in CRIT))
rep("\n== MISURE PER LIVELLO E VERSIONE")
for l in LIV:
    for ver in ("G", "A", "B"):
        rr = [met[(s, l, ver)] for s in soggetti]
        rep(f"  {l} {ver}: colori {np.mean([float(r['colori_medi']) for r in rr]):7.1f}  deriva {np.mean([float(r['deriva']) for r in rr]):.2f}  "
            f"sfondo {np.mean([float(r['sfarf_sfondo']) for r in rr]):.3f}  soggetto {np.mean([float(r['sfarf_sogg']) for r in rr]):.3f}  "
            f"iou {np.mean([float(r['iou_media']) for r in rr]):.3f} (min {np.min([float(r['iou_min']) for r in rr]):.3f})  "
            f"chiusura {np.mean([float(r['chiusura']) for r in rr]):.3f}"
            + (f"  griglia {np.mean([float(r['griglia_512']) for r in rr]):.3f}" if ver == "G" else ""))

# ------------------------------------------------------------------ casi
rep("\n== ANIMAZIONI PER VOTO COMPLESSIVO (C5, versione B)")
ordine = sorted(soggetti, key=lambda s: -np.mean([medie[(s, l, "B")]["C5"] for l in LIV]))
for s in ordine:
    rep(f"  {s:<12} " + "  ".join(f"{l}: " + " ".join(f"{medie[(s, l, 'B')][c]:.1f}" for c in CRIT) for l in LIV))
migliori = sorted(chiavi_b, key=lambda k: -medie[k]["C5"])
rep("  migliori (C5, B): " + ", ".join(f"{k[0]} {k[1]} {medie[k]['C5']:.2f}" for k in migliori[:5]))
rep("  peggiori (C5, B): " + ", ".join(f"{k[0]} {k[1]} {medie[k]['C5']:.2f}" for k in migliori[-5:]))
utilizzabili = [k for k in chiavi_b if medie[k]["C5"] >= 4]
rep(f"  animazioni B con C5 medio >= 4: {len(utilizzabili)} su 30; >= 3: {sum(medie[k]['C5'] >= 3 for k in chiavi_b)}")

(OUT / "report.txt").write_text("\n".join(rep_righe), encoding="utf-8")

# ================================================================== figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
plt.rcParams["font.family"] = "DejaVu Sans"
virg = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}" if v == int(v) else f"{v:.1f}".replace(".", ","))
rng = np.random.default_rng(2026)

fig, axs = plt.subplots(1, 5, figsize=(10.5, 3.4), sharey=True)
for ax, c in zip(axs, CRIT):
    for j, ver in enumerate(("A", "B")):
        m = [np.mean(tab[(l, ver)][c]) for l in LIV]
        ic = [es.bootstrap_ic(tab[(l, ver)][c], rng, 5000) for l in LIV]
        xs = np.arange(3) + (j - 0.5) * 0.36
        ax.bar(xs, m, width=0.34, color=("#c9a26b" if ver == "A" else "#2f5d7c"), label=f"versione {ver}")
        ax.errorbar(xs, m, yerr=[[a - lo for a, (lo, _) in zip(m, ic)], [hi - a for a, (_, hi) in zip(m, ic)]],
                    fmt="none", color="#1b1b1b", capsize=2, lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(LIV)
    ax.set_title(f"{c} {NOMI_C[c]}", fontsize=8.6)
    ax.set_ylim(1, 5); ax.yaxis.set_major_formatter(virg)
    ax.grid(axis="y", color="#e2e2e2", lw=0.6)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
axs[0].set_ylabel("voto medio")
maniglie, etichette = axs[0].get_legend_handles_labels()
fig.legend(maniglie, etichette, frameon=False, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "voti_livelli.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "voti_livelli.png", dpi=180, bbox_inches="tight", pad_inches=0.03)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
for ver, col, mk in (("A", "#c9a26b", "o"), ("B", "#2f5d7c", "s")):
    ks = [k for k in chiavi if k[2] == ver]
    ax.scatter([float(met[k]["sfarf_sfondo"]) for k in ks], [medie[k]["C4"] for k in ks], color=col, marker=mk,
               s=28, label=f"versione {ver}", edgecolor="white", lw=0.5)
ax.set_xlabel("sfarfallio dello sfondo misurato $\\Psi^{\\mathrm{sf}}$", fontsize=9.5)
ax.set_ylabel("voto medio di stabilità (C4)", fontsize=9.5)
ax.set_ylim(1, 5); ax.yaxis.set_major_formatter(virg)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
for lato in ("top", "right"):
    ax.spines[lato].set_visible(False)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "stabilita_sfarfallio.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "stabilita_sfarfallio.png", dpi=180, bbox_inches="tight", pad_inches=0.03)

# modi di fallimento: fotogrammi della versione B
casi = [("Cat", "L3", "Gatto L3: azione ampia eseguita (movimento 3,7)"),
        ("Ballerina", "L3", "Ballerina L3: piroetta non eseguita (movimento 1,4)"),
        ("Cuoco", "L3", "Cuoco L3: gesto diverso da quello richiesto (movimento 1,3)"),
        ("Pirata", "L3", "Pirata L3: rotazione tridimensionale del busto"),
        ("Gnomo", "L3", "Gnomo L3: oggetto introdotto dal modello")]
K, L, T = [0, 16, 32, 48, 64], 180, 30
tela = Image.new("RGB", (len(K) * (L + 6), len(casi) * (L + T + 8)), "white")
d = ImageDraw.Draw(tela)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
except OSError:
    font = ImageFont.truetype("arial.ttf", 16)
for r, (s, l, tit) in enumerate(casi):
    y = r * (L + T + 8)
    d.text((4, y + 5), tit, fill="black", font=font)
    for c, k in enumerate(K):
        im = Image.open(VIDEO / s / l / "B_128" / f"{k:03d}.png").resize((L, L), Image.NEAREST)
        tela.paste(im, (c * (L + 6), y + T))
tela.save(OUT / "modi_fallimento.png")
print(f"\nFile scritti in {OUT}")
