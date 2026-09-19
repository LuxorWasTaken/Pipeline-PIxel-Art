"""
Riduzione della palette di un'immagine, replica fedele del Color Reducer di
sprite-ai.art (https://www.sprite-ai.art/tools/color-reducer).

L'algoritmo e' ricostruito dal codice JavaScript del sito (funzione
quantizeColors) e ne riproduce ogni scelta:

  1. Istogramma dei colori RGBA distinti, nell'ordine di prima comparsa;
     i pixel completamente trasparenti (alpha = 0) sono esclusi.
  2. Median cut pesato: a ogni passo si divide la scatola con punteggio
     massimo, dove punteggio = peso * max(0.4375*SSE_R, 0.5625*SSE_G,
     0.3125*SSE_B); il taglio avviene sul canale di varianza pesata
     maggiore, nel punto in cui il conteggio cumulato raggiunge meta' del peso.
  3. Raffinamento k-means (al piu' 8 iterazioni) sui colori distinti pesati
     per frequenza, con distanza percettiva
        d = (0.4375 dR)^2 + (0.5625 dG)^2 + (0.3125 dB)^2 + (0.25 dA)^2.
  4. Ogni pixel e' sostituito dal colore di palette piu' vicino secondo la
     stessa distanza, senza dithering; i pixel trasparenti diventano (0,0,0,0).

Uso:
    python riduci_palette.py immagine.png
    python riduci_palette.py immagine.png -o ridotta.png --colori 16
    python riduci_palette.py cartella_immagini/ -o cartella_uscita/
    python riduci_palette.py immagine.png --palette palette.png --scala 32

Requisiti: pip install pillow numpy
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Pesi percettivi usati dal sito (R, G, B, A).
PESO_R, PESO_G, PESO_B, PESO_A = 0.4375, 0.5625, 0.3125, 0.25
MAX_ITERAZIONI_KMEANS = 8
ESTENSIONI = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def arrotonda_js(x):
    """Math.round di JavaScript: arrotonda .5 verso +infinito."""
    return int(math.floor(x + 0.5))


def somma_sequenziale(v):
    """Somma elemento per elemento nello stesso ordine del ciclo JavaScript
    (np.sum usa la somma a coppie e puo' differire nell'ultima cifra)."""
    return float(np.cumsum(v)[-1]) if len(v) else 0.0


# --------------------------------------------------------------------------
#  1. Istogramma
# --------------------------------------------------------------------------
def istogramma(rgba):
    """Restituisce i colori distinti (N x 4, int64) e i conteggi, nell'ordine
    di prima comparsa scandendo i pixel riga per riga."""
    pixel = rgba.reshape(-1, 4).astype(np.int64)
    pixel = pixel[pixel[:, 3] != 0]
    if len(pixel) == 0:
        return np.empty((0, 4), np.int64), np.empty(0, np.int64)
    chiavi = (pixel[:, 0] << 24) | (pixel[:, 1] << 16) | (pixel[:, 2] << 8) | pixel[:, 3]
    _, primo, conteggi = np.unique(chiavi, return_index=True, return_counts=True)
    ordine = np.argsort(primo, kind="stable")
    return pixel[primo[ordine]], conteggi[ordine]


# --------------------------------------------------------------------------
#  2. Median cut pesato
# --------------------------------------------------------------------------
def sse_pesati(colori, conteggi, idx):
    """Somme pesate dei quadrati degli scarti su R, G, B e peso totale."""
    c = colori[idx]
    w = conteggi[idx]
    peso = int(w.sum())
    medie = [somma_sequenziale((c[:, k] * w).astype(np.float64)) / peso for k in range(3)]
    sse = []
    for k in range(3):
        e = c[:, k].astype(np.float64) - medie[k]
        sse.append(somma_sequenziale(e * e * w))
    return sse, peso


def crea_scatola(colori, conteggi, idx):
    sse, peso = sse_pesati(colori, conteggi, idx)
    if peso == 0:
        punteggio = 0.0
    else:
        punteggio = peso * max(PESO_R * sse[0], PESO_G * sse[1], PESO_B * sse[2])
    return {"idx": idx, "peso": peso, "punteggio": punteggio}


def median_cut(colori, conteggi, n_colori):
    if len(colori) <= n_colori:
        return [list(map(int, c)) for c in colori]

    scatole = [crea_scatola(colori, conteggi, np.arange(len(colori)))]
    while len(scatole) < n_colori:
        # scatola con punteggio massimo (a parita', la prima)
        i_max = 0
        for i in range(1, len(scatole)):
            if scatole[i]["punteggio"] > scatole[i_max]["punteggio"]:
                i_max = i
        s = scatole[i_max]
        if len(s["idx"]) < 2 or s["punteggio"] == 0:
            break

        sse, _ = sse_pesati(colori, conteggi, s["idx"])
        n, l, b = PESO_R * sse[0], PESO_G * sse[1], PESO_B * sse[2]
        canale = 1 if (l >= n and l >= b) else (0 if n >= b else 2)

        idx = s["idx"]
        idx = idx[np.argsort(colori[idx, canale], kind="stable")]  # sort stabile come in JS
        meta = s["peso"] / 2
        cumulato = np.cumsum(conteggi[idx])
        e = int(np.argmax(cumulato >= meta))
        taglio = max(1, min(e + 1, len(idx) - 1))

        scatole[i_max:i_max + 1] = [
            crea_scatola(colori, conteggi, idx[:taglio]),
            crea_scatola(colori, conteggi, idx[taglio:]),
        ]

    palette = []
    for s in scatole:
        c = colori[s["idx"]]
        w = conteggi[s["idx"]]
        palette.append([arrotonda_js(float((c[:, k] * w).sum()) / s["peso"]) for k in range(4)])
    return palette


# --------------------------------------------------------------------------
#  3. Raffinamento k-means e distanza percettiva
# --------------------------------------------------------------------------
def distanze(colori, palette):
    """Matrice N x K delle distanze percettive, con lo stesso ordine delle
    operazioni del sito."""
    c = colori.astype(np.float64)[:, None, :]
    p = np.asarray(palette, dtype=np.float64)[None, :, :]
    o = (c[..., 0] - p[..., 0]) * PESO_R
    d = (c[..., 1] - p[..., 1]) * PESO_G
    k = (c[..., 2] - p[..., 2]) * PESO_B
    u = (c[..., 3] - p[..., 3]) * PESO_A
    return o * o + d * d + k * k + u * u


def kmeans(colori, conteggi, palette):
    palette = [list(p) for p in palette]
    n_k = len(palette)
    for _ in range(MAX_ITERAZIONI_KMEANS):
        assegnazione = distanze(colori, palette).argmin(axis=1)  # primo minimo, come in JS
        pesi = np.bincount(assegnazione, weights=conteggi, minlength=n_k)
        cambiato = False
        for j in range(n_k):
            if pesi[j] == 0:
                continue
            sel = assegnazione == j
            w = conteggi[sel]
            nuovo = [arrotonda_js(float((colori[sel, k] * w).sum()) / pesi[j]) for k in range(4)]
            if nuovo != palette[j]:
                palette[j] = nuovo
                cambiato = True
        if not cambiato:
            break
    return palette


# --------------------------------------------------------------------------
#  4. Mappatura
# --------------------------------------------------------------------------
def quantizza(rgba, n_colori=16):
    """Riceve un array H x W x 4 uint8 e restituisce (immagine ridotta, palette)."""
    colori, conteggi = istogramma(rgba)
    if len(colori) == 0:
        return rgba.copy(), []

    palette = kmeans(colori, conteggi, median_cut(colori, conteggi, n_colori))
    pal = np.asarray(palette, dtype=np.uint8)

    pixel = rgba.reshape(-1, 4)
    uscita = np.zeros_like(pixel)
    opachi = pixel[:, 3] != 0
    distinti, inversa = np.unique(pixel[opachi], axis=0, return_inverse=True)
    vicino = distanze(distinti.astype(np.int64), palette).argmin(axis=1)
    uscita[opachi] = pal[vicino[inversa.reshape(-1)]]
    return uscita.reshape(rgba.shape), palette


def esadecimale(c):
    return "#{:02X}{:02X}{:02X}".format(c[0], c[1], c[2])


def salva_palette(palette, percorso, scala):
    colori = [c for c in palette if c[3] > 0]
    img = Image.new("RGBA", (len(colori), 1))
    img.putdata([tuple(c) for c in colori])
    img.resize((len(colori) * scala, scala), Image.NEAREST).save(percorso)


def elabora(ingresso, uscita, args):
    img = Image.open(ingresso)
    if img.width > 512 or img.height > 512:
        print(f"  nota: {ingresso.name} e' {img.width}x{img.height}; il sito accetta al massimo "
              f"512x512, lo script la elabora comunque.")
    rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    ridotta, palette = quantizza(rgba, args.colori)
    Image.fromarray(ridotta, "RGBA").save(uscita)

    visibili = [c for c in palette if c[3] > 0]
    n_ingresso = len(np.unique(rgba.reshape(-1, 4), axis=0))
    n_uscita = len(np.unique(ridotta.reshape(-1, 4), axis=0))
    print(f"{ingresso.name}: {n_ingresso} colori -> {n_uscita} colori  => {uscita}")
    print("  palette: " + " ".join(esadecimale(c) for c in visibili))
    if args.palette:
        percorso = Path(args.palette)
        if ingresso != uscita and Path(args.input).is_dir():
            percorso = uscita.with_name(uscita.stem + "_palette.png")
        salva_palette(palette, percorso, args.scala)


def main():
    ap = argparse.ArgumentParser(description="Riduzione della palette come sprite-ai.art Color Reducer.")
    ap.add_argument("input", help="immagine o cartella di immagini")
    ap.add_argument("-o", "--output", help="file (o cartella) di uscita; predefinito: <nome>_<N>colori.png")
    ap.add_argument("-c", "--colori", type=int, default=16, help="numero di colori, 2-64 (predefinito 16)")
    ap.add_argument("--palette", help="salva anche la palette come immagine PNG")
    ap.add_argument("--scala", type=int, default=32, help="lato in pixel di ogni campione della palette")
    args = ap.parse_args()

    args.colori = max(1, min(64, args.colori))  # stesso limite del sito
    ingresso = Path(args.input)

    if ingresso.is_dir():
        cartella = Path(args.output) if args.output else ingresso / f"ridotte_{args.colori}colori"
        cartella.mkdir(parents=True, exist_ok=True)
        file = sorted(p for p in ingresso.iterdir() if p.suffix.lower() in ESTENSIONI)
        if not file:
            sys.exit("Nessuna immagine trovata nella cartella.")
        for p in file:
            elabora(p, cartella / f"{p.stem}.png", args)
    else:
        uscita = Path(args.output) if args.output else ingresso.with_name(
            f"{ingresso.stem}_{args.colori}colori.png")
        elabora(ingresso, uscita, args)


if __name__ == "__main__":
    main()
