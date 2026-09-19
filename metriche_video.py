"""
Misure di coerenza temporale delle animazioni (Capitolo 5, Sezione 5.7).

Per ogni animazione (soggetto x livello) e per ciascuna versione
  G  fotogrammi grezzi discretizzati (128x128, nessuna riduzione della palette)
  A  palette indipendente per fotogramma
  B  palette vincolata a quella dello sprite
calcola, rispetto allo sprite di partenza S e fra fotogrammi consecutivi:

  colori          numero di colori distinti per fotogramma
  deriva          distanza percettiva media (pesata per pixel) fra i colori del
                  fotogramma e il colore piu' vicino della palette dello sprite
  griglia         (solo fotogrammi a 512 px) quota di blocchi 4x4 uniformi
  sfarf_sfondo    quota dei pixel di sfondo (in entrambi i fotogrammi) che cambiano colore
  sfarf_sogg      quota dei pixel del soggetto (in entrambi i fotogrammi) che cambiano colore
  iou             intersezione su unione fra la maschera del soggetto e quella dello sprite
  chiusura        quota di pixel diversi fra l'ultimo e il primo fotogramma del loop

Uscita: Risultati_video/metriche_video.csv (una riga per animazione e versione) e
        Risultati_video/metriche_per_fotogramma.csv (serie temporali)
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import pipeline as pl
from riduci_palette import distanze

sys.stdout.reconfigure(encoding="utf-8")
PIPELINE = Path(__file__).resolve().parent
VIDEO = PIPELINE / "Risultati_video"
RIS = PIPELINE / "Risultati"


def carica(cartella):
    return [np.asarray(Image.open(f).convert("RGB")) for f in sorted(cartella.glob("*.png"))]


def maschera_soggetto(a):
    return ~pl.valida_sfondo(a)["maschera"]


def deriva(frame, palette):
    colori, conteggi = np.unique(frame.reshape(-1, 3), axis=0, return_counts=True)
    rgba = np.column_stack([colori.astype(np.int64), np.full(len(colori), 255)])
    pal = [list(c) + [255] for c in palette]
    d = np.sqrt(distanze(rgba, pal).min(axis=1))
    return float((d * conteggi).sum() / conteggi.sum())


def blocchi_uniformi(img, s=4):
    h = img.shape[0] // s
    b = img[:h * s, :h * s].reshape(h, s, h, s, 3).transpose(0, 2, 1, 3, 4).reshape(h, h, s * s, 3)
    return float(np.all(b == b[:, :, :1, :], axis=(2, 3)).mean())


def main():
    import json
    sprite = json.loads((VIDEO / "sprite_scelti.json").read_text(encoding="utf-8"))
    righe, serie = [], []
    for s in sorted(p.name for p in VIDEO.iterdir() if p.is_dir()):
        S = np.asarray(Image.open(RIS / s / "16colori" / sprite[s]["file"]).convert("RGB"))
        pal_S = np.unique(S.reshape(-1, 3), axis=0).tolist()
        m_S = maschera_soggetto(S)
        for liv in ("L1", "L2", "L3"):
            c = VIDEO / s / liv
            if not (c / "B_128").exists():
                continue
            grezzi = carica(c / "grezzi_512")[:-1]
            griglia = [blocchi_uniformi(g) for g in grezzi]
            for ver, cart in (("G", "logici_128"), ("A", "A_128"), ("B", "B_128")):
                F = carica(c / cart)
                maschere = [maschera_soggetto(f) for f in F]
                col = [pl.conta_colori(f) for f in F]
                der = [deriva(f, pal_S) for f in F]
                iou = [float((m & m_S).sum() / max((m | m_S).sum(), 1)) for m in maschere]
                sf_bg, sf_sg = [np.nan], [np.nan]
                for t in range(1, len(F)):
                    cambia = np.any(F[t] != F[t - 1], axis=2)
                    bg = ~maschere[t] & ~maschere[t - 1]
                    sg = maschere[t] & maschere[t - 1]
                    sf_bg.append(float(cambia[bg].mean()) if bg.any() else np.nan)
                    sf_sg.append(float(cambia[sg].mean()) if sg.any() else np.nan)
                chiusura = float(np.any(F[-1] != F[0], axis=2).mean())
                righe.append({"soggetto": s, "livello": liv, "versione": ver, "fotogrammi": len(F),
                              "colori_medi": round(float(np.mean(col)), 1), "colori_max": int(max(col)),
                              "deriva": round(float(np.mean(der)), 3),
                              "griglia_512": round(float(np.mean(griglia)), 4) if ver == "G" else "",
                              "sfarf_sfondo": round(float(np.nanmean(sf_bg)), 4),
                              "sfarf_sogg": round(float(np.nanmean(sf_sg)), 4),
                              "iou_media": round(float(np.mean(iou)), 4), "iou_min": round(float(np.min(iou)), 4),
                              "chiusura": round(chiusura, 4)})
                for t in range(len(F)):
                    serie.append({"soggetto": s, "livello": liv, "versione": ver, "t": t, "colori": col[t],
                                  "deriva": round(der[t], 3), "iou": round(iou[t], 4),
                                  "sfarf_sfondo": "" if np.isnan(sf_bg[t]) else round(sf_bg[t], 4),
                                  "sfarf_sogg": "" if np.isnan(sf_sg[t]) else round(sf_sg[t], 4),
                                  "griglia_512": round(griglia[t], 4) if ver == "G" else ""})
            print(f"{s} {liv}: " + "  ".join(
                f"{r['versione']}: sfondo {r['sfarf_sfondo']:.3f} sogg {r['sfarf_sogg']:.3f} iou {r['iou_media']:.3f}"
                for r in righe[-3:]))
    for nome, dati in (("metriche_video.csv", righe), ("metriche_per_fotogramma.csv", serie)):
        with open(VIDEO / nome, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dati[0]), delimiter=";")
            w.writeheader()
            w.writerows(dati)
    print("\nMedie per livello e versione:")
    for liv in ("L1", "L2", "L3"):
        for ver in ("G", "A", "B"):
            rr = [r for r in righe if r["livello"] == liv and r["versione"] == ver]
            if rr:
                print(f"  {liv} {ver}: colori {np.mean([r['colori_medi'] for r in rr]):7.1f}  deriva {np.mean([r['deriva'] for r in rr]):5.2f}"
                      f"  sfarf.sfondo {np.mean([r['sfarf_sfondo'] for r in rr]):.3f}  sfarf.sogg {np.mean([r['sfarf_sogg'] for r in rr]):.3f}"
                      f"  iou {np.mean([r['iou_media'] for r in rr]):.3f} (min {np.min([r['iou_min'] for r in rr]):.3f})"
                      + (f"  griglia {np.mean([r['griglia_512'] for r in rr]):.3f}" if ver == "G" else ""))


if __name__ == "__main__":
    main()
