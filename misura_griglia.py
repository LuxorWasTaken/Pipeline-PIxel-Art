"""
Misura della griglia apparente delle immagini generate (Sezione 3.6.1).

Per ogni immagine 1024x1024 somma l'intensita' delle differenze fra pixel
adiacenti (orizzontali e verticali) in funzione della posizione del bordo
modulo 8 e modulo 16. Senza una griglia preferenziale ciascuna delle otto
posizioni raccoglierebbe il 12,5 % dell'energia.
"""
import collections
from pathlib import Path

import numpy as np
from PIL import Image

RISULTATI = Path(__file__).resolve().parent / "Risultati"


def energia_per_fase(a, m):
    dx = np.abs(np.diff(a, axis=1)).sum(0)
    dy = np.abs(np.diff(a, axis=0)).sum(1)
    e = np.concatenate([dx, dy])
    pos = np.concatenate([np.arange(len(dx)), np.arange(len(dy))])
    s = np.bincount(pos % m, weights=e, minlength=m)
    return s / s.sum()


quote, fasi, rapporti = [], collections.Counter(), []
for f in sorted(RISULTATI.glob("*/1024/*.png")):
    a = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
    s8, s16 = energia_per_fase(a, 8), energia_per_fase(a, 16)
    quote.append(s8.max())
    fasi[int(s8.argmax())] += 1
    rapporti.append(min(s16[7], s16[15]) / max(s16[7], s16[15]))

print(f"immagini {len(quote)}")
print(f"quota della fase dominante (mod 8): mediana {np.median(quote):.3f}, "
      f"10o percentile {np.percentile(quote, 10):.3f} (uniforme 0,125)")
print(f"fase dominante: {fasi.most_common()}  (7 = bordi sui multipli di 8)")
print(f"mod 16, rapporto fra le fasi 7 e 15: mediana {np.median(rapporti):.2f}")
