# Pipeline multimodale per la generazione sintetica e la validazione di immagini in pixel art

Codice della tesi di laurea triennale *Pipeline Multimodale per la Generazione Sintetica
e Validazione di Immagini in Pixel Art* Leonardo Quartucci (relatore Prof. Danilo Croce).

## Contenuto

| File | Ruolo | Capitolo |
|---|---|---|
| `pipeline.py` | Pipeline completa: front-end semantico (Gemma 4 via Ollama), generazione (ComfyUI), discretizzazione 1024→128, validazione dello sfondo, riduzione della palette, torneo con oracolo | 3 |
| `riduci_palette.py` | Riduzione della palette (median cut + k-means con pesi percettivi) | 3 |
| `desiderata.json` | Le dieci desiderata usate nella campagna sperimentale | 3 |
| `crea_sondaggi_v2.py` | Crea i sondaggi di valutazione umana degli sprite | 4 |
| `esperimento_strategie.py` | Confronto delle quattro strategie di ordinamento con le scelte umane (NDCG, τ_b, S@3, accordo) | 4 |
| `analisi_capitolo4.py` | Report e figure del capitolo 4 | 4 |
| `video_pipeline.py` | Generazione delle animazioni con Wan 2.2 e post-elaborazione (versioni A e B) | 5 |
| `metriche_video.py` | Metriche automatiche sulle animazioni | 5 |
| `crea_sondaggi_video.py` | Crea i sondaggi di valutazione delle animazioni | 5 |
| `analisi_video.py` | Analisi statistica dei voti sulle animazioni | 5 |

## Requisiti

- Python 3.12 e le librerie in `requirements.txt` (`pip install -r requirements.txt`)
- [Ollama](https://ollama.com) su `localhost:11434` con il modello `gemma4:31b-cloud`
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) su `127.0.0.1:8000` con:
  - immagini: checkpoint waiIllustriousSDXL v17.0 e LoRA ElinSpriteNoob;
  - video: Wan 2.2 I2V A14B (high/low noise, fp8), encoder umt5_xxl, VAE Wan 2.1,
    LoRA lightx2v e i nodi KJNodes (NAG, SageAttention).
- Per i sondaggi: l'applicazione web di raccolta voti, nella cartella indicata dalla
  variabile d'ambiente `SONDAGGI_TESI` (per impostazione predefinita `../Sondaggi tesi`).

Gli indirizzi dei servizi e i parametri si trovano nella sezione *CONFIGURAZIONE* in
testa a ciascuno script.

## Esecuzione

```bash
python pipeline.py                  # capitolo 3: tutte le desiderata di desiderata.json
python crea_sondaggi_v2.py          # capitolo 4: sondaggi sugli sprite
python esperimento_strategie.py     # capitolo 4: esperimento sulle strategie
python analisi_capitolo4.py
python video_pipeline.py            # capitolo 5: animazioni
python metriche_video.py
python crea_sondaggi_video.py
python analisi_video.py
```

L'esecuzione è riprendibile: rilanciando un comando le fasi già completate vengono saltate.
I risultati vengono scritti in `Risultati/`, `Risultati_video/` ed `Esperimento_strategie/`,
che non fanno parte del repository.

## Dati dei partecipanti

I voti raccolti e i nomi dei valutatori non sono pubblicati. Se lo stesso valutatore ha
firmato i sondaggi in modi diversi, le firme si possono unificare in un file
`alias_valutatori.json` (escluso dal repository) nella forma `{"firma": "nome_unificato"}`.
