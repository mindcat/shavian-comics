# shavian-comics
Convert comics with speech bubbles (primarily Calvin & Hobbes) from English (Latin orthography) to Shavian (phonemic neography).

## set-up

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync                                  # Python 3.12 + dependencies (CUDA PyTorch on Linux/Windows)
uv run python src/init.py                # download the speech-bubble model, dictionaries and fonts
uv run python src/shavian-comics.py      # convert everything in src/img/in/ into src/img/out/
uv run python src/compare.py             # run every backend side by side -> src/img/out/compare/index.html
uv run python src/server.py              # web tool at http://127.0.0.1:8765 (add --host to share it)
```

The web tool lets you drop in a comic, then check and correct every stage: switch regions on or off (sound effects and logos are found but off by default), fix the English (re-transliterated as you go) or the Shavian directly, draw a box around text that was missed, pick the eraser and font size per region, and download the result. The page in `src/web/` only talks to the server's `/api/`, so it can be hosted separately and pointed at a server with `?api=https://your-server`.

Each stage has interchangeable backends (`uv run python src/shavian-comics.py --help`):

| stage | options |
| --- | --- |
| `--detector` | `ctd` (default): [comic-text-detector](https://github.com/dmMaze/comic-text-detector) text lines grouped by bubble (also finds text without a bubble) · `yolo`: YOLO speech-bubble masks matched to moondream's bubble boxes |
| `--ocr` | `qwen` (default, Qwen3-VL-2B-Instruct) · `moondream` (moondream2) |
| `--eraser` | `auto` (default: fill in bubbles, LaMa elsewhere) · `fill` (flat bubble colour) · `telea` (OpenCV inpainting) · `lama` ([LaMa](https://github.com/advimman/lama) inpainting) |
| `--size` | `match` (match the original lettering height) · `fit` (largest size that fits) |

Transliteration runs dechifro's `shaw.py`, kept loaded in-process, with spaCy part-of-speech tagging (`--no-pos` turns it off). Everything runs offline once the models are downloaded; set `HF_HUB_OFFLINE=1` to make sure.

`init.py` grabs the .dict dictionaries from dechifro, the .pt machine vision model by kitsumed responsible for identifying our text bubbles, and comic-esque fonts from Haley Wakamatsu, plus the comic-text-detector and LaMa weights. The models and fonts are checked against pinned SHA-256 checksums. dechifro's incredible Shavian translation engine, `shaw.py`, is vendored in `src/` at a version tested with this pipeline. The first run also downloads the moondream2 OCR model (~4 GB) from Hugging Face.

## credit/dependecies & appreciation

what makes open source software so incredibly powerful is the willingness of programmers, artists, and creators so much more talented than myself to share their work with the world
so that we can build our own novel tools off of theirs.

<details>
  <summary>e.g.</summary>

  _img here of starting c&h comic_

  _img here of conversion_

</details>
