# shavian-comics
Convert comics with speech bubbles (primarily Calvin & Hobbes) from English (Latin orthography) to Shavian (phonemic neography).

## set-up

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync                                  # Python 3.12 + dependencies (CUDA PyTorch on Linux/Windows)
uv run python src/init.py                # download the speech-bubble model, dictionaries and fonts
uv run python src/shavian-comics.py      # convert everything in src/img/in/ into src/img/out/
```

`init.py` grabs the .dict dictionaries from dechifro, the .pt machine vision model by kitsumed responsible for identifying our text bubbles, and comic-esque fonts from Haley Wakamatsu. The model and fonts are checked against pinned SHA-256 checksums. dechifro's incredible Shavian translation engine, `shaw.py`, is vendored in `src/` at a version tested with this pipeline. The first run also downloads the moondream2 OCR model (~4 GB) from Hugging Face.

## credit/dependecies & appreciation

what makes open source software so incredibly powerful is the willingness of programmers, artists, and creators so much more talented than myself to share their work with the world
so that we can build our own novel tools off of theirs.

<details>
  <summary>e.g.</summary>

  _img here of starting c&h comic_

  _img here of conversion_

</details>
