---
title: Shavian Comics
emoji: 💬
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: gpl-3.0
short_description: Transliterate comic speech bubbles into the Shavian alphabet
---

# Shavian Comics

Drop in a comic page: its text is found, read and transliterated into the
[Shavian alphabet](https://en.wikipedia.org/wiki/Shavian_alphabet), and you can
check and correct every step before downloading the result.

Source: https://github.com/mindcat/shavian-comics

- Text detection: [comic-text-detector](https://github.com/dmMaze/comic-text-detector) (GPL-3.0)
- OCR: Groq API when configured, otherwise [Florence-2](https://huggingface.co/microsoft/Florence-2-base) (MIT) on CPU
- Transliteration: dechifro's [shaw.py](https://dechifro.org/shavian/) with spaCy part-of-speech tagging
- Erasing: flat fill, or [LaMa](https://github.com/advimman/lama) inpainting (Apache-2.0) on artwork

Uploaded images are kept in memory only while you work on them and are never stored.
When OCR uses Groq, crops of the text are sent to Groq's API.
