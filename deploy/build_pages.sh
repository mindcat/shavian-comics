#!/bin/bash
# Assemble the static GitHub Pages site (everything runs in the visitor's browser).
#   deploy/build_pages.sh [OUT_DIR] [API_URL]
# Models and dictionaries are downloaded (checksums verified) unless already present
# locally in src/models, src/dict and src/fonts. The comic font (Backlog Shavian, by
# Haley Wakamatsu, SIL Open Font License 1.1) is hosted with the site.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO/deploy/pages-build}"
API="${2:-}"

rm -rf "$OUT"
mkdir -p "$OUT/web/local" "$OUT/assets"
cp "$REPO/src/web/index.html" "$OUT/"
cp "$REPO"/src/web/*.js "$REPO"/src/web/*.css "$OUT/web/"
cp "$REPO"/src/web/local/*.js "$OUT/web/local/"
cp "$REPO/src/shaw.py" "$REPO/src/shavcomic/textproc.py" "$OUT/assets/"

get() {  # local copy, else download; then verify the checksum if given
  local src="$1" url="$2" out="$3" sum="${4:-}"
  if [ -f "$src" ]; then cp "$src" "$out"; else curl -sSfL "$url" -o "$out"; fi
  if [ -n "$sum" ]; then echo "$sum  $out" | sha256sum -c - >/dev/null; fi
}
get "$REPO/src/models/comictextdetector.pt.onnx" \
  https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx \
  "$OUT/assets/comictextdetector.onnx" 1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f
if [ -f "$REPO/src/dict/amer.dict" ]; then cp "$REPO/src/dict/amer.dict" "$OUT/assets/amer.dict"
else curl -sSfL https://dechifro.org/shavian/amer.dict.xz | xz -dc > "$OUT/assets/amer.dict"; fi
TAGGER="$REPO/src/models/nltk_data/taggers/averaged_perceptron_tagger_eng"
if [ ! -d "$TAGGER" ]; then
  TMP="$(mktemp -d)"
  get "" https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip \
    "$TMP/tagger.zip" 6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b
  (cd "$TMP" && unzip -q tagger.zip)
  TAGGER="$TMP/averaged_perceptron_tagger_eng"
fi
get "$REPO/src/fonts/backlogshavian.otf" https://2gd4.me/tidbit/shavian-fonts/backlogshavian.otf \
  "$OUT/assets/backlogshavian.otf" 9575619c156fadd005dc2aac065c3a780738c86563ac5c4d3ccd390b038339ec
cat > "$OUT/assets/FONTS.md" <<'NOTE'
# Fonts

`backlogshavian.otf`: Back Log Shavian, Copyright (c) 2021 Haley Wakamatsu
(https://2gd4.me/tidbit/shavian-fonts/), licensed under the SIL Open Font
License, Version 1.1 (https://openfontlicense.org). Its embedded notice also
credits Back Issues, Copyright (c) 2014 Nate Piekos, Blambot.com.
NOTE
for part in weights tagdict classes; do
  cp "$TAGGER/averaged_perceptron_tagger_eng.$part.json" "$OUT/assets/nltk-tagger-$part.json"
done

{
  echo 'window.SHAVIAN_ASSETS = "assets";'
  echo 'window.SHAVIAN_FONT_URL = "assets/backlogshavian.otf";'
  echo 'window.SHAVIAN_DEFAULT_MODE = "local";'
  if [ -n "$API" ]; then printf 'window.SHAVIAN_API = "%s";\n' "$API"; else echo 'window.SHAVIAN_NO_SERVER = true;'; fi
} > "$OUT/web/config.js"
touch "$OUT/.nojekyll"
echo "Pages site assembled in $OUT:"
(cd "$OUT" && du -ah . | sort -k2 | grep -v '^\S*\s*\.$')
