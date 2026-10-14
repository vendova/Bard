#!/usr/bin/env bash
# Fetch candidate Google Fonts used by the lyric template (Stage 1).
# All are SIL Open Font License; see assets/fonts/OFL.txt after download.
set -u
cd "$(dirname "$0")/.."
mkdir -p assets/fonts
BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"
fetch() {  # name url
  local name="$1" url="$2"
  local code size
  code=$(curl -sL -o "assets/fonts/$name" -w '%{http_code}' "$url")
  size=$(stat -c%s "assets/fonts/$name" 2>/dev/null || echo 0)
  printf '%s %8s %s\n' "$code" "$size" "$name"
}
fetch MrsSaintDelafield-Regular.ttf "$BASE/mrssaintdelafield/MrsSaintDelafield-Regular.ttf"
fetch HerrVonMuellerhoff-Regular.ttf "$BASE/herrvonmuellerhoff/HerrVonMuellerhoff-Regular.ttf"
fetch MonsieurLaDoulaise-Regular.ttf "$BASE/monsieurladoulaise/MonsieurLaDoulaise-Regular.ttf"
fetch Allison-Regular.ttf          "$BASE/allison/Allison-Regular.ttf"
fetch Anton-Regular.ttf            "$BASE/anton/Anton-Regular.ttf"
fetch BebasNeue-Regular.ttf        "$BASE/bebasneue/BebasNeue-Regular.ttf"
fetch LeagueGothic-Regular.ttf     "$BASE/leaguegothic/LeagueGothic%5Bwdth%2Cwght%5D.ttf"
fetch Oswald.ttf                   "$BASE/oswald/Oswald%5Bwght%5D.ttf"
fetch Montserrat.ttf               "$BASE/montserrat/Montserrat%5Bwght%5D.ttf"
fetch Inter.ttf                    "$BASE/inter/Inter%5Bopsz%2Cwght%5D.ttf"
fetch OFL.txt                      "$BASE/mrssaintdelafield/OFL.txt"
