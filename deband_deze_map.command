#!/bin/bash
# Dubbelklik: corrigeert alle RAW-foto's in de map waar dit bestand staat.
# Resultaat komt in de submap "deband" (lossless gecomprimeerde DNG's + samenvatting).
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 niet gevonden. Installeer eerst de Xcode command line tools:"
  echo "    xcode-select --install"
  echo "of Python via https://www.python.org/downloads/macos/"
  read -p "Druk op Enter om te sluiten"; exit 1
fi
python3 -c "import rawpy, scipy, tifffile, PIL, exifread" 2>/dev/null || {
  echo "Benodigde Python-pakketten installeren (eenmalig)..."
  python3 -m pip install --user -r requirements.txt || python3 -m pip install --user --break-system-packages -r requirements.txt
}
shopt -s nullglob nocaseglob
files=( *.cr3 *.cr2 *.arw *.nef *.raf *.dng *.orf *.rw2 )
if [ ${#files[@]} -eq 0 ]; then
  echo "Geen RAW-bestanden gevonden in: $(pwd)"; read -p "Druk op Enter om te sluiten"; exit 1
fi
echo "${#files[@]} RAW-bestand(en) gevonden. Eerst meten (--check) of direct corrigeren?"
read -p "[Enter] = corrigeren, c = alleen meten: " keuze
if [ "$keuze" = "c" ] || [ "$keuze" = "C" ]; then
  python3 deband_raw.py "${files[@]}" --check
else
  python3 deband_raw.py "${files[@]}" --outdir deband
  echo
  echo "Klaar. Resultaat staat in: $(pwd)/deband"
fi
read -p "Druk op Enter om te sluiten"
