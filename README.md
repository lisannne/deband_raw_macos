# deband_raw — LED-flicker banding uit RAW-foto's halen

Werkt direct op de RAW (Bayer) data: per sensorrij wordt gemeten hoeveel het
flikkerende licht die rij lichter of donkerder heeft gemaakt (per kleurkanaal, en
per tegel van het beeld: 8 kolomstroken x enkele rij-vensters, zodat verschillende
lampen in verschillende delen van het beeld apart worden gecorrigeerd) en dat wordt
weer weggedeeld. Er wordt niets geblurd, dus
scherpte en ruis blijven zoals ze waren. Het resultaat is een DNG die je als
gewone RAW in Lightroom / Capture One opent.

## Installatie (macOS, eenmalig)

1. Open Terminal en zorg dat python3 werkt (`python3 --version`). Zo niet:
   `xcode-select --install` (of Python van python.org).
2. In de map van dit script:
   ```
   python3 -m pip install --user -r requirements.txt
   ```

## Gebruik

Makkelijkste manier: zet `deband_raw.py`, `requirements.txt` en
`deband_deze_map.command` in de map met je foto's en dubbelklik op
`deband_deze_map.command`. Je kunt kiezen tussen alleen meten (`c`) of direct
corrigeren. Alle RAW's worden parallel verwerkt (standaard 8 tegelijk op een M-serie
Mac); resultaat in submap `deband/` (`*_deband.dng`, lossless gecomprimeerd, ~25-28 MB
per 24-MP foto) plus een samenvatting per foto in het Terminal-venster.

Of via Terminal:
```
python3 deband_raw.py IMG_1660.CR3                # -> IMG_1660_deband.dng naast het origineel
python3 deband_raw.py *.CR3 --check               # alleen meten: welke foto's hebben banding, hoe erg
python3 deband_raw.py *.CR3 --preview --outdir uit  # batch, met voor/na jpg's
python3 deband_raw.py *.CR3 --jobs 4              # minder parallel (minder geheugen/CPU)
python3 deband_raw.py foto.CR3 --no-compress      # ongecomprimeerde DNG (48 MB)
python3 deband_raw.py foto.CR3 --tiff             # ook een 16-bit TIFF (sRGB)
python3 deband_raw.py foto.CR3 --strength 0.8     # iets minder hard corrigeren
python3 deband_raw.py foto.CR3 --period 450       # periode (in sensorrijen) zelf opgeven
python3 deband_raw.py foto.CR3 --strips 4         # minder ruimtelijke variatie (vlakker)
python3 deband_raw.py foto.CR3 --ywin 1           # geen rij-vensters, alleen stroken (v1-gedrag)
python3 deband_raw.py foto.CR3 --strips 12 --ywin 6   # fijnere tegels; helpt soms een fractie
```

## Wat je in de terminal ziet

- `periode: 225 sensorrijen (bandsignaal 80% van het profiel)` — de afstand tussen
  twee banden, en hoe duidelijk dat patroon is. Binnen één shoot/zaal is de periode
  normaal voor alle foto's gelijk. Als de detectie bij een foto afwijkt, geef de periode
  dan met `--period` op.
- `banding: ~8.4% -> ~0.3%` — gemeten band-sterkte vóór en ná correctie (piek-dal, groen
  kanaal). Onder ~1% is het praktisch weg. Daarachter de gemeten sterkte per kolomstrook.
- Zelfcontrole: als er nauwelijks periodieke banding is, of als de correctie te weinig
  helpt, wordt de foto **overgeslagen** (er wordt dan niets geschreven). Met `--force`
  schrijft hij toch.

## Periodedetectie

De bandperiode wordt met drie onafhankelijke detectoren gezocht:

1. een periodogram van het gemeenschappelijke rijprofiel (goed voor zachte, golvende
   banding over de hele breedte);
2. een autocorrelatie van een scherp hoogdoorlaat-profiel (goed voor smalle, pulsachtige
   banden zoals bij sterk gedimde LED-wash);
3. een periodogram per tegel (8 kolomstroken x rij-vensters), voor banding die maar in
   een deel van het beeld zit: alleen in de rook/het doek bovenin, of alleen aan één
   kant omdat daar een andere lampgroep hangt. Zulke banding geeft in naburige tegels
   een piek op dezelfde periode én dezelfde fase (zelfde rijen); content niet.

Rijen waar de kolomstroken het sterk oneens zijn (mensen, speakers, podiumrand) tellen
in de detectie minder mee, zodat een drukke onderste beeldhelft de meting van de banding
in de bovenste helft niet verstoort.

Als de detectoren het oneens zijn (vaak: 225 of 450 rijen, want afwisselend diepe en ondiepe
banden geven beide), worden alle kandidaten doorgerekend en wint degene die de zelfcontrole
haalt en het minste banding overlaat, gemeten over alle kandidaat-periodes samen. Elke kandidaat wordt getoetst op
echte periodiciteit (de autocorrelatie moet zich op P én 2P herhalen); een onbevestigde
kandidaat wordt alleen geaccepteerd als de correctie overtuigend werkt (>= 2% banding,
>= 75% reductie). Zit de banding maar in een deel van het beeld, dan wordt de
zelfcontrole in dát deel gedaan (>= 2% banding in minstens 3 tegels met gelijke fase,
>= 50% reductie); in de terminal staat dan "gemeten in het deel van het beeld met banding".

## Shoot-periodes (tweede ronde)

Bij een map met 5+ foto's leert het script de lampfrequenties van de shoot uit de foto's
die met zekerheid gecorrigeerd zijn, en probeert daarmee de twijfelgevallen nog een keer.
Lampfrequenties veranderen niet binnen een shoot, dus een foto met een zwak globaal
profiel wordt zo alsnog gevonden. De shoot-periode wordt per foto fijn afgesteld en op
die foto gecontroleerd (herhaalt het patroon zich daar echt?). Zo ja: gewone eisen. Zo
niet: >= 2% banding en >= 60% reductie (een schone foto haalt dat niet; op schone foto's
helpt een correctie met de shoot-periode namelijk helemaal niets).
Draai een shoot daarom als één map, niet per foto.

## Meerdere lampfrequenties

Standaard doet het script één pass. Met `--passes 2` kijkt het na de eerste correctie of
er nog een duidelijk tweede periodiek patroon over is (een lamp met een andere
PWM-frequentie) en corrigeert dat dan ook, mits het aantoonbaar helpt. Gebruik dit voor
losse foto's waar je na de eerste pass nog banden ziet, niet standaard voor een hele map.

## Statussen in de samenvatting

- `gecorrigeerd` — banding gevonden, correctie haalt de zelfcontrole, DNG geschreven.
- `geen banding` / `weinig banding` / `geen banding (zwak signaal)` — overgeslagen, niets
  geschreven.
- `onzeker (waarschijnlijk content)` — er is wel een periodiek-achtig patroon, maar het
  herhaalt zich niet zuiver en de correctie helpt niet: vrijwel altijd content (tribune,
  doek, lampenrij). Overgeslagen.
- `model past niet` — duidelijk, zich herhalend bandsignaal, maar de correctie haalt er te
  weinig af. Dit zijn de foto's om even zelf te bekijken; vaak helpt `--period <rijen>`
  (de shoot-periode), `--strips 4` of `--force`.

In een enkel geval kan het script op een schone foto toch een minieme correctie (<1%)
doen; dat is onzichtbaar en kan geen kwaad.

## Beperkingen

- Werkt goed als de banding netjes over de hele breedte loopt (één of enkele
  flikkerende lichtbronnen). Als verschillende lampen met verschillende frequenties
  verschillende delen van het beeld belichten, blijft er wat over.
- De banden hebben geen "scherpte", dus de correctie ook niet: er gaat geen detail verloren.
- Pixels die in het origineel al uitgebeten waren (lampen, spots) blijven uitgebeten, zodat
  Lightroom ze als zodanig blijft herkennen. Pixels vlak onder wit in een donkere band kunnen
  door de correctie net boven wit komen en clippen dan (zonder flicker waren ze ook geclipt);
  dat gaat om ~0,1% van de pixels, allemaal in lichtbronnen.
- Alleen de zichtbare sensor-area wordt weggeschreven; de DNG krijgt camera-model,
  belichting, lens en oriëntatie uit de CR3 mee, zodat Lightroom hem als Canon-RAW ziet.
- De DNG is lossless gecomprimeerd (standaard DNG lossless-JPEG, zelfde methode als Adobe
  DNG Converter). De RAW-data is bit-voor-bit gelijk aan de ongecomprimeerde versie.
- Snelheid: ongeveer 10-12 s per foto per processorkern; met 8 parallel dus ruwweg 1,5 s
  per foto effectief.
