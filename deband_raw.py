#!/usr/bin/env python3
"""
deband_raw.py — verwijdert LED-flicker banding (rolling-shutter strepen) uit RAW foto's.

Werkt direct op de Bayer-data: per sensorrij wordt de 'gain' van het flikkerende licht
geschat (periodiek model, per kleurkanaal, per kolomstrook) en weer weggedeeld.
Output: DNG (blijft RAW, open je gewoon in Lightroom/Capture One) en optioneel 16-bit TIFF.

Gebruik:
    python3 deband_raw.py IMG_1660.CR3                 # -> IMG_1660_deband.dng (lossless gecomprimeerd)
    python3 deband_raw.py *.CR3                        # batch, parallel, met samenvatting
    python3 deband_raw.py *.CR3 --check                # alleen meten: welke foto's hebben banding?
    python3 deband_raw.py *.CR3 --preview              # ook voor/na jpg's
    python3 deband_raw.py foto.CR3 --tiff              # ook een 16-bit sRGB TIFF
    python3 deband_raw.py foto.CR3 --strength 0.8      # minder hard corrigeren

Opties:
    --period P       bandperiode in sensorrijen (standaard: automatisch)
    --harmonics K    max. aantal harmonischen in het model (standaard 16)
    --strips N       aantal kolomstroken voor ruimtelijke variatie (standaard 8)
    --ywin N         aantal rij-vensters (tegels) daarbinnen; 0 = automatisch, 1 = uit
    --strength S     0..1.5, hoeveel van de gemeten banding wordt verwijderd (1.0)
    --passes N       1 (standaard) of 2: pass 2 probeert een tweede lampfrequentie
    --iterations N   schattings-iteraties (2)
    --no-dng, --tiff, --preview, --outdir DIR, --force, --verbose
"""
import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import rawpy
from scipy.ndimage import gaussian_filter1d, gaussian_filter, uniform_filter1d


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def robust_fit(X, y, w=None, iters=5):
    """Iteratively reweighted least squares met Tukey biweight."""
    if w is None:
        w = np.ones(len(y))
    beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = r / (4.685 * s)
        tw = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        ww = w * tw
        beta = np.linalg.lstsq(X * ww[:, None], y * ww, rcond=None)[0]
    return beta


def fourier_basis(y, period, K):
    """Basis: [cos(2πk y/P), sin(2πk y/P)] voor k=1..K  -> (len(y), 2K)."""
    t = 2 * np.pi * np.asarray(y, dtype=np.float64)[:, None] / period
    k = np.arange(1, K + 1)[None, :]
    return np.concatenate([np.cos(t * k), np.sin(t * k)], axis=1)


def refine_period(c, P0, span=0.05, K=8, r=None):
    """Fijnafstelling van periode P0 (half-res rijen) op profiel c: beste harmonische fit binnen ±span."""
    H = len(c)
    y = np.arange(H)
    w = np.hanning(H) if r is None else np.hanning(H) * r
    cw = c * w
    K = int(min(K, max(2, P0 // 10)))
    best_P, best_E = P0, -1.0
    for P in np.arange(P0 * (1 - span), P0 * (1 + span), 0.05):
        X = fourier_basis(y, P, K) * w[:, None]
        beta = np.linalg.lstsq(X, cw, rcond=None)[0]
        E = np.sum((X @ beta) ** 2)
        if E > best_E:
            best_P, best_E = P, E
    return best_P


def period_candidates(plane):
    """Kandidaat-periodes (half-res rijen) uit twee onafhankelijke detectoren:
    - periodogram (één sinus): goed voor zachte, sinusvormige banding;
    - autocorrelatie van een scherp hoogdoorlaat-profiel: goed voor smalle, pulsachtige banden
      (daar zit de energie in veel harmonischen, en ziet het periodogram de grondtoon niet).
    Geeft [(P_half, score), ...] zonder dubbelen (binnen 5%), plus het profiel van de zelfcontrole."""
    profs32, profs32_30 = strip_profiles(plane, 32, (50, 30))
    c50, r50 = combine_profiles(profs32)
    P_sin, pw_sin, _ = estimate_period(plane, prof=(c50, r50, profs32))
    cands = [(P_sin, pw_sin)]
    prof = (c50, r50)
    profs8 = np.array([np.nanmedian(profs32[4 * s:4 * s + 4], axis=0) for s in range(8)])
    tiles = tile_candidates(profs8)
    c, r30 = combine_profiles(profs32_30)
    c30_raw = c
    H = len(c)
    c = c * np.sqrt(r30)                                      # onbetrouwbare rijen dempen
    ac = np.correlate(c, c, 'full')[H - 1:]
    ac = ac / max(ac[0], 1e-12) * H / np.arange(H, 0, -1)       # normaliseer voor overlaplengte
    from scipy.signal import find_peaks
    lo, hi = 20, int(H / 4.0)
    pk, _ = find_peaks(ac[lo:hi], distance=10)
    pk = pk + lo

    def ac_at(L):
        a, b = int(L * 0.95), int(L * 1.05) + 1
        return float(ac[a:b].max()) if b < len(ac) and a < b else float('nan')

    def confirmed(P):
        """Echte periodieke banding herhaalt zich: de autocorrelatie moet op P én op 2P een piek hebben."""
        a1, a2 = ac_at(P), ac_at(2 * P)
        if np.isnan(a2):
            return a1 >= 0.3
        return a1 >= 0.2 and a2 >= 0.6 * a1

    if len(pk):
        best = int(pk[np.argmax(ac[pk])])
        # een kortere piek die bijna even hoog is, is de echte (zuivere) periode
        for div in (3, 2):
            sub = [i for i in pk if abs(i - best / div) <= 0.05 * best / div]
            if sub and ac[sub[0]] >= 0.9 * ac[best]:
                best = int(sub[0])
                break
        if ac[best] >= 0.2 and all(abs(best - P) > 0.05 * P for P, _ in cands):
            # fijnafstelling zoals bij het periodogram
            P0 = float(best)
            y = np.arange(H)
            w = np.hanning(H)
            cw = c * w
            K = int(min(8, max(2, P0 // 10)))
            best_P, best_E = P0, -1.0
            for P in np.arange(P0 * 0.96, P0 * 1.04, 0.05):
                X = fourier_basis(y, P, K) * w[:, None]
                beta = np.linalg.lstsq(X, cw, rcond=None)[0]
                E = np.sum((X @ beta) ** 2)
                if E > best_E:
                    best_P, best_E = P, E
            cands.append((best_P, float(ac[best])))
    # Als de autocorrelatie op 2P (bijna) even sterk is als op P, kan het patroon een grondperiode 2P
    # met sterke 2e harmonische zijn (afwisselend diepe/ondiepe banden). Het model met periode 2P
    # bevat ook alle harmonischen van P, dus 2P kiezen is veilig en vangt beide gevallen.
    Pmax = H / 4.0
    out = []
    for P, sc in cands:
        opts = [(P, sc, confirmed(P))]
        if 2 * P < Pmax and ac_at(P) >= 0.2 and ac_at(2 * P) >= 0.8 * ac_at(P):
            opts.append((2 * P, sc, True))          # beide doorrekenen; de beste wint
        for Q, sq, cq in opts:
            if all(abs(Q - R) > 0.05 * R for R, _, _ in out):
                out.append((Q, sq, cq))
    # tegel-kandidaten: bestaande kandidaat binnen 5% -> die overnemen (hoogste score, bevestigd als
    # de tegels dat zeggen); anders nieuwe kandidaat
    for Pt, st, ct in tiles:
        merged = False
        for k, (Q, sq, cq) in enumerate(out):
            if abs(Pt - Q) <= 0.05 * Q:
                out[k] = (Q, max(sq, st), cq or ct)
                merged = True
                break
        if not merged:
            out.append((Pt, st, ct))
    return out, dict(prof=prof, profs8=profs8, confirm=confirmed, c30=c30_raw, r30=r30)


def strip_profiles(plane, nstrips=32, sigma=50):
    """Per kolomstrook het hoogdoorlaat log-rijprofiel (mediaan over de kolommen van de strook).
    NaN waar een rij in die strook te weinig bruikbare pixels heeft (donker, geclipt, randen).
    sigma mag een tuple zijn: dan komt er per sigma een profielenset terug (filters gedeeld)."""
    sigmas = sigma if isinstance(sigma, (tuple, list)) else (sigma,)
    plane = plane[:, ::2]                                 # om de andere kolom: zelfde mediaan, half werk
    H, W = plane.shape
    logp = np.log(np.maximum(plane, 4.0))
    dy = np.abs(np.diff(gaussian_filter(logp, 2.0), axis=0, prepend=logp[:1]))
    valid = (plane > 20) & (plane < 0.9 * plane.max()) & (dy < 0.03)
    edges = [(s * W // nstrips, (s + 1) * W // nstrips) for s in range(nstrips)]
    nmin = [max(10, 0.1 * (x1 - x0)) for x0, x1 in edges]
    out = []
    for sg in sigmas:
        trend = gaussian_filter1d(logp, sigma=sg, axis=0, mode='nearest')
        hp = np.where(valid, logp - trend, np.nan)
        profs = []
        for s, (x0, x1) in enumerate(edges):
            blk = hp[:, x0:x1]
            n = np.sum(~np.isnan(blk), axis=1)
            med = np.nanmedian(blk, axis=1)
            # te weinig bruikbare pixels in deze rij van deze strook -> geen uitspraak (NaN), geen ruis
            med[n < nmin[s]] = np.nan
            profs.append(med)
        out.append(np.array(profs))
    return out if isinstance(sigma, (tuple, list)) else out[0]


def combine_profiles(profs):
    """Gemeenschappelijk profiel c en betrouwbaarheid per rij r uit de strookprofielen."""
    nstrips = profs.shape[0]
    c = np.nanmedian(profs, axis=0)
    nvalid = np.sum(~np.isnan(profs), axis=0)
    # betrouwbaarheid per rij: banding is in alle stroken gelijk, content niet. Rijen waar de
    # stroken het sterk oneens zijn (grote spreiding) of waar weinig stroken bruikbaar zijn,
    # tellen minder mee in de periodedetectie en de zelfcontrole.
    spread = np.nanmedian(np.abs(profs - c[None, :]), axis=0)
    spread = np.where(np.isnan(spread), np.inf, spread)
    s0 = (np.median(spread[np.isfinite(spread)]) + 1e-6) if np.isfinite(spread).any() else 1.0
    r = 1.0 / (1.0 + (spread / s0) ** 2) * np.minimum(1.0, nvalid / (0.5 * nstrips))
    r = np.where(np.isnan(c), 0.0, r)
    c = np.where(np.isnan(c), 0.0, c)             # rijen zonder enige bruikbare strook: neutraal
    mad = 1.4826 * np.median(np.abs(c - np.median(c))) + 1e-6
    c = np.clip(c, -3 * mad, 3 * mad)            # restjes content begrenzen
    return c - c.mean(), r


def common_profile(plane, nstrips=32, sigma=50, with_weight=False, with_strips=False):
    """Het 'gemeenschappelijke' rijprofiel (log-domein, hoogdoorlaat) van een half-res plane.

    Per smalle kolomstrook wordt de mediaan over de kolommen genomen, en daarna de mediaan
    over de stroken: banding zit in elke strook op dezelfde rijen, content niet. Daardoor
    valt content (mensen, doeken, lampen) er grotendeels uit.
    Met with_weight: ook een betrouwbaarheid per rij (0..1); met with_strips: ook de strookprofielen.
    """
    profs = strip_profiles(plane, nstrips, sigma)
    c, r = combine_profiles(profs)
    out = (c,)
    if with_weight:
        out += (r,)
    if with_strips:
        out += (profs,)
    return out if len(out) > 1 else c


def sine_power_vec(c, Ps, w):
    """Als sine_power, maar voor een hele reeks periodes tegelijk (gesloten 2x2-oplossing, in blokken)."""
    L = len(c)
    y = np.arange(L)
    cw = c * w
    sy = float(np.sum(cw ** 2))
    Ps = np.asarray(Ps, dtype=np.float64)
    out = np.zeros(len(Ps))
    if sy <= 1e-12:
        return out
    w = w.astype(np.float32)
    cw = cw.astype(np.float32)
    y = y.astype(np.float32)
    for i0 in range(0, len(Ps), 256):
        t = (2 * np.pi / Ps[i0:i0 + 256, None]).astype(np.float32) * y[None, :]
        a = np.cos(t) * w
        b = np.sin(t) * w
        aa = np.sum(a * a, axis=1)
        bb = np.sum(b * b, axis=1)
        ab = np.sum(a * b, axis=1)
        ay = a @ cw
        by = b @ cw
        det = np.maximum(aa * bb - ab ** 2, 1e-12)
        b1 = (bb * ay - ab * by) / det
        b2 = (aa * by - ab * ay) / det
        out[i0:i0 + 256] = (b1 * ay + b2 * by) / sy
    return np.clip(out, 0.0, 1.0)


def tile_windows(H):
    """Rij-vensters voor de tegeldetectie: volle hoogte, plus vensters van H/2 en H/4 (50% overlap)."""
    wins = [(0, H)]
    for n in (2, 4):
        L = H // n
        for i in range(2 * n - 1):
            y0 = i * L // 2
            wins.append((y0, y0 + L))
    return wins


def tile_profile(p):
    """Maak een tegelprofiel (met NaN) klaar voor analyse: (q, ok) of None bij te weinig data."""
    ok = ~np.isnan(p)
    if ok.sum() < 0.5 * len(p):
        return None
    q = np.where(ok, p, 0.0)
    q = q - q[ok].mean()
    mad = 1.4826 * np.median(np.abs(q[ok])) + 1e-6
    q = np.clip(q, -3 * mad, 3 * mad)
    return q, ok.astype(np.float64)


def tile_periodic(q, ok, P):
    """Herhaalt het patroon in deze tegel zich echt (autocorrelatie piekt op P én 2P)?"""
    L = len(q)
    x = q * ok
    ac = np.correlate(x, x, 'full')[L - 1:]
    ac = ac / max(ac[0], 1e-12) * L / np.arange(L, 0, -1)

    def at(Lag):
        a, b = int(Lag * 0.95), int(Lag * 1.05) + 1
        return float(ac[a:b].max()) if b < 0.75 * L and a < b else float('nan')

    a1, a2 = at(P), at(2 * P)
    if np.isnan(a1):
        return False
    if np.isnan(a2):
        return a1 >= 0.3
    return a1 >= 0.2 and a2 >= 0.6 * a1


def tile_phase(q, ok, P):
    """Fase (rad) van de sinus met periode P in tegelprofiel q."""
    L = len(q)
    w = np.hanning(L) * ok
    t = 2 * np.pi * np.arange(L) / P
    X = np.stack([np.cos(t) * w, np.sin(t) * w], axis=1)
    beta = np.linalg.lstsq(X, q * w, rcond=None)[0]
    return float(np.arctan2(beta[1], beta[0]))


def coherent_tiles(entries, tol=np.radians(30)):
    """entries: (venster, strook, fase, payload). Echte banding zit in naburige stroken op dezelfde
    rijen, dus met dezelfde fase; content niet. Per venster: grootste groep stroken met gelijke fase.
    Geeft de payloads van tegels die in een groep van >= 2 stroken zitten."""
    by_win = {}
    for wi, st, ph, payload in entries:
        by_win.setdefault(wi, []).append((st, ph, payload))
    out = []
    for lst in by_win.values():
        best = []
        for _, ph0, _ in lst:
            grp = [x for x in lst if abs(np.angle(np.exp(1j * (x[1] - ph0)))) <= tol]
            if len({x[0] for x in grp}) > len({x[0] for x in best}):
                best = grp
        if len({x[0] for x in best}) >= 2:
            out.extend(x[2] for x in best)
    return out


def tile_candidates(profs8, min_power=0.35):
    """Derde detector: periodogram per tegel (kolomstrook x rij-venster). Banding die maar in een deel
    van het beeld zit (één lampgroep, of alleen in de lucht/het doek) is zo toch te vinden: zulke
    banding geeft in meerdere naburige tegels een scherpe piek op dezelfde periode én dezelfde fase;
    content niet. Geeft [(P_half, score, bevestigd), ...]."""
    nst, H = profs8.shape
    from scipy.signal import find_peaks
    peaks = []
    for st in range(nst):
        for wi, (y0, y1) in enumerate(tile_windows(H)):
            tp = tile_profile(profs8[st, y0:y1])
            if tp is None:
                continue
            q, ok = tp
            L = y1 - y0
            Ps = np.exp(np.arange(np.log(20.0), np.log(L / 4.0), 0.005))
            if len(Ps) < 3:
                continue
            pw = sine_power_vec(q, Ps, np.hanning(L) * ok)
            pk, _ = find_peaks(pw)
            if len(pk) == 0:
                continue
            i = int(pk[np.argmax(pw[pk])])
            if pw[i] < min_power:
                continue
            peaks.append((float(Ps[i]), float(pw[i]), st, wi, q, ok))
    peaks.sort(key=lambda t: -t[1])
    used = [False] * len(peaks)
    out = []
    for i, (P, pw, st, wi, q, ok) in enumerate(peaks):
        if used[i]:
            continue
        mem = [j for j in range(len(peaks)) if not used[j] and abs(peaks[j][0] - P) <= 0.05 * P]
        for j in mem:
            used[j] = True
        if len(mem) < 3:
            continue
        Pm = float(np.median([peaks[j][0] for j in mem]))
        entries = []
        for j in mem:
            Pj, pwj, stj, wij, qj, okj = peaks[j]
            entries.append((wij, stj, tile_phase(qj, okj, Pm), (pwj, tile_periodic(qj, okj, Pm))))
        coh = coherent_tiles(entries)
        if len(coh) >= 3:
            score = float(np.median([x[0] for x in coh]))
            nper = sum(1 for x in coh if x[1])
            out.append((Pm, score, nper >= 0.5 * len(coh)))
    return out


def tile_check(profs8_b, profs8_a, P, amp_thr, min_power=0.30):
    """Zelfcontrole per tegel: in de tegels waar duidelijk periodieke banding op periode P zit
    (zelfde fase in naburige stroken), hoeveel blijft er na correctie over?
    Geeft (aantal tegels, mediaan amp voor, mediaan amp na)."""
    nst, H = profs8_b.shape
    entries = []
    for st in range(nst):
        for wi, (y0, y1) in enumerate(tile_windows(H)):
            if P > (y1 - y0) / 4.0:
                continue
            tb = tile_profile(profs8_b[st, y0:y1])
            ta = tile_profile(profs8_a[st, y0:y1])
            if tb is None or ta is None:
                continue
            qb, okb = tb
            qa, oka = ta
            if sine_power(qb, P, okb) < min_power or not tile_periodic(qb, okb, P):
                continue
            amp_b = band_amplitude(qb, P, nharm=6, r=okb)
            if amp_b < amp_thr:
                continue
            entries.append((wi, st, tile_phase(qb, okb, P), (st, amp_b, band_amplitude(qa, P, nharm=6, r=oka))))
    coh = coherent_tiles(entries)
    if len(coh) >= 3 and len({x[0] for x in coh}) >= 2:
        return len(coh), float(np.median([x[1] for x in coh])), float(np.median([x[2] for x in coh]))
    return 0, 0.0, 0.0


def band_amplitude(c, P, nharm=4, r=None):
    """Piek-dal amplitude (log-eenheden ~ fractie) van het periodieke deel van profiel c op periode P,
    inclusief de eerste harmonischen (P, P/2, ...): zo telt ook pulsachtige banding volledig mee."""
    pw = sum(sine_power(c, P / k, r) for k in range(1, nharm + 1) if P / k >= 4)
    ms = np.mean(c ** 2) if r is None else np.sum(r * c ** 2) / max(np.sum(r), 1e-9)
    return float(np.sqrt(2 * min(pw, 1.0) * ms) * 2)


def band_amplitude_set(c, Ps, r=None):
    """Als band_amplitude, maar over een opgegeven verzameling periodes (bv. de harmonischen van
    meerdere kandidaten samen): één meetlat om verschillende modellen eerlijk te vergelijken."""
    pw = sum(sine_power(c, P, r) for P in Ps)
    ms = np.mean(c ** 2) if r is None else np.sum(r * c ** 2) / max(np.sum(r), 1e-9)
    return float(np.sqrt(2 * min(pw, 1.0) * ms) * 2)


def sine_power(c, P, r=None):
    """Fractie van de energie van profiel c die één sinus met periode P (half-res rijen) verklaart."""
    H = len(c)
    w = np.hanning(H) if r is None else np.hanning(H) * r
    cw = c * w
    t = 2 * np.pi * np.arange(H) / P
    X = np.stack([np.cos(t) * w, np.sin(t) * w], axis=1)
    beta = np.linalg.lstsq(X, cw, rcond=None)[0]
    return float(np.sum((X @ beta) ** 2) / max(np.sum(cw ** 2), 1e-9))


def estimate_period(plane, min_rows=40, max_rows=2000, prof=None):
    """Vind de bandperiode (in half-res rijen) met een periodogram van het gemeenschappelijke profiel.

    Geeft (periode_half, power, profiel). Power = fractie van de profiel-energie op die frequentie
    (0.3+ is een duidelijke band; < 0.1 betekent: geen betrouwbare periodieke banding).
    """
    c, r, profs = prof if prof is not None else common_profile(plane, with_weight=True, with_strips=True)
    H = len(c)
    Pmax = min(max_rows / 2.0, H / 4.0)                  # minstens 4 periodes in beeld
    Ps = np.arange(min_rows / 2.0, Pmax, 0.25)
    pw = sine_power_vec(c, Ps, np.hanning(H) * r)
    # alleen echte (lokale) pieken: een oplopende flank naar de rand van het zoekbereik is content
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(pw)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(pw))])
    i = int(peaks[np.argmax(pw[peaks])])
    P0, p0 = Ps[i], pw[i]
    # Zit er ook duidelijk energie op de dubbele periode (afwisselend diepe/ondiepe banden)?
    # Dan is 2*P0 de echte grondperiode en P0 de 2e harmonische.
    if 2 * P0 < Pmax:
        j = int(np.argmin(np.abs(Ps - 2 * P0)))
        p2 = pw[max(0, j - 8):j + 9].max()
        if p2 >= 0.25 * p0:
            P0 = Ps[max(0, j - 8) + int(np.argmax(pw[max(0, j - 8):j + 9]))]
    # Fijnafstelling: periode die met het volledige harmonische model het profiel het best verklaart
    y = np.arange(H)
    best_P, best_E = P0, -1.0
    K = int(min(8, max(2, P0 // 10)))
    w = np.hanning(H) * r
    cw = c * w
    # Het periodogram heeft een grove resolutie (dP ~ P^2/H, bij 220 rijen zo'n 12 rijen), dus
    # ruim zoeken: ±6%. De harmonische fit is scherp gepiekt op de echte periode.
    for P in np.arange(P0 * 0.94, P0 * 1.06, 0.05):
        X = fourier_basis(y, P, K) * w[:, None]
        beta = np.linalg.lstsq(X, cw, rcond=None)[0]
        E = np.sum((X @ beta) ** 2)
        if E > best_E:
            best_P, best_E = P, E
    return best_P, p0, (c, r, profs)


# ----------------------------------------------------------------------------
# core
# ----------------------------------------------------------------------------
def estimate_gain(planes, period_half, K, nstrips, iterations, nwin=0, verbose=True):
    """
    planes: dict naam -> half-res plane (H, W) lineair, zwart al afgetrokken.
    Schat per kanaal een 2D veld van Fourier-coëfficiënten: per kolomstrook én per
    rij-venster (tegels), zodat de sterkte en fase van de banding over het beeld mogen
    variëren (verschillende lampen belichten verschillende delen). Tegels met weinig
    data worden naar hun strook getrokken, stroken met weinig data naar het globale model.
    Geeft coef[name] -> (c (nstrips, nwin, 2K), gain_cols (H, nstrips)), strookcentra, venstercentra.
    """
    H, W = next(iter(planes.values())).shape
    yy = np.arange(H)
    B = fourier_basis(yy, period_half, K)                      # (H, 2K)
    edges = np.linspace(0, W, nstrips + 1).astype(int)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if nwin <= 0:
        nwin = int(max(1, round(H / (4.5 * period_half))))     # ~4-5 bandperiodes per venster
    wcenters = (np.arange(nwin) + 0.5) * H / nwin
    half_h = int(0.75 * H / nwin)
    Wy = interp_weights(yy, wcenters)                          # (H, nwin) lineaire gewichten
    # Sub-stroken: 2 per strook; een strook (1.5x breed, overlappend) = 4 sub-stroken.
    coef = {}
    for name, P in planes.items():
        white = P.max()
        logP = np.log(np.maximum(P, 4.0))
        # masker (eenmalig): niet te donker, niet geclipt, geen sterke verticale structuur
        dy = np.abs(np.diff(gaussian_filter(logP, 2.0), axis=0, prepend=logP[:1]))
        valid = (P > 50) & (P < 0.92 * white) & (dy < 0.03)
        del dy
        gain = np.ones((H, W), dtype=np.float32)
        c = np.zeros((nstrips, nwin, 2 * K))
        for it in range(iterations):
            corrected = logP - np.log(gain)
            trend = smooth_rows(corrected, period_half)
            ratio = np.where(valid, logP - trend, np.nan)       # log-gain samples
            n_all = np.sum(~np.isnan(ratio[:, ::2]), axis=1)
            med_all = np.clip(np.nan_to_num(np.nanmedian(ratio[:, ::2], axis=1)), -0.8, 0.8)
            ok_all = n_all > max(10, 0.05 * W)
            c_glob = robust_fit(B[ok_all], med_all[ok_all], np.sqrt(n_all[ok_all] / n_all[ok_all].max())) \
                if ok_all.sum() >= 4 * K + 10 else np.zeros(2 * K)
            amp_glob = float(np.ptp(B @ c_glob))
            gain_cols = np.zeros((H, nstrips), dtype=np.float32)
            half_w = int(0.75 * W / nstrips)
            for s in range(nstrips):
                x0 = max(0, int(centers[s]) - half_w)
                x1 = min(W, int(centers[s]) + half_w)
                block = ratio[:, x0:x1:2]
                n = np.sum(~np.isnan(block), axis=1)
                med = np.clip(np.nan_to_num(np.nanmedian(block, axis=1)), -0.8, 0.8)
                ok = n > max(10, 0.1 * (x1 - x0))
                w = np.sqrt(n / max(n.max(), 1))
                # strook-fit over de volle hoogte, geschoven naar globaal bij weinig data
                c_s = shrink_fit(B, med, w, ok, K, c_glob, amp_glob)
                amp_s = float(np.ptp(B @ c_s))
                # tegel-fits per rij-venster, geschoven naar de strook-fit
                logg = np.zeros((H, nwin))
                for j in range(nwin):
                    y0 = max(0, int(wcenters[j]) - half_h)
                    y1 = min(H, int(wcenters[j]) + half_h)
                    sel = np.zeros(H, bool)
                    sel[y0:y1] = True
                    c_sj = shrink_fit(B, med, w, ok & sel, K, c_s, amp_s, cap=2.0) if nwin > 1 else c_s
                    c[s, j] = c_sj
                    logg[:, j] = B @ c_sj
                gain_cols[:, s] = np.exp(np.sum(Wy * logg, axis=1))
            # glad over stroken (3-tap) en interpoleer over kolommen
            if nstrips >= 3:
                sm = gain_cols.copy()
                sm[:, 1:-1] = 0.25 * gain_cols[:, :-2] + 0.5 * gain_cols[:, 1:-1] + 0.25 * gain_cols[:, 2:]
                gain_cols = sm
            gain = np.empty((H, W), dtype=np.float32)
            xs = np.arange(W)
            for r in range(H):
                gain[r] = np.interp(xs, centers, gain_cols[r])
            gain = np.clip(gain, 0.4, 1.6)
            if verbose:
                amp = gain_cols.max(axis=0) / gain_cols.min(axis=0) - 1
                print(f"  [{name}] iter {it + 1}: band-amplitude per strook "
                      + " ".join(f"{a * 100:.1f}%" for a in amp))
        coef[name] = (c, gain_cols)
    return coef, centers, wcenters


def smooth_rows(a, period_half):
    """Verticale trend: 3x een box-filter van precies één bandperiode. Een box van lengte P
    heeft een nulpunt op P en al zijn harmonischen, dus de banding valt volledig uit de trend;
    en het is O(1) per pixel, ongeacht de periode."""
    n = max(3, int(round(period_half)))
    out = a
    for _ in range(3):
        out = uniform_filter1d(out, size=n, axis=0, mode='nearest')
    return out


def interp_weights(y, knots):
    """Lineaire interpolatiegewichten (len(y), len(knots)); constant buiten de buitenste knopen."""
    y = np.asarray(y, dtype=np.float64)
    n = len(knots)
    Wm = np.zeros((len(y), n))
    if n == 1:
        Wm[:, 0] = 1.0
        return Wm
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        Wm[:, j] = np.interp(y, knots, e)
    return Wm


def shrink_fit(B, med, w, ok, K, prior, amp_prior, cap=2.5):
    """Robuuste fit op de rijen 'ok'; geschoven naar 'prior' naarmate er minder data is."""
    if ok.sum() < 4 * K + 10:
        return prior.copy()
    beta = robust_fit(B[ok], med[ok], w[ok])
    frac = ok.sum() / max(1, len(ok)) if prior is None else ok.mean()
    lam = 1.0 - min(1.0, frac / 0.6)
    beta = (1 - lam) * beta + lam * prior
    if amp_prior > 0 and np.ptp(B @ beta) > cap * amp_prior + 0.02:
        return prior.copy()
    return beta


def gain_plane(c, centers, wcenters, period_half, K, H_sub, W_sub, nstrips, strength):
    """Evalueer het 2D model voor één CFA-subplane (zelfde raster als de half-res schatting)."""
    yy = np.arange(H_sub)
    B = fourier_basis(yy, period_half, K)                      # (H_sub, 2K)
    Wy = interp_weights(yy, wcenters)                          # (H_sub, nwin)
    cols = np.zeros((H_sub, nstrips))
    for s in range(nstrips):
        logg = B @ c[s].T                                      # (H_sub, nwin)
        cols[:, s] = np.exp(strength * np.sum(Wy * logg, axis=1))
    if nstrips >= 3:
        sm = cols.copy()
        sm[:, 1:-1] = 0.25 * cols[:, :-2] + 0.5 * cols[:, 1:-1] + 0.25 * cols[:, 2:]
        cols = sm
    xs = np.arange(W_sub)
    out = np.empty((H_sub, W_sub), dtype=np.float32)
    for r in range(H_sub):
        out[r] = np.interp(xs, centers, cols[r])
    return np.clip(out, 0.4, 1.6)


def process(path, args, known_periods=None):
    base = os.path.splitext(os.path.basename(path))[0]
    outdir = args.outdir or os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)
    print(f"\n== {os.path.basename(path)}")
    raw = rawpy.imread(path)
    black = float(np.mean(raw.black_level_per_channel))
    white = float(raw.white_level)
    vis = raw.raw_image_visible
    lin = np.clip(vis.astype(np.float32) - black, 0, None)
    pat = raw.raw_pattern.copy()
    desc = raw.color_desc.decode()           # bv 'RGBG'
    # metadata vastleggen vóór postprocess() (LibRaw past daarna interne velden aan)
    meta = dict(pattern=pat, desc=desc, sizes=raw.sizes, flip=raw.sizes.flip,
                cm=np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float64),
                wb=np.array(raw.camera_whitebalance[:3], dtype=np.float64))
    # kanaalnaam per positie in het 2x2 patroon
    names = {}
    for i in range(2):
        for j in range(2):
            c = desc[pat[i, j]]
            names[(i, j)] = c
    H_full, W_full = lin.shape
    planes = {}
    for (i, j), c in names.items():
        pl = lin[i::2, j::2]
        key = c
        if c == 'G':
            key = 'G'
        planes.setdefault(key, []).append(((i, j), pl))
    # G: gemiddelde van beide groene posities voor de schatting
    est_planes = {}
    for key, lst in planes.items():
        est_planes[key] = np.mean([p for _, p in lst], axis=0) if len(lst) > 1 else lst[0][1]
    Hh, Wh = est_planes['G'].shape

    # Schatting in (max.) meerdere passes: elke pass vindt de sterkste periodieke banding die nog
    # over is (verschillende lampen kunnen verschillende PWM-frequenties hebben) en deelt die weg.
    result = dict(file=os.path.basename(path), period=0.0, power=0.0, before=None, after=None, status='', conf=False)
    g_total = {key: np.ones_like(pl) for key, pl in est_planes.items()}
    est_cur = est_planes
    passes_done = 0
    for npass in range(1, args.passes + 1):
        cands, info = period_candidates(est_cur['G'])
        prof_before, profs8_before = info['prof'], info['profs8']
        cands = [(P, sc, cf, sc, False) for P, sc, cf in cands]   # (periode, score, bevestigd, echte power, shoot-periode?)
        if known_periods and npass == 1:
            # Periodes die elders in deze shoot zeker zijn vastgesteld: lampfrequenties veranderen niet
            # binnen een shoot, dus die worden hier ook geprobeerd, ook als het globale profiel zwak is.
            # Wel per foto verfijnd en op deze foto gecontroleerd (herhaalt het patroon zich echt?).
            kp = [k / 2.0 for k in known_periods]
            c30, r30 = info['c30'], info['r30']
            c50, r50 = prof_before
            # detecties dicht bij een shoot-periode vervangen door de (per foto verfijnde) shoot-periode:
            # een periodogram-schatting kan er een paar procent naast zitten, en dat is over 18 banden fataal
            cands = [cd for cd in cands if all(abs(cd[0] - k) > 0.08 * k for k in kp)]
            for k in kp:
                Pk = refine_period(c30, k, span=0.06, r=r30)
                if abs(Pk - k) >= 0.055 * k:
                    continue                                   # loopt tegen de zoekrand: zit hier niet
                cands.append((Pk, 1.0, bool(info['confirm'](Pk)), sine_power(c50, Pk, r50), True))
        tag = f" (pass {npass})" if npass > 1 else ""
        if args.period and npass == 1:
            cands = [(args.period / 2.0, 1.0, True, 1.0, False)]
            print(f"  periode: {args.period:.1f} rijen (opgegeven)")
        else:
            pw_thr = 0.20 if npass == 1 else 0.25
            cands = [cd for cd in cands if cd[1] >= pw_thr or args.force]
            if not cands:
                if npass == 1:
                    print("  geen duidelijke periodieke banding gevonden; bestand overgeslagen (--period of --force om toch te doen)")
                break
        evaluated = []
        for period_half, power, conf, power_true, shoot in cands:
            K = int(min(args.harmonics, max(2, period_half // 10)))   # fijnste harmonische >= ~20 rijen
            coef, centers, wcenters = estimate_gain(est_cur, period_half, K, args.strips,
                                                    args.iterations, nwin=args.ywin, verbose=args.verbose)
            g = {}
            test = {}
            for key, pl in est_cur.items():
                g[key] = gain_plane(coef[key][0], centers, wcenters, period_half, K, pl.shape[0], pl.shape[1],
                                    args.strips, args.strength)
                test[key] = pl / g[key]
            prof_after, profs32_after = common_profile(test['G'], with_strips=True)
            c_b, r_b = prof_before
            amp_b = band_amplitude(c_b, period_half, nharm=6, r=r_b)
            amp_a = band_amplitude(prof_after, period_half, nharm=6, r=r_b)
            # Bevestigde periodiciteit (autocorrelatie herhaalt op P en 2P): gewone eisen.
            # Onbevestigd: alleen accepteren als de correctie overtuigend werkt (anders is het content).
            # Een shoot-periode (elders in deze map zeker vastgesteld) zit daar tussenin: lampfrequenties
            # veranderen niet binnen een shoot, dus bij >= 2% banding en >= 60% reductie is het banding.
            if conf:
                amp_thr, red_thr = (0.010 if npass == 1 else 0.015), 0.6
            elif shoot:
                amp_thr, red_thr = 0.020, 0.4
            else:
                amp_thr, red_thr = 0.020, 0.25
            passes = (amp_b >= amp_thr) and (amp_a <= red_thr * amp_b) or args.force
            local = False
            if not passes and npass == 1:
                # Banding in een deel van het beeld (of veel content in de rest): het globale profiel
                # zegt dan weinig. Kijk per tegel waar echt periodieke banding op deze periode zit.
                profs8_after = np.array([np.nanmedian(profs32_after[4 * s:4 * s + 4], axis=0) for s in range(8)])
                nt, tb, ta = tile_check(profs8_before, profs8_after, period_half, 0.02)
                if nt and ta <= 0.5 * tb:
                    passes, local = True, True
                    amp_b, amp_a = tb, ta
            removed = amp_b - amp_a
            if args.verbose and len(cands) > 1:
                print(f"  kandidaat {2 * period_half:.1f} rijen ({'bevestigd' if conf else 'onbevestigd'}): "
                      f"band {amp_b * 100:.1f}% -> {amp_a * 100:.1f}% ({'voldoet' if passes else 'voldoet niet'})")
            evaluated.append(dict(period_half=period_half, power=power_true, K=K, coef=coef, g=g, test=test,
                                  prof_after=prof_after, amp_b=amp_b, amp_a=amp_a, passes=passes, conf=conf,
                                  local=local, amp_thr=amp_thr, red_thr=red_thr, removed=removed))
        # keuze: de kandidaat die de zelfcontrole haalt én het minste banding overlaat, gemeten met één
        # gezamenlijke meetlat (alle harmonischen van alle kandidaten samen): een 225-model meet zichzelf
        # anders alleen op 225 en ziet niet dat het de 450-component laat staan.
        # Haalt geen enkele kandidaat de zelfcontrole, dan de sterkste (voor de rapportage).
        freqs = []
        for e in evaluated:
            for k in range(1, 7):
                Q = e['period_half'] / k
                if Q >= 4 and all(abs(Q - F) > 0.02 * F for F in freqs):
                    freqs.append(Q)
        c_b, r_b = prof_before
        for e in evaluated:
            e['rest'] = band_amplitude_set(e['prof_after'], freqs, r_b)
        if any(e['passes'] for e in evaluated):
            best = min((e for e in evaluated if e['passes']), key=lambda e: e['rest'])
        else:
            best = max(evaluated, key=lambda e: e['amp_b'])
        if args.verbose and len(evaluated) > 1:
            print("  gezamenlijke meetlat (rest na correctie): " + ", ".join(
                f"{2 * e['period_half']:.0f}r {e['rest'] * 100:.1f}%" for e in evaluated))
        period_half, power, K, coef, g, test = (best['period_half'], best['power'], best['K'], best['coef'],
                                                best['g'], best['test'])
        amp_b, amp_a, conf, local = best['amp_b'], best['amp_a'], best['conf'], best['local']
        if not (args.period and npass == 1):
            print(f"  periode{tag}: {2 * period_half:.1f} sensorrijen (bandsignaal {power * 100:.0f}% van het profiel, "
                  f"{'periodiciteit bevestigd' if conf else 'niet bevestigd'})")
        if args.verbose:
            print(f"  model: {args.strips} stroken x {len(wcenters)} rij-vensters, {K} harmonischen")
        amp_per_strip = np.median([cols.max(axis=0) / cols.min(axis=0) - 1 for _, cols in coef.values()], axis=0)
        print(f"  banding{tag}: ~{amp_b * 100:.1f}% -> ~{amp_a * 100:.1f}% (piek-dal, groen kanaal"
              f"{', gemeten in het deel van het beeld met banding' if local else ''}); "
              f"gemeten per strook: " + " ".join(f"{a * 100:.0f}%" for a in amp_per_strip))
        amp_thr, red_thr = best['amp_thr'], best['red_thr']
        if local:
            amp_thr, red_thr = 0.0, 0.5
        if amp_b < amp_thr and not args.force:
            if npass == 1:
                print(f"  weinig banding gevonden (<{amp_thr * 100:.0f}%); bestand overgeslagen")
                result['status'] = 'weinig banding'
            break
        if amp_a > red_thr * amp_b and not args.force:
            if npass == 1:
                if power >= 0.30 and amp_b >= 0.015 and conf:
                    print("  !! duidelijk bandsignaal, maar de correctie helpt te weinig (model past niet); overgeslagen.")
                    print("     Bekijk deze foto; probeer --period <rijen>, --strips 4, of --force.")
                    result['status'] = 'model past niet'
                elif power >= 0.30 and amp_b >= 0.015:
                    print("  periodiek signaal, maar het herhaalt zich niet zuiver en de correctie helpt niet: "
                          "waarschijnlijk content (doek, tribune, lampenrij). Overgeslagen.")
                    result['status'] = 'onzeker (waarschijnlijk content)'
                else:
                    print("  zwak signaal en de correctie helpt niet; waarschijnlijk geen banding. Overgeslagen.")
                    result['status'] = 'geen banding (zwak signaal)'
            break
        # pass geaccepteerd
        passes_done += 1
        for key in g_total:
            g_total[key] *= g[key]
        est_cur = test
        if npass == 1:
            result.update(period=2 * period_half, power=power, before=amp_b, conf=bool(conf))
        result['after'] = amp_a
    if passes_done == 0:
        raw.close()
        if not result['status']:
            result['status'] = 'geen banding'
        return result
    if args.check:
        raw.close()
        result['status'] = 'te corrigeren' + (f' ({passes_done} passes)' if passes_done > 1 else '')
        return result
    result['status'] = 'gecorrigeerd' + (f' ({passes_done} passes)' if passes_done > 1 else '')

    # toepassen op volle resolutie, per CFA-positie (zelfde raster als de half-res schatting)
    corrected = np.empty_like(lin)
    for (i, j), c in names.items():
        key = 'G' if c == 'G' else c
        sub = lin[i::2, j::2]
        gt = g_total[key]
        if gt.shape != sub.shape:
            gt = np.pad(gt, ((0, max(0, sub.shape[0] - gt.shape[0])), (0, max(0, sub.shape[1] - gt.shape[1]))),
                        mode='edge')[:sub.shape[0], :sub.shape[1]]
        corrected[i::2, j::2] = sub / gt
    corrected = np.clip(corrected + black, 0, white)
    # Pixels die in het origineel al geclipt waren (lampen, spots) blijven geclipt: hun echte waarde
    # is onbekend, en Lightroom moet ze als 'uitgebeten' blijven herkennen voor highlight-herstel.
    corrected[vis >= white] = white

    # schrijf terug in LibRaw buffer
    vis[:, :] = np.round(corrected).astype(vis.dtype)

    if args.dng:
        try:
            write_dng(meta, path, os.path.join(outdir, f"{base}_deband.dng"), corrected, black, white,
                      compress=not args.no_compress)
        except Exception as e:
            import traceback
            print(f"  DNG schrijven mislukt: {e}")
            traceback.print_exc()

    if args.preview:
        # voor/na jpg met dezelfde ontwikkeling
        from PIL import Image
        raw_before = rawpy.imread(path)
        b = raw_before.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8, half_size=True)
        a = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8, half_size=True)
        Image.fromarray(b).save(os.path.join(outdir, f"{base}_before.jpg"), quality=92)
        Image.fromarray(a).save(os.path.join(outdir, f"{base}_after.jpg"), quality=92)
        raw_before.close()
        print("  preview: _before.jpg / _after.jpg")

    if args.tiff:
        import tifffile
        rgb16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                output_color=rawpy.ColorSpace.sRGB)
        tpath = os.path.join(outdir, f"{base}_deband.tif")
        tifffile.imwrite(tpath, rgb16, photometric='rgb', compression='zlib')
        print(f"  tiff: {tpath}  (sRGB, 16-bit, camera-WB)")

    raw.close()
    return result


def read_cr3_exif(path):
    """Haal Make/Model/belichting/orientatie uit de CMT1/CMT2 boxes van een CR3 (of EXIF van andere RAW)."""
    info = {}
    try:
        import exifread
        import io
        with open(path, 'rb') as f:
            data = f.read(4_000_000)
        found = False
        for box in (b'CMT1', b'CMT2'):
            i = data.find(box)
            if i < 0:
                continue
            found = True
            tags = exifread.process_file(io.BytesIO(data[i + 4:i + 4 + 300000]), details=False)
            for k, v in tags.items():
                info[k.split(' ', 1)[-1]] = v
        if not found:   # geen CR3: gewone EXIF
            with open(path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            for k, v in tags.items():
                info[k.split(' ', 1)[-1]] = v
    except Exception:
        pass
    return info


def _rat(tagval, scale=1000000):
    """exifread Ratio -> [num, den]."""
    try:
        v = tagval.values[0]
        return [int(v.num), int(v.den)]
    except Exception:
        try:
            f = float(str(tagval))
            return [int(round(f * scale)), scale]
        except Exception:
            return None


# ----------------------------------------------------------------------------
# Lossless JPEG (ITU T.81 proces 14, SOF3) encoder in NumPy, voor DNG-tiles.
# Twee componenten per rij (even/oneven kolom = twee Bayer-kleuren), predictor 1,
# zoals Adobe DNG Converter het doet. Bit-exact (geverifieerd met LibRaw en imagecodecs).
# ----------------------------------------------------------------------------
def _build_huffman(hist):
    """JPEG Annex K.2: optimale codelengtes (max 16) voor symbolen 0..len(hist)-1.
    Geeft (bits[1..16], huffval, code_of, len_of)."""
    nsym = len(hist)
    freq = np.zeros(257, dtype=np.float64)
    freq[:nsym] = hist
    freq[256] = 1.0                                     # gereserveerd codepunt
    codesize = np.zeros(257, dtype=np.int64)
    others = -np.ones(257, dtype=np.int64)
    while True:
        nz = np.nonzero(freq > 0)[0]
        if len(nz) < 2:
            break
        order = sorted(nz, key=lambda i: (freq[i], -i))
        v1, v2 = int(order[0]), int(order[1])
        freq[v1] += freq[v2]
        freq[v2] = 0
        codesize[v1] += 1
        while others[v1] != -1:
            v1 = int(others[v1])
            codesize[v1] += 1
        others[v1] = v2
        codesize[v2] += 1
        while others[v2] != -1:
            v2 = int(others[v2])
            codesize[v2] += 1
    bits = np.zeros(40, dtype=np.int64)
    for i in range(257):
        if codesize[i] > 0:
            bits[codesize[i]] += 1
    i = 39
    while i > 16:
        while bits[i] > 0:
            j = i - 2
            while bits[j] == 0:
                j -= 1
            bits[i] -= 2
            bits[i - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1
        i -= 1
    while bits[i] == 0:
        i -= 1
    bits[i] -= 1                                        # gereserveerd codepunt eruit
    huffval = [s for L in range(1, 40) for s in range(256) if codesize[s] == L]
    code_of = np.zeros(nsym, dtype=np.uint64)
    len_of = np.zeros(nsym, dtype=np.int64)
    code = 0
    k = 0
    for L in range(1, 17):
        for _ in range(int(bits[L])):
            s = huffval[k]
            k += 1
            code_of[s] = code
            len_of[s] = L
            code += 1
        code <<= 1
    return bits[1:17], huffval[:k], code_of, len_of


def _pack_bits(word, wlen):
    """Pak woorden van wlen bits (MSB eerst) tot een bytestring, opgevuld met 1-bits."""
    total = int(wlen.sum())
    padded = (total + 7) // 8 * 8
    bitarr = np.ones(padded, dtype=np.uint8)
    offsets = np.empty(len(wlen), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(wlen[:-1], out=offsets[1:])
    maxlen = int(wlen.max())
    idx_all = np.arange(len(wlen))
    for k in range(maxlen):
        m = wlen > k
        idx = idx_all[m]
        shift = (wlen[idx] - 1 - k).astype(np.uint64)
        bitarr[offsets[idx] + k] = ((word[idx] >> shift) & np.uint64(1)).astype(np.uint8)
    packed = np.packbits(bitarr)
    # byte stuffing: 0xFF -> 0xFF 0x00
    ff = packed == 0xFF
    if ff.any():
        before = np.cumsum(ff) - ff
        out = np.zeros(len(packed) + int(ff.sum()), dtype=np.uint8)
        out[np.arange(len(packed)) + before] = packed
        packed = out
    return packed.tobytes()


def ljpeg_encode_2c(tile, precision=16):
    """Codeer een (h, w) uint16 Bayer-tile (w even) als lossless JPEG met 2 componenten."""
    h, w = tile.shape
    assert w % 2 == 0
    comp = tile.reshape(h, w // 2, 2).astype(np.int32)
    pred = np.empty_like(comp)
    pred[:, 1:, :] = comp[:, :-1, :]                    # links, zelfde component
    pred[1:, 0, :] = comp[:-1, 0, :]                    # eerste kolom: boven
    pred[0, 0, :] = 1 << (precision - 1)
    d = comp - pred
    d = ((d + 32768) & 0xFFFF) - 32768                  # modulo 2^16 naar signed
    flat = d.reshape(-1)
    mag = np.abs(flat)
    ssss = np.frexp(mag.astype(np.float64))[1].astype(np.int64)   # bitlengte (0 voor 0)
    extra_bits = np.where(ssss == 16, 0, ssss)
    extra = (np.where(flat >= 0, flat, flat - 1) & ((1 << extra_bits) - 1)).astype(np.uint64)
    hist = np.bincount(ssss, minlength=17)[:17]
    bits, huffval, code_of, len_of = _build_huffman(hist)
    hcode = code_of[ssss]
    hlen = len_of[ssss]
    word = (hcode << extra_bits.astype(np.uint64)) | extra
    wlen = hlen + extra_bits
    payload = _pack_bits(word, wlen)

    out = bytearray(b'\xff\xd8')                                        # SOI
    dht = bytes([0x00]) + bytes(int(b) for b in bits) + bytes(huffval)
    out += b'\xff\xc4' + (len(dht) + 2).to_bytes(2, 'big') + dht           # DHT
    sof = bytes([precision]) + h.to_bytes(2, 'big') + (w // 2).to_bytes(2, 'big') + bytes([2])
    sof += bytes([1, 0x11, 0, 2, 0x11, 0])
    out += b'\xff\xc3' + (len(sof) + 2).to_bytes(2, 'big') + sof           # SOF3
    sos = bytes([2, 1, 0x00, 2, 0x00, 1, 0, 0x00])                        # Ns=2, predictor 1, Pt 0
    out += b'\xff\xda' + (len(sos) + 2).to_bytes(2, 'big') + sos           # SOS
    out += payload
    out += b'\xff\xd9'                                                   # EOI
    return bytes(out)


def write_dng(meta, src_path, out_path, corrected_visible, black, white, compress=True):
    """Schrijf de gecorrigeerde Bayer-data als DNG (eigen TIFF/DNG-writer, geen compiler nodig)."""
    data = np.round(corrected_visible).astype(np.uint16)
    H, W = data.shape
    pat = meta['pattern']
    desc = meta['desc']
    code = {'R': 0, 'G': 1, 'B': 2}
    cfa = [code[desc[pat[i, j]]] for i in range(2) for j in range(2)]
    # LibRaw rgb_xyz_matrix = Adobe ColorMatrix (XYZ -> camera, D65)
    cm = meta['cm']
    if not np.any(cm):
        cm = np.eye(3)
    wb = meta['wb']
    wb = wb / wb[1]
    exif = read_cr3_exif(src_path)
    make = str(exif.get('Make', 'Unknown')).strip()
    model = str(exif.get('Model', 'Unknown')).strip()
    unique = model if model.lower().startswith(make.lower()) else f"{make} {model}"
    ori_map = {'Horizontal (normal)': 1, 'Rotated 180': 3, 'Rotated 90 CW': 6, 'Rotated 90 CCW': 8}
    orientation = ori_map.get(str(exif.get('Orientation', '')), {6: 6, 5: 8, 3: 3}.get(meta['flip'], 1))

    def srat(v, den=10000):          # signed rational
        return (int(round(v * den)), den)

    def to_rat(tagval):
        r = _rat(tagval)
        return tuple(r) if r else None

    s = meta['sizes']
    tags = [
        (254, 'I', 1, 0),                                   # NewSubfileType: main image
        (274, 'H', 1, orientation),                         # Orientation
        (271, 's', None, make),                             # Make
        (272, 's', None, model),                            # Model
        (33421, 'H', 2, (2, 2)),                            # CFARepeatPatternDim
        (33422, 'B', 4, tuple(cfa)),                        # CFAPattern
        (50706, 'B', 4, (1, 4, 0, 0)),                      # DNGVersion 1.4
        (50707, 'B', 4, (1, 2, 0, 0)),                      # DNGBackwardVersion 1.2
        (50708, 's', None, unique),                         # UniqueCameraModel
        (50714, 'I', 1, int(round(black))),                 # BlackLevel
        (50717, 'I', 1, int(white)),                        # WhiteLevel
        (50721, '2i', 9, tuple(x for v in cm.flatten() for x in srat(v))),   # ColorMatrix1 (SRATIONAL)
        (50728, '2I', 3, tuple(x for v in wb for x in srat(1.0 / v))),       # AsShotNeutral (RATIONAL)
        (50730, '2i', 1, (0, 1)),                           # BaselineExposure
        (50778, 'H', 1, 21),                                # CalibrationIlluminant1 = D65
    ]
    if s.crop_width > 0 and s.crop_height > 0:
        tags.append((50719, 'I', 2, (max(0, s.crop_left_margin - s.left_margin),
                                      max(0, s.crop_top_margin - s.top_margin))))   # DefaultCropOrigin
        tags.append((50720, 'I', 2, (min(s.crop_width, W), min(s.crop_height, H))))  # DefaultCropSize
    # EXIF-basisgegevens in IFD0 (toegestaan in TIFF/EP en DNG)
    for key, code_, kind in (('ExposureTime', 33434, '2I'), ('FNumber', 33437, '2I'),
                             ('FocalLength', 37386, '2I')):
        if key in exif:
            r = to_rat(exif[key])
            if r:
                tags.append((code_, kind, 1, r))
    if 'ISOSpeedRatings' in exif:
        try:
            tags.append((34855, 'H', 1, int(str(exif['ISOSpeedRatings']))))
        except Exception:
            pass
    if 'DateTimeOriginal' in exif:
        tags.append((36867, 's', None, str(exif['DateTimeOriginal'])))
        tags.append((306, 's', None, str(exif['DateTimeOriginal'])))
    if 'LensModel' in exif:
        tags.append((42036, 's', None, str(exif['LensModel'])))

    tags += [
        (256, 'I', 1, W),                                   # ImageWidth
        (257, 'I', 1, H),                                   # ImageLength
        (258, 'H', 1, 16),                                  # BitsPerSample
        (262, 'H', 1, 32803),                               # PhotometricInterpretation = CFA
        (277, 'H', 1, 1),                                   # SamplesPerPixel
        (284, 'H', 1, 1),                                   # PlanarConfiguration
        (50711, 'H', 1, 1),                                 # CFALayout: rectangular
        (50710, 'B', 3, (0, 1, 2)),                         # CFAPlaneColor: R G B
    ]
    if compress:
        # Tiles van 256 rijen, 2 kolommen (breedte <= beeldbreedte: LibRaw gaat anders de mist in),
        # elk een eigen lossless-JPEG stream. Randen opgevuld door de laatste 2 rijen/kolommen te herhalen.
        TL = 256
        ncols = 1 if W % 16 == 0 else 2
        TW = ((W + ncols - 1) // ncols + 15) // 16 * 16
        blobs = []
        for y in range(0, H, TL):
            for x in range(0, ncols * TW, TW):
                t = np.empty((TL, TW), dtype=np.uint16)
                blk = data[y:y + TL, x:x + TW]
                h, w = blk.shape
                t[:h, :w] = blk
                if w < TW:
                    t[:h, w:] = np.tile(blk[:, w - 2:w], (1, (TW - w) // 2))
                if h < TL:
                    t[h:] = np.tile(t[h - 2:h], ((TL - h) // 2, 1))
                blobs.append(ljpeg_encode_2c(t))
        tags += [(259, 'H', 1, 7), (322, 'I', 1, TW), (323, 'I', 1, TL)]   # Compression=JPEG, TileWidth/Length
        write_tiff(out_path, tags, blobs, tiled=True)
    else:
        blobs = [np.ascontiguousarray(data).tobytes()]
        tags += [(259, 'H', 1, 1), (278, 'I', 1, H)]                       # Compression=none, RowsPerStrip
        write_tiff(out_path, tags, blobs, tiled=False)
    mb = os.path.getsize(out_path) / 1e6
    print(f"  dng: {out_path}  ({unique}, {mb:.0f} MB, {'lossless gecomprimeerd' if compress else 'ongecomprimeerd'})")


def write_tiff(path, tags, blobs, tiled):
    """Minimale TIFF-schrijver (little-endian, klassiek TIFF) voor één IFD met tiles of één strip.

    tags: lijst (code, type, count, value) met type 'B' (BYTE), 'H' (SHORT), 'I' (LONG),
          's' (ASCII), '2I' (RATIONAL), '2i' (SRATIONAL). blobs: de (gecomprimeerde) tile-data.
    Geen afhankelijkheid van tifffile/imagecodecs, zodat compressie 7 (lossless JPEG) altijd werkt.
    """
    import struct
    TYPES = {'B': (1, 1, 'B'), 's': (2, 1, None), 'H': (3, 2, 'H'), 'I': (4, 4, 'I'),
             '2I': (5, 8, 'II'), '2i': (10, 8, 'ii')}
    # tile-data direct na de header; offsets vastleggen
    offsets, counts = [], []
    pos = 8
    for b in blobs:
        offsets.append(pos)
        counts.append(len(b))
        pos += len(b) + (len(b) & 1)                       # word-aligned
    off_tag, cnt_tag = (324, 325) if tiled else (273, 279)
    tags = list(tags) + [(off_tag, 'I', len(offsets), tuple(offsets)), (cnt_tag, 'I', len(counts), tuple(counts))]
    tags.sort(key=lambda t: t[0])
    ifd_offset = pos
    n = len(tags)
    entries = bytearray()
    extra = bytearray()
    extra_base = ifd_offset + 2 + 12 * n + 4
    for code, kind, count, value in tags:
        ttype, size, fmt = TYPES[kind]
        if kind == 's':
            payload = str(value).encode('utf-8', 'replace') + b'\x00'
            count = len(payload)
        else:
            vals = value if isinstance(value, (tuple, list)) else (value,)
            if kind in ('2I', '2i'):
                count = len(vals) // 2
                payload = struct.pack('<' + fmt[0] * len(vals), *[int(v) for v in vals])
            else:
                count = len(vals)
                payload = struct.pack('<' + fmt * len(vals), *[int(v) for v in vals])
        if len(payload) <= 4:
            field = payload + b'\x00' * (4 - len(payload))
        else:
            field = struct.pack('<I', extra_base + len(extra))
            extra += payload
            if len(extra) & 1:
                extra += b'\x00'
        entries += struct.pack('<HHI', code, ttype, count) + field
    with open(path, 'wb') as f:
        f.write(b'II*\x00' + struct.pack('<I', ifd_offset))
        for b in blobs:
            f.write(b)
            if len(b) & 1:
                f.write(b'\x00')
        f.write(struct.pack('<H', n) + bytes(entries) + struct.pack('<I', 0) + bytes(extra))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--period', type=float, default=None, help='bandperiode in sensorrijen')
    ap.add_argument('--harmonics', type=int, default=16)
    ap.add_argument('--strips', type=int, default=8)
    ap.add_argument('--ywin', type=int, default=0, help='aantal rij-vensters (0 = automatisch, 1 = alleen stroken)')
    ap.add_argument('--strength', type=float, default=1.0)
    ap.add_argument('--passes', type=int, default=1, help='aantal passes; 2 = probeer ook een tweede lampfrequentie (standaard 1)')
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--dng', dest='dng', action='store_true', default=True)
    ap.add_argument('--no-dng', dest='dng', action='store_false')
    ap.add_argument('--tiff', action='store_true')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true', help='ook schrijven als de zelfcontrole twijfelt')
    ap.add_argument('--no-compress', action='store_true', help='DNG ongecomprimeerd schrijven (groter, iets sneller)')
    ap.add_argument('--check', action='store_true', help='alleen meten en rapporteren, niets schrijven')
    ap.add_argument('--jobs', type=int, default=0, help="aantal foto's tegelijk (0 = automatisch)")
    ap.add_argument('--verbose', action='store_true', help='toon de schatting per iteratie')
    args = ap.parse_args()
    files = [f for f in args.files if os.path.isfile(f)]
    if not files:
        print("geen bestanden gevonden")
        return
    jobs = args.jobs if args.jobs > 0 else max(1, min(8, (os.cpu_count() or 2) - 2))
    jobs = min(jobs, len(files))
    t0 = time.time()

    def run_batch(flist, known):
        out = []
        if jobs == 1:
            for f in flist:
                out.append(process(f, args, known_periods=known))
        else:
            import multiprocessing as mp
            with mp.get_context('spawn').Pool(min(jobs, len(flist)), initializer=_worker_init) as pool:
                for text, res in pool.imap_unordered(_worker, [(f, args, known) for f in flist]):
                    print(text, end='')
                    out.append(res)
        return [r for r in out if r]

    if jobs > 1:
        print(f"{len(files)} bestanden, {jobs} tegelijk")
    results = run_batch(files, None)
    # Stap 2: shoot-periodes leren uit de zeker gecorrigeerde foto's (bevestigde periodiciteit),
    # en de twijfelgevallen daarmee nog eens proberen. Lampfrequenties zijn constant binnen een shoot.
    if len(files) >= 5 and not args.period:
        sure = [r['period'] for r in results if r['status'].startswith('gecorrigeerd') and r.get('conf') and r['period'] > 0]
        clusters = []
        for P in sorted(sure):
            for c in clusters:
                if abs(P - c['P']) <= 0.03 * c['P']:
                    c['n'] += 1
                    c['P'] = (c['P'] * (c['n'] - 1) + P) / c['n']
                    break
            else:
                clusters.append(dict(P=P, n=1))
        known = [c['P'] for c in clusters if c['n'] >= max(3, 0.05 * len(files))]
        retry = [r['file'] for r in results if r['status'] in ('model past niet', 'onzeker (waarschijnlijk content)',
                                                                 'geen banding (zwak signaal)', 'geen banding')]
        if known and retry:
            byname = {os.path.basename(f): f for f in files}
            print(f"\nshoot-periodes gevonden: {', '.join(f'{k:.0f}' for k in known)} rijen "
                  f"-> {len(retry)} twijfelgevallen opnieuw proberen")
            res2 = run_batch([byname[n] for n in retry], known)
            upd = {r['file']: r for r in res2}
            results = [upd.get(r['file'], r) for r in results]
    dt = time.time() - t0
    if len(files) > 1:
        print(f"\nklaar: {len(files)} bestanden in {dt:.0f} s ({dt / len(files):.1f} s per foto)")
    if args.check or len(results) > 1:
        print("\nsamenvatting:")
        for r in sorted(results, key=lambda r: r['file']):
            b = f"{r['before'] * 100:4.1f}%" if r['before'] is not None else "   - "
            a = f"{r['after'] * 100:4.1f}%" if r['after'] is not None else "   - "
            per = f"{r['period']:6.1f}" if r['period'] else "   -  "
            print(f"  {r['file']:28s} periode {per}  banding {b} -> {a}  {r['status']}")


def _worker_init():
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    import warnings
    warnings.filterwarnings('ignore')


def _worker(job):
    import io
    import contextlib
    path, args, known = job
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            res = process(path, args, known_periods=known)
        except Exception as e:
            import traceback
            print(f"  FOUT bij {os.path.basename(path)}: {e}")
            traceback.print_exc()
            res = dict(file=os.path.basename(path), period=0, power=0, before=None, after=None, status='fout')
    return buf.getvalue(), res


if __name__ == '__main__':
    main()
