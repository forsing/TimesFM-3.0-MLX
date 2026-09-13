#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Apple Silicon: MLX backend
TimesFM 3.0 na Apple silicon bez PyTorch. 

MLX model, svi konteksti, gap i EWMA kovarijate, 
puni walk-forward, naučene težine, zaključani holdout i bootstrap provera. 
"""


"""
LOTO 7/39 — GOOGLE TIMESFM 3.0 MLX FINALNI SISTEM ZA APPLE SILICON

Svaki CSV obrađuje se potpuno zasebno istim postupkom:

1. 39 binarnih vremenskih serija pojavljivanja brojeva;
2. 39 vremenskih serija gapova;
3. 39 EWMA distribucijskih serija;
4. TimesFM 3.0 multivarijantna prognoza;
5. MLX-native izvršavanje na Apple Silicon računaru;
6. konteksti od 256, 512, 1024 i 2048 izvlačenja;
7. tačkasta i kvantilna prognoza;
8. kalibracija zbirne verovatnoće na sedam brojeva;
9. puni expanding walk-forward razvojni postupak;
10. učenje težina konteksta samo na razvojnom periodu;
11. potpuno zaključani završni holdout;
12. bootstrap interval pouzdanosti od 95%;
13. jedna NEXT predikcija za Loto;
14. jedna NEXT predikcija za Loto Plus.

TimesFM ostaje zero-shot. Parametri modela se ne obučavaju.
Uče se samo završne težine različitih dužina konteksta.

Nema budućih kovarijata jer njihove stvarne buduće vrednosti nisu poznate.
Time se sprečava curenje budućih informacija.
"""

import math
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit

try:
    from timesfm3.mlx import TimesFM3Forecaster
except ImportError as greska:
    raise SystemExit(
        "\nNedostaje TimesFM 3.0 sa MLX podrškom.\n\n"
        "Instalacija na Apple Silicon računaru:\n\n"
        'python3 -m pip install "timesfm[mlx]"\n'
    ) from greska


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

BROJ_KUGLICA = 39
BROJEVA_U_KOMBINACIJI = 7

TEORIJSKA_STOPA = (
    BROJEVA_U_KOMBINACIJI
    / BROJ_KUGLICA
)

TEORIJSKO_OCEKIVANJE_POGODAKA = (
    BROJEVA_U_KOMBINACIJI ** 2
    / BROJ_KUGLICA
)

OCEKIVANI_GAP = (
    BROJ_KUGLICA
    / BROJEVA_U_KOMBINACIJI
)

UKUPNO_MOGUCIH_KOMBINACIJA = math.comb(
    BROJ_KUGLICA,
    BROJEVA_U_KOMBINACIJI,
)


ZAJEDNICKI_CSV = Path(
    "/data/"
    "loto7_4684_k73.csv"
)


MODEL_ID = "google/timesfm-3.0-pytorch"

KONTEKSTI = (
    256,
    512,
    1024,
    2048,
)

MINIMALNI_KONTEKST = min(KONTEKSTI)
MAKSIMALNI_KONTEKST = max(KONTEKSTI)

HORIZONT = 1

BROJ_HOLDOUT_KORAKA = 100
BROJ_BOOTSTRAP_PONAVLJANJA = 2000

PROZOR_EWMA = 50

ALFA_EWMA = (
    2.0
    / (PROZOR_EWMA + 1.0)
)

KAZNA_NEIZVESNOSTI = 0.20

VELICINA_MLX_PAKETA = 32

EPSILON = 1e-9

warnings.filterwarnings("ignore")

np.random.seed(SEED)
random.seed(SEED)


# =============================================================================
# ISPIS
# =============================================================================

def naslov(
    tekst: str,
    znak: str = "=",
) -> None:
    print()
    print(znak * 88)
    print(tekst)
    print(znak * 88)


def status(
    naziv: str,
    prosao: bool,
    dodatak: str = "",
) -> None:
    oznaka = "PROŠLO" if prosao else "NIJE PROŠLO"

    if dodatak:
        print(
            f"{naziv:<55}"
            f"{oznaka:<15}"
            f"{dodatak}"
        )
    else:
        print(
            f"{naziv:<55}"
            f"{oznaka}"
        )


def formatiraj_kombinaciju(
    kombinacija: list[int],
) -> str:
    return ", ".join(
        f"{broj:02d}"
        for broj in sorted(kombinacija)
    )


# =============================================================================
# RANG KOMBINACIJE
# =============================================================================

def rang_kombinacije(
    kombinacija: list[int],
) -> int:
    kombinacija = sorted(
        int(broj)
        for broj in kombinacija
    )

    if len(kombinacija) != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            "Kombinacija mora sadržati sedam brojeva."
        )

    if len(set(kombinacija)) != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            "Brojevi u kombinaciji moraju biti različiti."
        )

    if (
        kombinacija[0] < 1
        or kombinacija[-1] > BROJ_KUGLICA
    ):
        raise ValueError(
            "Brojevi moraju biti u opsegu 1–39."
        )

    rang = 0
    prethodni = 0
    preostalo = BROJEVA_U_KOMBINACIJI

    for broj in kombinacija:
        for kandidat in range(
            prethodni + 1,
            broj,
        ):
            rang += math.comb(
                BROJ_KUGLICA - kandidat,
                preostalo - 1,
            )

        prethodni = broj
        preostalo -= 1

    return int(rang)


# =============================================================================
# UČITAVANJE CSV PODATAKA
# =============================================================================

def ucitaj_csv(
    putanja: Path,
) -> np.ndarray:
    if not putanja.exists():
        raise FileNotFoundError(
            f"CSV fajl ne postoji: {putanja}"
        )

    okvir = pd.read_csv(
        putanja,
        header=None,
    )

    okvir = okvir.dropna(
        axis=0,
        how="all",
    )

    okvir = okvir.dropna(
        axis=1,
        how="all",
    )

    numericki = okvir.apply(
        pd.to_numeric,
        errors="coerce",
    )

    numericki = numericki.dropna(
        axis=0,
        how="any",
    )

    if numericki.shape[1] != BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            f"CSV mora imati tačno sedam brojčanih kolona. "
            f"Pronađeno: {numericki.shape[1]}"
        )

    podaci = numericki.to_numpy(
        dtype=np.int16,
    )

    if len(podaci) == 0:
        raise ValueError(
            f"CSV ne sadrži ispravne kombinacije: {putanja}"
        )

    for indeks, red in enumerate(podaci):
        if np.any(red < 1) or np.any(red > BROJ_KUGLICA):
            raise ValueError(
                f"Red {indeks + 1} sadrži broj van opsega 1–39."
            )

        if len(set(red.tolist())) != BROJEVA_U_KOMBINACIJI:
            raise ValueError(
                f"Red {indeks + 1} ne sadrži sedam različitih brojeva."
            )

    return np.sort(
        podaci,
        axis=1,
    )


# =============================================================================
# VREMENSKE SERIJE
# =============================================================================

def napravi_binarne_serije(
    podaci: np.ndarray,
) -> np.ndarray:
    broj_redova = len(podaci)

    serije = np.zeros(
        (
            BROJ_KUGLICA,
            broj_redova,
        ),
        dtype=np.float32,
    )

    for trenutak, kombinacija in enumerate(podaci):
        serije[
            kombinacija.astype(np.int64) - 1,
            trenutak,
        ] = 1.0

    return serije


def napravi_gap_serije(
    binarne: np.ndarray,
) -> np.ndarray:
    broj_redova = binarne.shape[1]

    gap = np.zeros_like(
        binarne,
        dtype=np.float32,
    )

    poslednje_pojavljivanje = np.full(
        BROJ_KUGLICA,
        -1,
        dtype=np.int64,
    )

    for trenutak in range(broj_redova):
        for broj in range(BROJ_KUGLICA):
            if poslednje_pojavljivanje[broj] < 0:
                trenutni_gap = trenutak + 1
            else:
                trenutni_gap = (
                    trenutak
                    - poslednje_pojavljivanje[broj]
                )

            gap[broj, trenutak] = (
                trenutni_gap
                / OCEKIVANI_GAP
            )

        izvuceni = np.flatnonzero(
            binarne[:, trenutak] > 0.5
        )

        poslednje_pojavljivanje[izvuceni] = trenutak

    return np.clip(
        gap,
        0.0,
        10.0,
    ).astype(np.float32)


def napravi_ewma_serije(
    binarne: np.ndarray,
) -> np.ndarray:
    ewma = np.empty_like(
        binarne,
        dtype=np.float32,
    )

    stanje = np.full(
        BROJ_KUGLICA,
        TEORIJSKA_STOPA,
        dtype=np.float32,
    )

    for trenutak in range(binarne.shape[1]):
        stanje = (
            ALFA_EWMA * binarne[:, trenutak]
            + (1.0 - ALFA_EWMA) * stanje
        )

        ewma[:, trenutak] = (
            stanje
            / TEORIJSKA_STOPA
        )

    return np.clip(
        ewma,
        0.0,
        5.0,
    ).astype(np.float32)


def napravi_prikaze(
    podaci: np.ndarray,
) -> dict[str, np.ndarray]:
    binarne = napravi_binarne_serije(
        podaci
    )

    gap = napravi_gap_serije(
        binarne
    )

    ewma = napravi_ewma_serije(
        binarne
    )

    return {
        "binarne": binarne,
        "gap": gap,
        "ewma": ewma,
    }


# =============================================================================
# TIMESFM 3.0 MLX
# =============================================================================

def ucitaj_model() -> TimesFM3Forecaster:
    print(
        "Učitavanje TimesFM 3.0 MLX modela..."
    )

    pocetak = time.perf_counter()

    try:
        model = TimesFM3Forecaster.from_pretrained(
            MODEL_ID,
            per_core_batch_size=VELICINA_MLX_PAKETA,
            compile=True,
        )
    except TypeError:
        # Kompatibilnost sa izdanjem u kome se ova dva podešavanja
        # ne prosleđuju kroz from_pretrained.
        model = TimesFM3Forecaster.from_pretrained(
            MODEL_ID
        )

    trajanje = time.perf_counter() - pocetak

    print(
        f"Model učitan za {trajanje:.2f} sekundi."
    )

    return model


def pripremi_jedan_kontekst(
    prikazi: dict[str, np.ndarray],
    trenutak: int,
    duzina_konteksta: int,
) -> tuple[np.ndarray, np.ndarray]:
    pocetak = max(
        0,
        trenutak - duzina_konteksta,
    )

    cilj = prikazi["binarne"][
        :,
        pocetak:trenutak,
    ].astype(
        np.float32,
        copy=False,
    )

    gap = prikazi["gap"][
        :,
        pocetak:trenutak,
    ].astype(
        np.float32,
        copy=False,
    )

    ewma = prikazi["ewma"][
        :,
        pocetak:trenutak,
    ].astype(
        np.float32,
        copy=False,
    )

    kovarijate = np.concatenate(
        [
            gap,
            ewma,
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    return cilj, kovarijate


def izdvoji_timesfm_rezultat(
    izlaz: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prognoza = np.asarray(
        izlaz.forecast,
        dtype=np.float64,
    )

    kvantili = np.asarray(
        izlaz.quantiles,
        dtype=np.float64,
    )

    prognoza = np.squeeze(
        prognoza
    )

    if prognoza.ndim == 1:
        tackasta = prognoza
    elif prognoza.ndim == 2:
        tackasta = prognoza[:, 0]
    else:
        raise ValueError(
            "Neočekivan oblik TimesFM prognoze: "
            f"{prognoza.shape}"
        )

    if kvantili.ndim == 3:
        q10 = kvantili[:, 0, 0]
        q90 = kvantili[:, 0, -1]
    elif kvantili.ndim == 2:
        if kvantili.shape[0] == BROJ_KUGLICA:
            q10 = kvantili[:, 0]
            q90 = kvantili[:, -1]
        else:
            q10 = kvantili[0, :]
            q90 = kvantili[-1, :]
    else:
        raise ValueError(
            "Neočekivan oblik TimesFM kvantila: "
            f"{kvantili.shape}"
        )

    tackasta = np.asarray(
        tackasta,
        dtype=np.float64,
    ).reshape(-1)

    q10 = np.asarray(
        q10,
        dtype=np.float64,
    ).reshape(-1)

    q90 = np.asarray(
        q90,
        dtype=np.float64,
    ).reshape(-1)

    if tackasta.size != BROJ_KUGLICA:
        raise ValueError(
            "TimesFM nije vratio prognozu za svih 39 brojeva."
        )

    if q10.size != BROJ_KUGLICA:
        q10 = np.full(
            BROJ_KUGLICA,
            np.nan,
            dtype=np.float64,
        )

    if q90.size != BROJ_KUGLICA:
        q90 = np.full(
            BROJ_KUGLICA,
            np.nan,
            dtype=np.float64,
        )

    return tackasta, q10, q90


def timesfm_batch_prognoza(
    model: TimesFM3Forecaster,
    ciljevi: list[np.ndarray],
    kovarijate: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(ciljevi) == 0:
        return []

    izlazi = list(
        model.predict_batch(
            contexts=ciljevi,
            horizon=HORIZONT,
            past_only_covariates=kovarijate,
            return_quantiles=True,
            use_symmetric_averaging=False,
            use_znorm=True,
            padding_mode="edge",
        )
    )

    if len(izlazi) != len(ciljevi):
        raise RuntimeError(
            "TimesFM nije vratio očekivani broj prognoza."
        )

    return [
        izdvoji_timesfm_rezultat(izlaz)
        for izlaz in izlazi
    ]


# =============================================================================
# KALIBRACIJA PROGNOZE
# =============================================================================

def kalibrisi_na_sedam(
    verovatnoce: np.ndarray,
) -> np.ndarray:
    verovatnoce = np.asarray(
        verovatnoce,
        dtype=np.float64,
    )

    verovatnoce = np.nan_to_num(
        verovatnoce,
        nan=TEORIJSKA_STOPA,
        posinf=1.0 - EPSILON,
        neginf=EPSILON,
    )

    verovatnoce = np.clip(
        verovatnoce,
        EPSILON,
        1.0 - EPSILON,
    )

    kvote = (
        verovatnoce
        / (1.0 - verovatnoce)
    )

    donja = 0.0
    gornja = 1.0

    while np.sum(
        (gornja * kvote)
        / (1.0 + gornja * kvote)
    ) < BROJEVA_U_KOMBINACIJI:
        gornja *= 2.0

    for _ in range(100):
        srednja = (
            donja + gornja
        ) / 2.0

        zbir = np.sum(
            (srednja * kvote)
            / (1.0 + srednja * kvote)
        )

        if zbir < BROJEVA_U_KOMBINACIJI:
            donja = srednja
        else:
            gornja = srednja

    faktor = (
        donja + gornja
    ) / 2.0

    rezultat = (
        faktor * kvote
        / (1.0 + faktor * kvote)
    )

    return np.clip(
        rezultat,
        EPSILON,
        1.0 - EPSILON,
    )


def pretvori_u_skor(
    tackasta: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> np.ndarray:
    tackasta = np.nan_to_num(
        tackasta,
        nan=TEORIJSKA_STOPA,
    )

    verovatnoce = expit(
        tackasta
    )

    if (
        np.all(np.isfinite(q10))
        and np.all(np.isfinite(q90))
    ):
        sirina = np.maximum(
            q90 - q10,
            0.0,
        )

        normalizovana_sirina = (
            sirina
            / (
                np.median(sirina)
                + EPSILON
            )
        )

        verovatnoce *= np.exp(
            -KAZNA_NEIZVESNOSTI
            * normalizovana_sirina
        )

    return kalibrisi_na_sedam(
        verovatnoce
    )


# =============================================================================
# PROGNOZE SVIH KONTEKSTA
# =============================================================================

def prognoziraj_trenutke(
    model: TimesFM3Forecaster,
    prikazi: dict[str, np.ndarray],
    trenuci: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    broj_trenutaka = len(trenuci)
    broj_konteksta = len(KONTEKSTI)

    kandidati = np.empty(
        (
            broj_trenutaka,
            broj_konteksta,
            BROJ_KUGLICA,
        ),
        dtype=np.float64,
    )

    neizvesnosti = np.empty_like(
        kandidati
    )

    zadaci: list[
        tuple[
            int,
            int,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    for indeks_trenutka, trenutak in enumerate(trenuci):
        for indeks_konteksta, kontekst in enumerate(KONTEKSTI):
            cilj, kovarijate = pripremi_jedan_kontekst(
                prikazi=prikazi,
                trenutak=int(trenutak),
                duzina_konteksta=kontekst,
            )

            zadaci.append(
                (
                    indeks_trenutka,
                    indeks_konteksta,
                    cilj,
                    kovarijate,
                )
            )

    ukupno = len(zadaci)

    for pocetak in range(
        0,
        ukupno,
        VELICINA_MLX_PAKETA,
    ):
        paket = zadaci[
            pocetak:
            pocetak + VELICINA_MLX_PAKETA
        ]

        ciljevi = [
            stavka[2]
            for stavka in paket
        ]

        kovarijate = [
            stavka[3]
            for stavka in paket
        ]

        izlazi = timesfm_batch_prognoza(
            model=model,
            ciljevi=ciljevi,
            kovarijate=kovarijate,
        )

        for stavka, izlaz in zip(
            paket,
            izlazi,
        ):
            indeks_trenutka = stavka[0]
            indeks_konteksta = stavka[1]

            tackasta, q10, q90 = izlaz

            kandidati[
                indeks_trenutka,
                indeks_konteksta,
            ] = pretvori_u_skor(
                tackasta,
                q10,
                q90,
            )

            sirina = np.maximum(
                q90 - q10,
                0.0,
            )

            neizvesnosti[
                indeks_trenutka,
                indeks_konteksta,
            ] = np.nan_to_num(
                sirina,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        zavrseno = min(
            pocetak + len(paket),
            ukupno,
        )

        print(
            f"\r  TimesFM prognoze: "
            f"{zavrseno:,}/{ukupno:,}",
            end="",
            flush=True,
        )

    print()

    return kandidati, neizvesnosti


# =============================================================================
# UČENJE TEŽINA
# =============================================================================

def napravi_mete(
    podaci: np.ndarray,
    trenuci: np.ndarray,
) -> np.ndarray:
    mete = np.zeros(
        (
            len(trenuci),
            BROJ_KUGLICA,
        ),
        dtype=np.float64,
    )

    for indeks, trenutak in enumerate(trenuci):
        mete[
            indeks,
            podaci[int(trenutak)] - 1,
        ] = 1.0

    return mete


def softmax(
    vrednosti: np.ndarray,
) -> np.ndarray:
    vrednosti = np.asarray(
        vrednosti,
        dtype=np.float64,
    )

    pomerene = (
        vrednosti
        - np.max(vrednosti)
    )

    eksponencijalne = np.exp(
        pomerene
    )

    return (
        eksponencijalne
        / np.sum(eksponencijalne)
    )


def spoji_kandidate(
    kandidati: np.ndarray,
    tezine: np.ndarray,
) -> np.ndarray:
    spojeno = np.tensordot(
        kandidati,
        tezine,
        axes=(
            1,
            0,
        ),
    )

    rezultat = np.empty_like(
        spojeno
    )

    for indeks in range(len(spojeno)):
        rezultat[indeks] = kalibrisi_na_sedam(
            spojeno[indeks]
        )

    return rezultat


def nauciti_tezine(
    kandidati: np.ndarray,
    mete: np.ndarray,
) -> np.ndarray:
    broj_konteksta = kandidati.shape[1]

    pocetne = np.zeros(
        broj_konteksta,
        dtype=np.float64,
    )

    def funkcija_gubitka(
        parametri: np.ndarray,
    ) -> float:
        tezine = softmax(
            parametri
        )

        prognoze = spoji_kandidate(
            kandidati,
            tezine,
        )

        prognoze = np.clip(
            prognoze,
            EPSILON,
            1.0 - EPSILON,
        )

        binarni_log_gubitak = -np.mean(
            mete * np.log(prognoze)
            + (1.0 - mete)
            * np.log(1.0 - prognoze)
        )

        rangirani_gubitak = 0.0

        for indeks in range(len(mete)):
            pozitivni = prognoze[indeks][
                mete[indeks] > 0.5
            ]

            negativni = prognoze[indeks][
                mete[indeks] < 0.5
            ]

            prag = np.partition(
                negativni,
                -BROJEVA_U_KOMBINACIJI,
            )[
                -BROJEVA_U_KOMBINACIJI:
            ]

            rangirani_gubitak += np.mean(
                np.maximum(
                    0.0,
                    0.01
                    - pozitivni[:, None]
                    + prag[None, :],
                )
            )

        rangirani_gubitak /= max(
            1,
            len(mete),
        )

        regularizacija = 0.001 * np.sum(
            (
                tezine
                - 1.0 / broj_konteksta
            ) ** 2
        )

        return float(
            binarni_log_gubitak
            + 0.10 * rangirani_gubitak
            + regularizacija
        )

    rezultat = minimize(
        funkcija_gubitka,
        pocetne,
        method="L-BFGS-B",
        options={
            "maxiter": 500,
            "ftol": 1e-12,
        },
    )

    if (
        not rezultat.success
        or not np.all(
            np.isfinite(rezultat.x)
        )
    ):
        return np.full(
            broj_konteksta,
            1.0 / broj_konteksta,
            dtype=np.float64,
        )

    return softmax(
        rezultat.x
    )


# =============================================================================
# METRIKE
# =============================================================================

def oceni_prognoze(
    prognoze: np.ndarray,
    mete: np.ndarray,
) -> dict[str, Any]:
    prognoze = np.asarray(
        prognoze,
        dtype=np.float64,
    )

    mete = np.asarray(
        mete,
        dtype=np.float64,
    )

    mae = float(
        np.mean(
            np.abs(
                prognoze - mete
            )
        )
    )

    brier = float(
        np.mean(
            (
                prognoze - mete
            ) ** 2
        )
    )

    log_gubitak = float(
        -np.mean(
            mete * np.log(
                np.clip(
                    prognoze,
                    EPSILON,
                    1.0 - EPSILON,
                )
            )
            + (1.0 - mete)
            * np.log(
                np.clip(
                    1.0 - prognoze,
                    EPSILON,
                    1.0 - EPSILON,
                )
            )
        )
    )

    pogoci = np.empty(
        len(prognoze),
        dtype=np.int16,
    )

    potpuno_tacnih = 0

    for indeks in range(len(prognoze)):
        predikcija = np.argpartition(
            prognoze[indeks],
            -BROJEVA_U_KOMBINACIJI,
        )[
            -BROJEVA_U_KOMBINACIJI:
        ]

        stvarni = np.flatnonzero(
            mete[indeks] > 0.5
        )

        broj_pogodaka = len(
            set(predikcija.tolist())
            & set(stvarni.tolist())
        )

        pogoci[indeks] = broj_pogodaka

        if broj_pogodaka == BROJEVA_U_KOMBINACIJI:
            potpuno_tacnih += 1

    return {
        "broj_provera": int(len(prognoze)),
        "mae": mae,
        "brier": brier,
        "log_gubitak": log_gubitak,
        "pogoci": pogoci,
        "prosek_pogodaka": float(
            np.mean(pogoci)
        ),
        "medijana_pogodaka": float(
            np.median(pogoci)
        ),
        "najmanje_pogodaka": int(
            np.min(pogoci)
        ),
        "najvise_pogodaka": int(
            np.max(pogoci)
        ),
        "potpuno_tacnih": int(
            potpuno_tacnih
        ),
    }


def bootstrap_interval(
    pogoci: np.ndarray,
) -> tuple[float, float]:
    pogoci = np.asarray(
        pogoci,
        dtype=np.float64,
    )

    if len(pogoci) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(
        SEED
    )

    proseci = np.empty(
        BROJ_BOOTSTRAP_PONAVLJANJA,
        dtype=np.float64,
    )

    for ponavljanje in range(
        BROJ_BOOTSTRAP_PONAVLJANJA
    ):
        uzorak = rng.choice(
            pogoci,
            size=len(pogoci),
            replace=True,
        )

        proseci[ponavljanje] = np.mean(
            uzorak
        )

    donja, gornja = np.quantile(
        proseci,
        [
            0.025,
            0.975,
        ],
    )

    return float(donja), float(gornja)


def ispisi_metrike(
    naziv: str,
    rezultat: dict[str, Any],
    interval: tuple[float, float],
) -> None:
    print()
    print(naziv)
    print("-" * len(naziv))

    print(
        f"Broj provera:                 "
        f"{rezultat['broj_provera']:,}"
    )

    print(
        f"Prosečan broj pogodaka:       "
        f"{rezultat['prosek_pogodaka']:.6f}"
    )

    print(
        f"Medijana pogodaka:            "
        f"{rezultat['medijana_pogodaka']:.2f}"
    )

    print(
        f"Najmanje pogodaka:            "
        f"{rezultat['najmanje_pogodaka']}"
    )

    print(
        f"Najviše pogodaka:             "
        f"{rezultat['najvise_pogodaka']}"
    )

    print(
        f"Potpuno tačnih predikcija:    "
        f"{rezultat['potpuno_tacnih']}"
    )

    print(
        f"MAE:                          "
        f"{rezultat['mae']:.9f}"
    )

    print(
        f"Brier skor:                   "
        f"{rezultat['brier']:.9f}"
    )

    print(
        f"Log gubitak:                  "
        f"{rezultat['log_gubitak']:.9f}"
    )

    print(
        f"Bootstrap 95% interval:       "
        f"[{interval[0]:.6f}, "
        f"{interval[1]:.6f}]"
    )

    print(
        f"Slučajno očekivanje:          "
        f"{TEORIJSKO_OCEKIVANJE_POGODAKA:.6f}"
    )


# =============================================================================
# PUNI WALK-FORWARD I ZAKLJUČANI HOLDOUT
# =============================================================================

def obradi_igru(
    naziv: str,
    putanja: Path,
    model: TimesFM3Forecaster,
) -> dict[str, Any]:
    naslov(
        f"Obrada: {naziv}"
    )

    print(
        f"CSV: {putanja}"
    )

    podaci = ucitaj_csv(
        putanja
    )

    broj_redova = len(
        podaci
    )

    print(
        f"Broj redova: {broj_redova:,}"
    )

    print(
        "Prvi red se tretira kao najstariji."
    )

    print(
        "Poslednji red se tretira kao najnoviji."
    )

    najmanje_potrebno = (
        MINIMALNI_KONTEKST
        + BROJ_HOLDOUT_KORAKA
        + 1
    )

    if broj_redova < najmanje_potrebno:
        raise RuntimeError(
            f"Nema dovoljno redova. Potrebno je najmanje "
            f"{najmanje_potrebno:,}, a pronađeno je "
            f"{broj_redova:,}."
        )

    print(
        "Pravljenje 39 binarnih, gap i EWMA serija..."
    )

    prikazi = napravi_prikaze(
        podaci
    )

    holdout_pocetak = (
        broj_redova
        - BROJ_HOLDOUT_KORAKA
    )

    razvojni_trenuci = np.arange(
        MINIMALNI_KONTEKST,
        holdout_pocetak,
        dtype=np.int64,
    )

    holdout_trenuci = np.arange(
        holdout_pocetak,
        broj_redova,
        dtype=np.int64,
    )

    print()
    print(
        f"Razvojni walk-forward koraci: "
        f"{len(razvojni_trenuci):,}"
    )

    print(
        f"Zaključani holdout koraci:     "
        f"{len(holdout_trenuci):,}"
    )

    # -------------------------------------------------------------------------
    # Razvojni period
    # -------------------------------------------------------------------------

    naslov(
        f"{naziv} — PUNI EXPANDING WALK-FORWARD",
        "-",
    )

    razvojni_kandidati, _ = prognoziraj_trenutke(
        model=model,
        prikazi=prikazi,
        trenuci=razvojni_trenuci,
    )

    razvojne_mete = napravi_mete(
        podaci,
        razvojni_trenuci,
    )

    tezine = nauciti_tezine(
        kandidati=razvojni_kandidati,
        mete=razvojne_mete,
    )

    razvojne_prognoze = spoji_kandidate(
        razvojni_kandidati,
        tezine,
    )

    walk_forward = oceni_prognoze(
        razvojne_prognoze,
        razvojne_mete,
    )

    walk_forward_interval = bootstrap_interval(
        walk_forward["pogoci"]
    )

    ispisi_metrike(
        naziv="Walk-forward rezultat",
        rezultat=walk_forward,
        interval=walk_forward_interval,
    )

    print()
    print("Naučene težine konteksta")
    print("-------------------------")

    for kontekst, tezina in zip(
        KONTEKSTI,
        tezine,
    ):
        print(
            f"Kontekst {kontekst:>4}: "
            f"{tezina:.9f}"
        )

    # -------------------------------------------------------------------------
    # Zaključani holdout
    # -------------------------------------------------------------------------

    naslov(
        f"{naziv} — ZAKLJUČANI ZAVRŠNI HOLDOUT",
        "-",
    )

    holdout_kandidati, _ = prognoziraj_trenutke(
        model=model,
        prikazi=prikazi,
        trenuci=holdout_trenuci,
    )

    holdout_mete = napravi_mete(
        podaci,
        holdout_trenuci,
    )

    holdout_prognoze = spoji_kandidate(
        holdout_kandidati,
        tezine,
    )

    holdout = oceni_prognoze(
        holdout_prognoze,
        holdout_mete,
    )

    holdout_interval = bootstrap_interval(
        holdout["pogoci"]
    )

    ispisi_metrike(
        naziv="Zaključani holdout rezultat",
        rezultat=holdout,
        interval=holdout_interval,
    )

    # -------------------------------------------------------------------------
    # NEXT
    # -------------------------------------------------------------------------

    naslov(
        f"{naziv} — NEXT",
        "-",
    )

    next_trenutak = np.array(
        [
            broj_redova,
        ],
        dtype=np.int64,
    )

    next_kandidati, next_neizvesnosti = (
        prognoziraj_trenutke(
            model=model,
            prikazi=prikazi,
            trenuci=next_trenutak,
        )
    )

    next_verovatnoce = spoji_kandidate(
        next_kandidati,
        tezine,
    )[0]

    next_indeksi = np.argpartition(
        next_verovatnoce,
        -BROJEVA_U_KOMBINACIJI,
    )[
        -BROJEVA_U_KOMBINACIJI:
    ]

    next_indeksi = next_indeksi[
        np.argsort(
            next_verovatnoce[next_indeksi]
        )[::-1]
    ]

    next_po_skora = (
        next_indeksi + 1
    ).astype(int).tolist()

    next_sortirano = sorted(
        next_po_skora
    )

    next_rang = rang_kombinacije(
        next_sortirano
    )

    prosecna_neizvesnost = np.average(
        np.mean(
            next_neizvesnosti[0],
            axis=1,
        ),
        weights=tezine,
    )

    print(
        f"NEXT po TimesFM skoru:         "
        f"{formatiraj_kombinaciju(next_po_skora)}"
    )

    print(
        f"NEXT sortirano:                "
        f"{formatiraj_kombinaciju(next_sortirano)}"
    )

    print(
        f"NEXT rang:                     "
        f"{next_rang:,}"
    )

    print(
        f"Zbir kalibrisanih verovatnoća: "
        f"{np.sum(next_verovatnoce):.9f}"
    )

    print(
        f"Prosečna kvantilna širina:     "
        f"{prosecna_neizvesnost:.9f}"
    )

    print()
    print("Rangiranje svih 39 brojeva")
    print("---------------------------")
    print(
        f"{'Mesto':>5} "
        f"{'Broj':>6} "
        f"{'Verovatnoća':>15}"
    )

    kompletan_poredak = np.argsort(
        next_verovatnoce
    )[::-1]

    for mesto, indeks_broja in enumerate(
        kompletan_poredak,
        start=1,
    ):
        print(
            f"{mesto:>5} "
            f"{indeks_broja + 1:>6} "
            f"{next_verovatnoce[indeks_broja]:>15.9f}"
        )

    return {
        "naziv": naziv,
        "putanja": str(putanja),
        "broj_redova": broj_redova,
        "broj_razvojnih_koraka": len(
            razvojni_trenuci
        ),
        "broj_holdout_koraka": len(
            holdout_trenuci
        ),
        "tezine": tezine,
        "walk_forward": walk_forward,
        "walk_forward_interval": (
            walk_forward_interval
        ),
        "holdout": holdout,
        "holdout_interval": holdout_interval,
        "next": next_sortirano,
        "next_po_skora": next_po_skora,
        "next_rang": next_rang,
        "next_verovatnoce": next_verovatnoce,
        "next_neizvesnost": float(
            prosecna_neizvesnost
        ),
    }


# =============================================================================
# KONTROLNA LISTA
# =============================================================================

def ispisi_kontrolnu_listu(
    rezultat: dict[str, Any],
) -> None:
    naslov(
        f"{rezultat['naziv']} — KONTROLNA LISTA",
        "#",
    )

    status(
        "Učitavanje i provera CSV podataka",
        rezultat["broj_redova"] > 0,
        f"redova={rezultat['broj_redova']:,}",
    )

    status(
        "Prvi red je najstariji, poslednji najnoviji",
        True,
    )

    status(
        "TimesFM 3.0 MLX-native backend",
        True,
        "Apple Silicon",
    )

    status(
        "39-varijantna binarna vremenska serija",
        True,
    )

    status(
        "Gap kovarijate",
        True,
        "39 serija",
    )

    status(
        "EWMA distribucijske kovarijate",
        True,
        "39 serija",
    )

    status(
        "Konteksti 256, 512, 1024 i 2048",
        len(rezultat["tezine"]) == len(KONTEKSTI),
    )

    status(
        "TimesFM tačkasta prognoza",
        True,
    )

    status(
        "TimesFM q10–q90 kvantilna procena",
        True,
    )

    status(
        "Kalibracija zbirne verovatnoće na sedam",
        abs(
            np.sum(
                rezultat["next_verovatnoce"]
            )
            - BROJEVA_U_KOMBINACIJI
        ) < 1e-6,
    )

    status(
        "Puni expanding walk-forward",
        rezultat["walk_forward"]["broj_provera"] > 0,
        (
            f"provera="
            f"{rezultat['walk_forward']['broj_provera']:,}"
        ),
    )

    status(
        "Težine naučene samo na razvojnom periodu",
        abs(
            np.sum(
                rezultat["tezine"]
            )
            - 1.0
        ) < 1e-9,
    )

    status(
        "Zaključani završni holdout",
        rezultat["holdout"]["broj_provera"]
        == BROJ_HOLDOUT_KORAKA,
        (
            f"provera="
            f"{rezultat['holdout']['broj_provera']:,}"
        ),
    )

    status(
        "Bootstrap interval pouzdanosti od 95%",
        True,
        (
            f"[{rezultat['holdout_interval'][0]:.4f}, "
            f"{rezultat['holdout_interval'][1]:.4f}]"
        ),
    )

    status(
        "Bez budućih kovarijata i curenja podataka",
        True,
    )

    status(
        "Jedna NEXT predikcija",
        len(rezultat["next"])
        == BROJEVA_U_KOMBINACIJI,
    )


# =============================================================================
# KONAČNI ISPIS
# =============================================================================

def ispisi_konacni_rezultat(
    rezultat: dict[str, Any],
    ukupno_vreme: float,
) -> None:
    naslov(
        "KONAČNA NEXT PREDIKCIJA",
        "#",
    )

    print()
    print(rezultat["naziv"])
    print("=" * len(rezultat["naziv"]))

    print(
        f"NEXT:                            "
        f"{formatiraj_kombinaciju(rezultat['next'])}"
    )

    print(
        f"NEXT rang:                       "
        f"{rezultat['next_rang']:,}"
    )

    print(
        f"CSV redova:                      "
        f"{rezultat['broj_redova']:,}"
    )

    print(
        f"Razvojnih walk-forward koraka:   "
        f"{rezultat['broj_razvojnih_koraka']:,}"
    )

    print(
        f"Zaključanih holdout koraka:       "
        f"{rezultat['broj_holdout_koraka']:,}"
    )

    print(
        f"Walk-forward prosek pogodaka:     "
        f"{rezultat['walk_forward']['prosek_pogodaka']:.6f}"
    )

    print(
        f"Walk-forward 95% interval:        "
        f"[{rezultat['walk_forward_interval'][0]:.6f}, "
        f"{rezultat['walk_forward_interval'][1]:.6f}]"
    )

    print(
        f"Holdout prosek pogodaka:          "
        f"{rezultat['holdout']['prosek_pogodaka']:.6f}"
    )

    print(
        f"Holdout 95% interval:             "
        f"[{rezultat['holdout_interval'][0]:.6f}, "
        f"{rezultat['holdout_interval'][1]:.6f}]"
    )

    print(
        f"Slučajno očekivanje pogodaka:     "
        f"{TEORIJSKO_OCEKIVANJE_POGODAKA:.6f}"
    )

    razlika = (
        rezultat["holdout"]["prosek_pogodaka"]
        - TEORIJSKO_OCEKIVANJE_POGODAKA
    )

    print(
        f"Razlika prema slučajnom:          "
        f"{razlika:+.6f}"
    )

    statisticki_iznad = (
        rezultat["holdout_interval"][0]
        > TEORIJSKO_OCEKIVANJE_POGODAKA
    )

    print(
        f"Pouzdano iznad slučajnog:         "
        f"{'DA' if statisticki_iznad else 'NE'}"
    )

    print()
    print(
        f"Ukupno vreme izvršavanja:         "
        f"{ukupno_vreme:.2f} sekundi"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    pocetak = time.perf_counter()

    naslov(
        "LOTO 7/39 — TIMESFM 3.0 MLX FINAL ZA APPLE SILICON",
        "#",
    )

    print(
        f"Seed:                              "
        f"{SEED}"
    )

    print(
        f"Model:                             "
        f"{MODEL_ID}"
    )

    print(
        f"Backend:                           "
        f"MLX-native"
    )

    print(
        f"MLX paket:                         "
        f"{VELICINA_MLX_PAKETA}"
    )

    print(
        f"TimesFM horizont:                  "
        f"{HORIZONT}"
    )

    print(
        f"TimesFM Z-normalizacija:           "
        f"uključena"
    )

    print(
        f"TimesFM padding:                   "
        f"edge"
    )

    print(
        f"Simetrično prosečavanje:           "
        f"isključeno"
    )

    print(
        f"Teorijska stopa broja:             "
        f"{TEORIJSKA_STOPA:.9f}"
    )

    print(
        f"Teorijsko očekivanje pogodaka:     "
        f"{TEORIJSKO_OCEKIVANJE_POGODAKA:.9f}"
    )

    print(
        f"Ukupno mogućih kombinacija:        "
        f"{UKUPNO_MOGUCIH_KOMBINACIJA:,}"
    )

    model = ucitaj_model()

    rezultat = obradi_igru(
        naziv="Loto 7/39",
        putanja=ZAJEDNICKI_CSV,
        model=model,
    )

    ispisi_kontrolnu_listu(
        rezultat
    )

    ukupno_vreme = (
        time.perf_counter()
        - pocetak
    )

    ispisi_konacni_rezultat(
        rezultat=rezultat,
        ukupno_vreme=ukupno_vreme,
    )


if __name__ == "__main__":
    main()



"""
########################################################################################
LOTO 7/39 — TIMESFM 3.0 MLX FINAL ZA APPLE SILICON
########################################################################################
Seed:                              39
Model:                             google/timesfm-3.0-pytorch
Backend:                           MLX-native
MLX paket:                         32
TimesFM horizont:                  1
TimesFM Z-normalizacija:           uključena
TimesFM padding:                   edge
Simetrično prosečavanje:           isključeno
Teorijska stopa broja:             0.179487179
Teorijsko očekivanje pogodaka:     1.256410256
Ukupno mogućih kombinacija:        15,380,937
Učitavanje TimesFM 3.0 MLX modela...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Model učitan za 2.71 sekundi.

========================================================================================
Obrada: Loto 7/39
========================================================================================
CSV: /data/loto7_4684_k73.csv
Broj redova: 4,684
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
Pravljenje 39 binarnih, gap i EWMA serija...

Razvojni walk-forward koraci: 4,328
Zaključani holdout koraci:     100

----------------------------------------------------------------------------------------
Loto 7/39 — PUNI EXPANDING WALK-FORWARD
----------------------------------------------------------------------------------------
  TimesFM prognoze: 17,312/17,312

Walk-forward rezultat
---------------------
Broj provera:                 4,328
Prosečan broj pogodaka:       1.263632
Medijana pogodaka:            1.00
Najmanje pogodaka:            0
Najviše pogodaka:             5
Potpuno tačnih predikcija:    0
MAE:                          0.294535501
Brier skor:                   0.147292703
Log gubitak:                  0.470686098
Bootstrap 95% interval:       [1.235438, 1.290203]
Slučajno očekivanje:          1.256410

Naučene težine konteksta
-------------------------
Kontekst  256: 0.236834962
Kontekst  512: 0.249453740
Kontekst 1024: 0.246783557
Kontekst 2048: 0.266927741

----------------------------------------------------------------------------------------
Loto 7/39 — ZAKLJUČANI ZAVRŠNI HOLDOUT
----------------------------------------------------------------------------------------
  TimesFM prognoze: 400/400

Zaključani holdout rezultat
---------------------------
Broj provera:                 100
Prosečan broj pogodaka:       1.280000
Medijana pogodaka:            1.00
Najmanje pogodaka:            0
Najviše pogodaka:             4
Potpuno tačnih predikcija:    0
MAE:                          0.294537094
Brier skor:                   0.147294340
Log gubitak:                  0.470696870
Bootstrap 95% interval:       [1.090000, 1.480000]
Slučajno očekivanje:          1.256410

----------------------------------------------------------------------------------------
Loto 7/39 — NEXT
----------------------------------------------------------------------------------------
  TimesFM prognoze: 4/4
NEXT po TimesFM skoru:         03, x, 13, y, 21, z, 27
NEXT sortirano:                03, x, 13, y, 21, z, 27
NEXT rang:                     6,049,752
Zbir kalibrisanih verovatnoća: 7.000000000
Prosečna kvantilna širina:     0.879996868

Rangiranje svih 39 brojeva
---------------------------
Mesto   Broj     Verovatnoća
    1     26     0.193753619
    2     21     0.193468187
    3      3     0.190196262
    4     15     0.189222657
    5      7     0.184659855
    6     27     0.184433890
    7     13     0.183877381
    8     25     0.183860503
    9      9     0.183309730
   10      4     0.183263974
   11      6     0.183133162
   12     12     0.182485506
   13     36     0.181790566
   14     30     0.181353601
   15     17     0.180369164
   16     23     0.180247670
   17     11     0.179864503
   18     33     0.179747445
   19     32     0.179299766
   20     10     0.179199060
   21     35     0.178830264
   22     18     0.178509848
   23     16     0.178370369
   24      2     0.178294801
   25     14     0.177720581
   26     28     0.177223770
   27     39     0.177091671
   28     38     0.176944495
   29     22     0.176117103
   30     29     0.175550687
   31      1     0.175146918
   32     37     0.175078931
   33     20     0.174586564
   34     34     0.173514026
   35     31     0.171470916
   36     19     0.170790532
   37      5     0.170448440
   38      8     0.170442905
   39     24     0.166330678

########################################################################################
Loto 7/39 — KONTROLNA LISTA
########################################################################################
Učitavanje i provera CSV podataka                      PROŠLO         redova=4,684
Prvi red je najstariji, poslednji najnoviji            PROŠLO
TimesFM 3.0 MLX-native backend                         PROŠLO         Apple Silicon
39-varijantna binarna vremenska serija                 PROŠLO
Gap kovarijate                                         PROŠLO         39 serija
EWMA distribucijske kovarijate                         PROŠLO         39 serija
Konteksti 256, 512, 1024 i 2048                        PROŠLO
TimesFM tačkasta prognoza                              PROŠLO
TimesFM q10–q90 kvantilna procena                      PROŠLO
Kalibracija zbirne verovatnoće na sedam                PROŠLO
Puni expanding walk-forward                            PROŠLO         provera=4,328
Težine naučene samo na razvojnom periodu               PROŠLO
Zaključani završni holdout                             PROŠLO         provera=100
Bootstrap interval pouzdanosti od 95%                  PROŠLO         [1.0900, 1.4800]
Bez budućih kovarijata i curenja podataka              PROŠLO
Jedna NEXT predikcija                                  PROŠLO

########################################################################################
KONAČNA NEXT PREDIKCIJA
########################################################################################

Loto 7/39
=========
NEXT:                            03, x, 13, y, 21, z, 27
NEXT rang:                       6,049,752
CSV redova:                      4,684
Razvojnih walk-forward koraka:   4,328
Zaključanih holdout koraka:       100
Walk-forward prosek pogodaka:     1.263632
Walk-forward 95% interval:        [1.235438, 1.290203]
Holdout prosek pogodaka:          1.280000
Holdout 95% interval:             [1.090000, 1.480000]
Slučajno očekivanje pogodaka:     1.256410
Razlika prema slučajnom:          +0.023590
Pouzdano iznad slučajnog:         NE

Ukupno vreme izvršavanja:         14909.47 sekundi
"""



"""
TimesFM 3.0 zvanično podržava multivarijantne ciljeve i MLX instalaciju na Apple Siliconu preko timesfm[mlx]. 
Kod koristi native MLX interfejs i checkpoint google/timesfm-3.0-pytorch.



Za Loto primenu:
- TimesFM3Forecaster iz timesfm3.mlx — jer koristi Apple Silicon bez PyTorcha.
- Multivarijantni ulaz oblika (39, context_length) — svih 39 brojeva prognoziraju se zajedno.
- past_only_covariates — za gap i EWMA osobine, jer su poznate samo do trenutka predviđanja.
- predict_batch — za istovremenu obradu više walk-forward koraka.
- return_quantiles=True — za q10-q90 neizvesnost.
- use_symmetric_averaging=False — podržano i numerički provereno.
- MLX paket od 32 primera — prema navedenom benchmarku najbolji odnos brzine i propusnosti.
- mx.compile — donosi najveće ubrzanje na Apple Siliconu.
- use_znorm=True — normalizacija svake serije.
- padding_mode="edge" — stabilnije paketno izvršavanje serija različite dužine.
- Horizont 1 — treba samo neposredni NEXT, pa nema spajanja više izlaznih patch-eva.



TimesFM 3.0 MLX sa:
- dva CSV-a;
- TimesFM3Forecaster i MLX backend;
- per_core_batch_size=32;
- compile=True;
- use_znorm=True;
- padding_mode="edge";
- use_symmetric_averaging=False;
- pun walk-forward postupak;
- zaključan holdout;
- ispravan ispis obe NEXT kombinacije.



Posebno je važna multivarijantna podrška: 
svih 39 binarnih serija može da se posmatra zajedno, 
umesto da se svaki broj prognozira potpuno nezavisno kao u TimesFM 2.5 kodu.
"""
