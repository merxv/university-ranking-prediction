"""Generator of a synthetic ranking table in the raw Times Higher Education format.

The public THE / QS files cannot be redistributed in this repository, so tests,
the CI smoke run and the demo use a synthetic table that reproduces the
structure *and the formatting problems* of the real files: tied ranks ("=45"),
rank bands ("201-250"), thousands separators ("20,152"), percentages ("25%"),
missing values ("-"), inconsistent university names and duplicated rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from urps.data import PILLAR_WEIGHTS, PILLARS

_PREFIXES = ["North", "South", "East", "West", "Central", "Royal", "National", "Technical", "State", "Pacific"]
_PLACES = [
    "Aldermoor",
    "Brightwater",
    "Caldera",
    "Dunmore",
    "Eastvale",
    "Fairhaven",
    "Glenrock",
    "Highgate",
    "Ironbridge",
    "Juniper",
    "Kingsport",
    "Lakeside",
    "Marlow",
    "Northfield",
    "Oakridge",
    "Pinecrest",
    "Queensbury",
    "Riverton",
    "Stonebrook",
    "Thornbury",
    "Upton",
    "Valemont",
    "Westbury",
    "Yarrow",
    "Ashford",
    "Bellmont",
    "Clearwater",
    "Deepwell",
    "Elmstead",
    "Foxhill",
    "Greystone",
    "Hollowmere",
]
_COUNTRIES = ["Kazakhstan", "Germany", "United States", "United Kingdom", "Japan", "Canada", "Australia", "France"]


def _names(n: int, rng: np.random.Generator) -> list[str]:
    pool = (
        [f"University of {p}" for p in _PLACES]
        + [f"{pre} {p} University" for pre in _PREFIXES for p in _PLACES]
        + [f"{p} Institute of Technology" for p in _PLACES]
    )
    if n > len(pool):
        raise ValueError(f"Can generate at most {len(pool)} distinct universities, got {n}")
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def _format_rank(rank: int, tied: bool) -> str:
    if rank <= 200:
        return f"={rank}" if tied else str(rank)
    lower = ((rank - 1) // 50) * 50 + 1
    return f"{lower}-{lower + 49}"


def generate_the_table(
    n_universities: int = 300,
    years: range = range(2016, 2026),
    seed: int = 42,
) -> pd.DataFrame:
    """Return a raw, deliberately messy THE-style ranking table.

    Indicator dynamics: each university has a latent quality level that drifts
    with a persistent trend (momentum); published indicators revert towards
    that level with yearly noise and are clipped to the 0-100 scale, which
    introduces a saturation non-linearity near the top. The table is meant to
    exercise the pipeline, not to reproduce real ranking behaviour.
    """
    rng = np.random.default_rng(seed)
    names = _names(n_universities, rng)
    countries = rng.choice(_COUNTRIES, size=n_universities)
    level = rng.normal(0, 1, n_universities)
    base = {p: np.clip(55 + 15 * level + rng.normal(0, 8, n_universities), 5, 99) for p in PILLARS}
    trend = rng.normal(0, 0.8, n_universities)
    students = rng.integers(3_000, 60_000, n_universities)
    staff_ratio = rng.uniform(6, 30, n_universities)
    intl_share = rng.uniform(2, 45, n_universities)

    rows = []
    current = {p: base[p].copy() for p in PILLARS}
    for year in years:
        for p in PILLARS:
            momentum = trend * (1.2 if p in ("research", "citations") else 0.6)
            # The latent level drifts (momentum); published values revert towards it with noise.
            base[p] = np.clip(base[p] + momentum, 1, 100)
            shock = rng.normal(0, 1.5, n_universities)
            current[p] = np.clip(0.6 * current[p] + 0.4 * base[p] + shock, 1, 100)
        total = np.sum([PILLAR_WEIGHTS[p] * current[p] for p in PILLARS], axis=0)
        order = np.argsort(-total)
        ranks = np.empty(n_universities, dtype=int)
        ranks[order] = np.arange(1, n_universities + 1)
        rounded = np.round(total, 1)
        for i in range(n_universities):
            tied = int(np.sum(rounded == rounded[i])) > 1
            name = names[i]
            roll = rng.random()
            if roll < 0.03:
                name = f"  {name} "  # stray whitespace
            elif roll < 0.05:
                name = name.upper()  # inconsistent capitalisation
            income = f"{current['income'][i]:.1f}" if rng.random() > 0.06 else "-"
            rows.append(
                {
                    "world_rank": _format_rank(int(ranks[i]), tied),
                    "university_name": name,
                    "country": countries[i],
                    "teaching": f"{current['teaching'][i]:.1f}",
                    "international": f"{current['international'][i]:.1f}",
                    "research": f"{current['research'][i]:.1f}",
                    "citations": f"{current['citations'][i]:.1f}",
                    "income": income,
                    "total_score": f"{total[i]:.1f}",
                    "num_students": f"{int(students[i] * rng.uniform(0.97, 1.03)):,}",
                    "student_staff_ratio": f"{staff_ratio[i]:.1f}",
                    "international_students": f"{intl_share[i]:.0f}%",
                    "year": year,
                }
            )
    raw = pd.DataFrame(rows)
    # A few exact duplicates, as produced by careless copy-paste of web tables.
    dupes = raw.sample(n=max(1, len(raw) // 200), random_state=seed)
    return pd.concat([raw, dupes], ignore_index=True)
