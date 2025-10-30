#!/usr/bin/env python3
"""Build a Vertex AI friendly training table from NBA box score data."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


MIN_SEASON_START_YEAR = 2013
DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (4, 15)
EWM_COMBINATION_SPAN = 4
EWM_COMBINATION_WEIGHT = 0.75

# Player aggregation outputs begin with ``ply_``. Allowing that prefix keeps
# roster-composition signals in the rolling feature set without exploding the
# column count with unrelated identifiers (coach IDs, arena numbers, etc.).
PLAYER_NUMERIC_PREFIXES: tuple[str, ...] = ("ply_",)

# Player-level momentum features are excluded from the experimental build to
# honor the user's request for team-level metrics only.

TEAM_EXCLUDED_NUMERIC_FEATURES: set[str] = {
    "coach_id",
    "arena_id",
}

# Only the metrics derived from the supplied CSVs that align with the
# user-provided list are kept for pre-game feature generation. Columns listed
# here are referenced directly from the cleaned team statistics table after all
# necessary opponent joins and derived calculations have been completed.
TEAM_OPTIONAL_DROP_COLUMNS: set[str] = {
    "win",
    "loss",
    "plus_minus_points",
    "opp_pf",
}

TEAM_ALLOWED_BASE_COLUMNS: set[str] = {
    "games_played",
    "win_pct",
    "season_win_pct",
    "days_since_last_game",
    "games_last_7_days",
    "games_last_10_days",
    "games_last_14_days",
    "consecutive_away_games",
    "home_win_pct",
    "away_win_pct",
    "win_loss_streak",
    "home_win_loss_streak",
    "away_win_loss_streak",
    "season_home_games_played_total",
    "season_home_wins_total",
    "season_home_win_pct",
    "season_away_games_played_total",
    "season_away_wins_total",
    "season_away_win_pct",
    "season_home_margin_avg",
    "season_away_margin_avg",
    "season_home_off_rtg_avg",
    "season_home_def_rtg_avg",
    "season_away_off_rtg_avg",
    "season_away_def_rtg_avg",
    "season_home_points_avg",
    "season_home_points_allowed_avg",
    "season_away_points_avg",
    "season_away_points_allowed_avg",
    "elo_rating",
    "minutes",
    "team_score",
    "field_goals_made",
    "field_goals_attempted",
    "field_goals_percentage",
    "three_pointers_made",
    "three_pointers_attempted",
    "three_pointers_percentage",
    "free_throws_made",
    "free_throws_attempted",
    "free_throws_percentage",
    "rebounds_offensive",
    "rebounds_defensive",
    "rebounds_total",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "fouls_personal",
    "ast_ratio",
    "ast_to_ratio",
    "poss",
    "efg",
    "ts_pct",
    "ft_rate",
    "tov_pct",
    "oreb_pct",
    "dreb_pct",
    "reb_pct",
    "off_rtg",
    "def_rtg",
    "net_rtg",
    "pace",
    "opp_efg_pct",
    "opp_ft_rate",
    "opp_tov_pct",
    "opp_oreb_pct",
    "opp_fgm",
    "opp_fga",
    "opp_fg_pct",
    "opp_fg3m",
    "opp_fg3a",
    "opp_fg3_pct",
    "opp_ftm",
    "opp_fta",
    "opp_ft_pct",
    "opp_oreb",
    "opp_dreb",
    "opp_reb",
    "opp_ast",
    "opp_tov",
    "opp_stl",
    "opp_blk",
    "opp_pts",
    "travel_distance_km",
    "travel_km_last_7_days",
    "travel_km_last_10_days",
    "travel_km_last_14_days",
}

TEAM_ALLOWED_BASE_COLUMNS -= TEAM_OPTIONAL_DROP_COLUMNS

# Head-to-head history and season-aware features get added later in the pipeline
# and should be eligible for rolling/pre-game feature generation as well.
TEAM_ALLOWED_BASE_COLUMNS.update(
    {
        "season_h2h_games_prior",
        "season_h2h_wins_prior",
        "season_h2h_win_pct_prior",
        "season_h2h_margin_avg_prior",
    }
)


PRESEASON_PATTERN = re.compile(r"\bpre[-\s]?season\b|\bexhibition\b")

EARTH_RADIUS_KM = 6371.0


ARENA_COORDINATES: dict[str, tuple[float, float]] = {
    "atlanta": (33.7573, -84.3963),
    "atlantahawks": (33.7573, -84.3963),
    "boston": (42.3663, -71.0621),
    "bostonceltics": (42.3663, -71.0621),
    "brooklyn": (40.6826, -73.9754),
    "brooklynnets": (40.6826, -73.9754),
    "charlotte": (35.2251, -80.8392),
    "charlottehornets": (35.2251, -80.8392),
    "chicago": (41.8807, -87.6742),
    "chicagobulls": (41.8807, -87.6742),
    "cleveland": (41.4965, -81.6882),
    "clevelandcavaliers": (41.4965, -81.6882),
    "dallas": (32.7905, -96.8104),
    "dallasmavericks": (32.7905, -96.8104),
    "denver": (39.7487, -105.0077),
    "denvernuggets": (39.7487, -105.0077),
    "detroit": (42.3410, -83.0550),
    "detroitpistons": (42.3410, -83.0550),
    "goldenstate": (37.7503, -122.2029),
    "goldenstatewarriors": (37.7503, -122.2029),
    "oakland": (37.7503, -122.2029),
    "sanfrancisco": (37.7680, -122.3877),
    "houston": (29.7508, -95.3621),
    "houstonrockets": (29.7508, -95.3621),
    "indianapolis": (39.7640, -86.1555),
    "indiana": (39.7640, -86.1555),
    "indianapacers": (39.7640, -86.1555),
    "memphis": (35.1382, -90.0506),
    "memphisgrizzlies": (35.1382, -90.0506),
    "miami": (25.7814, -80.1870),
    "miamiheat": (25.7814, -80.1870),
    "milwaukee": (43.0451, -87.9171),
    "milwaukeebucks": (43.0451, -87.9171),
    "minneapolis": (44.9795, -93.2760),
    "minnesota": (44.9795, -93.2760),
    "minnesotatimberwolves": (44.9795, -93.2760),
    "neworleans": (29.9490, -90.0812),
    "neworleanspelicans": (29.9490, -90.0812),
    "newyork": (40.7505, -73.9934),
    "newyorkknicks": (40.7505, -73.9934),
    "oklahomacity": (35.4634, -97.5151),
    "oklahomacitythunder": (35.4634, -97.5151),
    "orlando": (28.5392, -81.3839),
    "orlandomagic": (28.5392, -81.3839),
    "philadelphia": (39.9012, -75.1720),
    "philadelphia76ers": (39.9012, -75.1720),
    "phoenix": (33.4458, -112.0712),
    "phoenixsuns": (33.4458, -112.0712),
    "portland": (45.5316, -122.6668),
    "portlandtrailblazers": (45.5316, -122.6668),
    "sacramento": (38.5802, -121.4997),
    "sacramentokings": (38.5802, -121.4997),
    "sanantonio": (29.4269, -98.4375),
    "sanantoniospurs": (29.4269, -98.4375),
    "sanjose": (37.7503, -122.2029),
    "toronto": (43.6435, -79.3791),
    "torontoraptors": (43.6435, -79.3791),
    "saltlakecity": (40.7683, -111.9011),
    "utah": (40.7683, -111.9011),
    "utahjazz": (40.7683, -111.9011),
    "washington": (38.8981, -77.0209),
    "washingtonwizards": (38.8981, -77.0209),
    "losangeles": (34.0430, -118.2673),
    "losangeleslakers": (34.0430, -118.2673),
    "losangelesclippers": (34.0430, -118.2673),
    "lakers": (34.0430, -118.2673),
    "clippers": (34.0430, -118.2673),
    "la": (34.0430, -118.2673),
    "lac": (34.0430, -118.2673),
    "lal": (34.0430, -118.2673),
}

MISSING_LOCATION_KEYS: set[str] = set()


def normalize_location_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    normalized = re.sub(r"[^a-z0-9]", "", text)
    return normalized or None


def lookup_location_coordinates(
    city_value: object, team_value: object | None = None
) -> tuple[float, float] | None:
    candidates: list[str] = []
    for raw in (city_value, team_value):
        key = normalize_location_key(raw)
        if key and key not in candidates:
            candidates.append(key)
    for key in candidates:
        coords = ARENA_COORDINATES.get(key)
        if coords is not None:
            return coords
    for key in candidates:
        if key not in MISSING_LOCATION_KEYS:
            MISSING_LOCATION_KEYS.add(key)
            logging.debug("No arena coordinates for location key: %s", key)
    return None


def compute_haversine_series(
    prev_lat: pd.Series, prev_lon: pd.Series, lat: pd.Series, lon: pd.Series
) -> pd.Series:
    if lat.empty:
        return pd.Series(dtype=float)
    prev_lat_vals = pd.to_numeric(prev_lat, errors="coerce")
    prev_lon_vals = pd.to_numeric(prev_lon, errors="coerce")
    lat_vals = pd.to_numeric(lat, errors="coerce")
    lon_vals = pd.to_numeric(lon, errors="coerce")
    mask = (
        prev_lat_vals.notna()
        & prev_lon_vals.notna()
        & lat_vals.notna()
        & lon_vals.notna()
    )
    result = np.full(len(lat_vals), np.nan, dtype=float)
    if mask.any():
        lat1 = np.radians(prev_lat_vals[mask].to_numpy(dtype=float))
        lon1 = np.radians(prev_lon_vals[mask].to_numpy(dtype=float))
        lat2 = np.radians(lat_vals[mask].to_numpy(dtype=float))
        lon2 = np.radians(lon_vals[mask].to_numpy(dtype=float))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        distances = EARTH_RADIUS_KM * c
        result[mask.to_numpy()] = distances
    return pd.Series(result, index=lat.index, dtype=float)


def compute_recent_travel_sum(
    dates: pd.Series, distances: pd.Series, window_days: int
) -> pd.Series:
    if dates.empty:
        return pd.Series(dtype=float)

    normalized_dates = pd.to_datetime(dates, errors="coerce")
    distance_vals = pd.to_numeric(distances, errors="coerce")
    valid_mask = normalized_dates.notna() & distance_vals.notna()
    result = np.full(len(dates), np.nan, dtype=float)

    if valid_mask.any():
        valid_dates = normalized_dates[valid_mask].to_numpy(dtype="datetime64[ns]")
        valid_indices = np.flatnonzero(valid_mask.to_numpy())
        valid_distances = distance_vals[valid_mask].to_numpy(dtype=float)
        valid_ns = valid_dates.view("int64")
        window_ns = pd.to_timedelta(window_days, unit="D").value
        start = 0
        running = 0.0
        for pos, (ns_val, dist_val) in enumerate(zip(valid_ns, valid_distances)):
            running += dist_val
            while start <= pos and ns_val - valid_ns[start] > window_ns:
                running -= valid_distances[start]
                start += 1
            result[valid_indices[pos]] = running

    return pd.Series(result, index=dates.index, dtype=float)


def determine_game_era(game_date: pd.Timestamp) -> str:
    """Categorize games into eras, highlighting the modern era shift."""

    if pd.isna(game_date):
        return "unknown_era"
    if isinstance(game_date, pd.Timestamp) and getattr(game_date, "tzinfo", None):
        # Normalize to timezone-naive comparisons by dropping tz awareness.
        try:
            game_date = game_date.tz_localize(None)
        except TypeError:
            game_date = game_date.tz_convert(None)
    if game_date >= pd.Timestamp("2014-10-01"):
        return "modern_era"
    if game_date >= pd.Timestamp("2004-10-01"):
        return "pace_and_space_era"
    return "pre_modern_era"


def determine_season(game_date: pd.Timestamp) -> str | None:
    """Map a game date to an NBA season string like '2014-2015'."""

    if pd.isna(game_date):
        return None
    year = int(game_date.year)
    start_year = year if game_date.month >= 7 else year - 1
    end_year = start_year + 1
    return f"{start_year}-{end_year}"


def season_start_year(season: object) -> int | None:
    """Extract the starting year from a season label like '2014-2015'."""

    if season is None or pd.isna(season):
        return None
    if isinstance(season, (int, np.integer)):
        return int(season)
    if isinstance(season, float):
        return int(season)
    if not isinstance(season, str):
        season = str(season)
    if not season:
        return None
    try:
        return int(season.split("-")[0])
    except (ValueError, IndexError):
        return None


def drop_preseason_rows(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Remove preseason/exhibition rows if game metadata exposes the label."""

    if df.empty:
        return df

    candidate_columns = [
        col
        for col in (
            "season_type",
            "game_type",
            "game_label",
            "game_sub_label",
        )
        if col in df.columns
    ]
    if not candidate_columns:
        return df

    mask = pd.Series(False, index=df.index)
    for col in candidate_columns:
        series = df[col]
        if pd.api.types.is_string_dtype(series) or series.dtype == "O":
            normalized = series.str.lower().str.strip()
        else:
            normalized = series.astype(str).str.lower().str.strip()
        mask |= normalized.str.contains(PRESEASON_PATTERN, na=False)

    removed = int(mask.sum())
    if removed:
        logging.info(
            "Filtered %d preseason rows from %s data using columns %s",
            removed,
            dataset_name,
            candidate_columns,
        )
        df = df.loc[~mask].copy()
    return df


def add_season_column(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns and "season" not in df.columns:
        df["season"] = df["game_date"].map(determine_season)
    return df


def filter_min_season(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep only rows whose season is on/after the configured minimum."""

    if "season" not in df.columns:
        return df
    start_years = df["season"].map(season_start_year)
    mask = start_years.isna() | (start_years >= MIN_SEASON_START_YEAR)
    if not bool(mask.all()):
        removed = int((~mask).sum())
        if removed:
            logging.info(
                "Filtered %d %s rows prior to %s-%s season",
                removed,
                label,
                MIN_SEASON_START_YEAR,
                MIN_SEASON_START_YEAR + 1,
            )
        df = df.loc[mask].copy()
    return df


def detect_sep(path: str) -> str:
    """Detect delimiter (comma vs tab) for a text file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(4096)
    if "\t" in sample.splitlines()[0]:
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


_FIRST_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")


def to_snake(name: str) -> str:
    """Normalize column names to snake_case while handling all-caps headers."""

    name = name.strip()
    if not name:
        return name
    name = re.sub(r"[\s\-]+", "_", name)
    if name.upper() == name:
        normalized = name.lower()
    else:
        normalized = _FIRST_CAMEL_RE.sub(r"\1_\2", name)
        normalized = _ACRONYM_RE.sub(r"\1_\2", normalized)
        normalized = normalized.lower()
    normalized = re.sub(r"__+", "_", normalized)
    return normalized.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: to_snake(col) for col in df.columns})


_DATE_ONLY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a wide range of timestamp strings, keeping only the calendar day."""

    if series.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if isinstance(parsed.dtype, DatetimeTZDtype):
            try:
                parsed = parsed.dt.tz_localize(None)
            except TypeError:
                parsed = parsed.dt.tz_convert(None)
        return parsed.dt.normalize()

    as_str = series.astype(str)
    extracted = as_str.str.extract(_DATE_ONLY_RE, expand=False)
    parsed = pd.to_datetime(extracted, format="%Y-%m-%d", errors="coerce")
    return parsed


def ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    """Populate a timezone-naive ``game_date`` column when possible."""

    if df.empty:
        df.attrs["_game_date_sources"] = {}
        return df

    candidate_columns: list[str] = []
    if "game_date" in df.columns:
        candidate_columns.append("game_date")

    fallback_columns = [
        "game_date_est",
        "game_date_time",
        "game_date_time_est",
        "game_date_time_utc",
        "game_date_utc",
    ]
    candidate_columns.extend([col for col in fallback_columns if col in df.columns])

    dynamic_candidates = [
        col
        for col in df.columns
        if "date" in col
        and col not in candidate_columns
        and df[col].notna().any()
    ]
    candidate_columns.extend(dynamic_candidates)

    if "game_date" in df.columns:
        result = _parse_datetime(df["game_date"])
    else:
        result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    coverage: dict[str, int] = {}
    for col in candidate_columns:
        parsed = _parse_datetime(df[col])
        non_na = parsed.notna()
        if not non_na.any():
            continue
        coverage[col] = int(non_na.sum())
        fill_mask = result.isna() & non_na
        if fill_mask.any():
            result.loc[fill_mask] = parsed.loc[fill_mask]
            if col != "game_date":
                logging.info("Filled game_date from %s", col)

    df["game_date"] = result
    df.attrs["_game_date_sources"] = coverage
    return df


def backfill_game_dates(
    games_df: pd.DataFrame, sources: Sequence[tuple[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Fill missing ``game_date`` values using other tables keyed by ``game_id``.

    The raw games feed occasionally ships schedule-style rows without concrete
    timestamps. When the team or player box-score files do carry the dates we
    can safely copy them over so those games survive the history filters.
    """

    if games_df.empty or "game_id" not in games_df.columns:
        return games_df

    result = games_df.copy()
    if "game_date" not in result.columns:
        result["game_date"] = pd.NaT

    for name, df in sources:
        if df is None or df.empty:
            continue
        if "game_id" not in df.columns or "game_date" not in df.columns:
            continue
        available = df.loc[df["game_date"].notna(), ["game_id", "game_date"]]
        if available.empty:
            continue
        available = (
            available.sort_values(["game_id", "game_date"]).drop_duplicates(
                "game_id", keep="first"
            )
        )
        merge_col = f"_{name}_game_date"
        result = result.merge(
            available.rename(columns={"game_date": merge_col}),
            on="game_id",
            how="left",
        )
        fill_mask = result["game_date"].isna() & result[merge_col].notna()
        if fill_mask.any():
            result.loc[fill_mask, "game_date"] = result.loc[fill_mask, merge_col]
            logging.info(
                "Filled %d missing game_date values from %s data",
                int(fill_mask.sum()),
                name,
            )
        result = result.drop(columns=[merge_col])

    return result


def load_table(path: str) -> pd.DataFrame:
    sep = detect_sep(path)
    logging.info("Loading %s with sep='%s'", path, sep.replace("\t", "\\t"))
    df = pd.read_csv(
        path,
        sep=sep,
        low_memory=False,
        on_bad_lines="skip",
    )
    df = normalize_columns(df)
    df = ensure_game_date(df)
    sources: dict[str, int] = df.attrs.get("_game_date_sources", {})
    if sources:
        primary = max(sources.items(), key=lambda item: item[1])[0]
        logging.info("Detected game_date source column: %s", primary)
        for col, count in sources.items():
            logging.info("Parsed %d/%d game_date values from %s", count, len(df), col)
    elif "game_date" in df.columns:
        valid_dates = int(df["game_date"].notna().sum())
        logging.info(
            "Parsed %d/%d game_date values from game_date",
            valid_dates,
            len(df),
        )
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    for col in ("team_id", "opponent_team_id", "home_team_id", "away_team_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def coerce_numeric(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    exclude_set = set(exclude)
    for col in df.columns:
        if col in exclude_set:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def map_binary_home_away(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"home", "h", "1", "true", "t"}:
        return 1
    if text in {"away", "a", "0", "false", "f"}:
        return 0
    try:
        return int(float(text))
    except Exception:
        return np.nan


def map_binary_win_loss(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"win", "w", "1", "true", "t"}:
        return 1
    if text in {"loss", "l", "0", "false", "f"}:
        return 0
    try:
        return int(float(text))
    except Exception:
        return np.nan


def compute_streak(values: pd.Series) -> pd.Series:
    streak: list[float] = []
    current = 0
    for val in values.astype(float):
        if pd.isna(val):
            current = 0
            streak.append(np.nan)
            continue
        if val >= 0.5:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
        streak.append(float(current))
    return pd.Series(streak, index=values.index, dtype=float)


def compute_filtered_streak(values: pd.Series, mask: pd.Series) -> pd.Series:
    streak: list[float] = []
    current = 0
    for val, include in zip(values.astype(float), mask.astype(int)):
        if not include:
            streak.append(float(current))
            continue
        if pd.isna(val):
            current = 0
            streak.append(np.nan)
            continue
        if val >= 0.5:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
        streak.append(float(current))
    return pd.Series(streak, index=values.index, dtype=float)


def compute_consecutive_true(mask: pd.Series) -> pd.Series:
    """Count consecutive truthy values within a binary mask."""

    if mask.empty:
        return pd.Series(dtype=float)
    numeric = mask.fillna(0).astype(int)
    groups = (numeric == 0).cumsum()
    streak = numeric.groupby(groups).cumsum() * numeric
    return streak.astype(float)


def compute_recent_game_counts(dates: pd.Series, window_days: int) -> pd.Series:
    """Count prior games within a lookback window for fatigue features."""

    if dates.empty:
        return pd.Series(dtype=float)

    normalized = dates.copy()
    valid_mask = normalized.notna()
    result = np.full(len(normalized), np.nan, dtype=float)

    if valid_mask.any():
        valid_dates = normalized[valid_mask].to_numpy(dtype="datetime64[ns]")
        valid_indices = np.flatnonzero(valid_mask.to_numpy())
        valid_ns = valid_dates.view("int64")
        window_ns = pd.to_timedelta(window_days, unit="D").value
        start = 0
        counts = np.zeros(len(valid_ns), dtype=float)
        for idx, value in enumerate(valid_ns):
            while start < idx and value - valid_ns[start] > window_ns:
                start += 1
            counts[idx] = float(idx - start)
        result[valid_indices] = counts

    return pd.Series(result, index=dates.index, dtype=float)


def clean_team_table(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.copy()
    exclude = {
        "game_id",
        "game_date",
        "team_city",
        "team_name",
        "opponent_team_city",
        "opponent_team_name",
        "coach_id",
        "team_id",
        "opponent_team_id",
        "game_type",
        "season",
    }
    if "home" in team_df.columns:
        team_df["home"] = team_df["home"].apply(map_binary_home_away)
    if "win" in team_df.columns:
        team_df["win"] = team_df["win"].apply(map_binary_win_loss)
    team_df = coerce_numeric(team_df, exclude)
    if "home" in team_df.columns:
        team_df["home"] = team_df["home"].fillna(0).astype(int)
    if "win" in team_df.columns:
        team_df["win"] = team_df["win"].fillna(0).astype(int)
    return team_df


def minutes_to_float(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    text = str(val).strip()
    if not text:
        return np.nan
    try:
        if text.upper().startswith("PT"):
            return pd.to_timedelta(text).total_seconds() / 60.0
        if ":" in text:
            return pd.to_timedelta(text).total_seconds() / 60.0
        return float(text)
    except Exception:
        return np.nan


def clean_player_table(player_df: pd.DataFrame) -> pd.DataFrame:
    player_df = player_df.copy()
    exclude = {
        "first_name",
        "last_name",
        "person_id",
        "game_id",
        "game_date",
        "playerteam_city",
        "playerteam_name",
        "opponentteam_city",
        "opponentteam_name",
        "game_type",
        "game_label",
        "game_sub_label",
        "series_game_number",
        "season",
    }
    player_df = coerce_numeric(player_df, exclude)
    if "num_minutes" in player_df.columns:
        player_df.loc[:, "num_minutes_float"] = player_df["num_minutes"].map(
            minutes_to_float
        )
    else:
        player_df.loc[:, "num_minutes_float"] = np.nan
    if "win" in player_df.columns:
        player_df.loc[:, "win"] = player_df["win"].apply(map_binary_win_loss)
    return player_df


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace({0: np.nan})


def masked_cumulative_average(
    values: pd.Series, mask: pd.Series, group_index: Sequence[pd.Series]
) -> pd.Series:
    """Running average of ``values`` filtered by a binary ``mask`` within groups."""

    if values.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)

    mask_int = mask.fillna(0).astype(int)
    valid = mask_int.eq(1) & values.notna()
    counts = valid.astype(int)
    if group_index:
        counts = counts.groupby(group_index).cumsum()
        sums = values.where(valid, 0.0).groupby(group_index).cumsum()
    else:
        counts = counts.cumsum()
        sums = values.where(valid, 0.0).cumsum()
    return safe_div(sums, counts)


def add_elo_ratings(
    team_df: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 75.0,
) -> pd.DataFrame:
    if "team_id" not in team_df.columns or "game_id" not in team_df.columns:
        return team_df
    if "opponent_team_id" not in team_df.columns:
        team_df["elo_rating"] = np.nan
        return team_df

    working = team_df.copy()
    working["elo_rating"] = np.nan

    sort_keys = [
        col
        for col in ["game_date", "game_id", "team_id"]
        if col in working.columns
    ]
    if sort_keys:
        ordered = working.sort_values(sort_keys)
    else:
        ordered = working

    ratings: dict[int, float] = {}

    active_season_year: int | None = None

    for _, game_rows in ordered.groupby("game_id", sort=False):
        game_year: int | None = None
        if "season" in game_rows.columns:
            season_years = [
                season_start_year(val)
                for val in game_rows["season"].dropna().unique().tolist()
            ]
            season_years = [year for year in season_years if year is not None]
            if season_years:
                game_year = int(min(season_years))

        if game_year is not None:
            if active_season_year is None:
                active_season_year = game_year
            elif game_year > active_season_year:
                ratings = {}
                active_season_year = game_year

        entries: list[dict[str, object]] = []
        for idx, row in game_rows.iterrows():
            team_id = row.get("team_id")
            if pd.isna(team_id):
                continue
            opponent_id = row.get("opponent_team_id")
            try:
                team_id_int = int(team_id)
            except Exception:
                continue
            opp_id_int: int | None
            if pd.isna(opponent_id):
                opp_id_int = None
            else:
                try:
                    opp_id_int = int(opponent_id)
                except Exception:
                    opp_id_int = None

            rating = ratings.get(team_id_int, float(base_rating))
            home_val = row.get("home", 0) if "home" in row else 0
            try:
                is_home = bool(int(home_val))
            except Exception:
                is_home = False
            win_value = row.get("win") if "win" in row else np.nan
            entries.append(
                {
                    "index": idx,
                    "team_id": team_id_int,
                    "opponent_id": opp_id_int,
                    "rating": float(rating),
                    "is_home": is_home,
                    "win": win_value,
                }
            )

        if not entries:
            continue

        for entry in entries:
            working.at[entry["index"], "elo_rating"] = entry["rating"]

        updates: list[tuple[int, float]] = []
        for entry in entries:
            opp_id = entry["opponent_id"]
            if opp_id is None:
                updates.append((entry["team_id"], entry["rating"]))
                continue
            opponent_entry = next(
                (e for e in entries if e["team_id"] == opp_id), None
            )
            if opponent_entry is None:
                opp_rating = ratings.get(opp_id, float(base_rating))
                opp_is_home = False
            else:
                opp_rating = float(opponent_entry["rating"])
                opp_is_home = bool(opponent_entry["is_home"])

            team_adj = float(entry["rating"]) + (
                home_advantage if entry["is_home"] else 0.0
            )
            opp_adj = float(opp_rating) + (
                home_advantage if opp_is_home else 0.0
            )
            expected = 1.0 / (1.0 + 10 ** ((opp_adj - team_adj) / 400.0))
            win_val = entry["win"]
            if pd.isna(win_val):
                updates.append((entry["team_id"], float(entry["rating"])))
                continue
            actual = float(win_val)
            new_rating = float(entry["rating"]) + k_factor * (actual - expected)
            updates.append((entry["team_id"], new_rating))

        for team_id_int, new_rating in updates:
            ratings[team_id_int] = new_rating

    return working


def add_head_to_head_history(team_df: pd.DataFrame) -> pd.DataFrame:
    """Track prior same-season head-to-head results for each matchup."""

    required = {
        "team_id",
        "opponent_team_id",
        "season",
        "game_date",
        "game_id",
        "win",
        "margin",
    }
    if not required.issubset(team_df.columns):
        return team_df

    working = team_df.copy()
    sort_cols = [
        col
        for col in ["team_id", "season", "opponent_team_id", "game_date", "game_id"]
        if col in working.columns
    ]
    if sort_cols:
        working = working.sort_values(sort_cols)

    group_cols = ["team_id", "opponent_team_id", "season"]
    grouped = working.groupby(group_cols, sort=False)

    working["_h2h_games_prior"] = grouped.cumcount().astype(float)
    working["_h2h_prev_win"] = grouped["win"].shift(1)
    working["_h2h_prev_margin"] = grouped["margin"].shift(1)

    working["_h2h_prev_win"] = working["_h2h_prev_win"].fillna(0.0)
    working["_h2h_prev_margin"] = working["_h2h_prev_margin"].fillna(0.0)

    working["season_h2h_wins_prior"] = grouped["_h2h_prev_win"].cumsum().astype(float)
    working["season_h2h_games_prior"] = working["_h2h_games_prior"].astype(float)
    working["season_h2h_margin_sum_prior"] = (
        grouped["_h2h_prev_margin"].cumsum().astype(float)
    )
    working["season_h2h_win_pct_prior"] = safe_div(
        working["season_h2h_wins_prior"], working["season_h2h_games_prior"]
    )
    working["season_h2h_margin_avg_prior"] = safe_div(
        working["season_h2h_margin_sum_prior"], working["season_h2h_games_prior"]
    )

    result = team_df.copy()
    for col in (
        "season_h2h_games_prior",
        "season_h2h_wins_prior",
        "season_h2h_win_pct_prior",
        "season_h2h_margin_avg_prior",
    ):
        result[col] = working[col].astype(float)

    return result


def build_team_features(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.copy()
    base_numeric = [
        "field_goals_attempted",
        "free_throws_attempted",
        "rebounds_offensive",
        "turnovers",
        "field_goals_made",
        "three_pointers_made",
        "three_pointers_attempted",
        "team_score",
        "opponent_score",
        "assists",
        "blocks",
        "steals",
        "rebounds_defensive",
        "fouls_personal",
        "rebounds_total",
        "plus_minus_points",
        "num_minutes",
    ]
    for col in base_numeric:
        if col not in team_df.columns:
            team_df[col] = np.nan
    for travel_col in (
        "travel_distance_km",
        "travel_km_last_7_days",
        "travel_km_last_10_days",
        "travel_km_last_14_days",
    ):
        if travel_col not in team_df.columns:
            team_df[travel_col] = np.nan
    team_df["minutes"] = pd.to_numeric(team_df.get("num_minutes"), errors="coerce")
    team_df.drop(columns=["num_minutes"], inplace=True, errors="ignore")
    team_df["games_played"] = 1.0
    if "win" in team_df.columns:
        win_numeric = pd.to_numeric(team_df["win"], errors="coerce")
        team_df["win"] = win_numeric
        team_df["loss"] = np.where(win_numeric.notna(), 1 - win_numeric, np.nan)
    else:
        team_df["loss"] = np.nan
    team_df["win_pct"] = team_df["win"]
    team_df["poss"] = (
        team_df["field_goals_attempted"]
        + 0.44 * team_df["free_throws_attempted"]
        - team_df["rebounds_offensive"]
        + team_df["turnovers"]
    )
    valid_poss = team_df["poss"].dropna()
    if not valid_poss.empty:
        pos_q = valid_poss.quantile([0.01, 0.99])
        lower = pos_q.iloc[0] if not pd.isna(pos_q.iloc[0]) else 70
        upper = pos_q.iloc[1] if not pd.isna(pos_q.iloc[1]) else 135
        if lower >= upper:
            lower, upper = 70, 135
        team_df["poss"] = team_df["poss"].clip(lower=lower, upper=upper)
    else:
        team_df["poss"] = team_df["poss"].clip(lower=70, upper=135)
    team_df["poss"] = team_df["poss"].where(team_df["poss"] > 0, np.nan)
    team_df["efg"] = safe_div(
        team_df["field_goals_made"] + 0.5 * team_df["three_pointers_made"],
        team_df["field_goals_attempted"],
    )
    team_df["tov_pct"] = safe_div(team_df["turnovers"], team_df["poss"])
    team_df["ft_rate"] = safe_div(
        team_df["free_throws_attempted"], team_df["field_goals_attempted"]
    )
    team_df["ast_ratio"] = safe_div(
        team_df["assists"], team_df["field_goals_made"].replace(0, np.nan)
    )
    team_df["ast_to_ratio"] = safe_div(team_df["assists"], team_df["turnovers"])
    team_df["ts_pct"] = safe_div(
        team_df["team_score"],
        2
        * (
            team_df["field_goals_attempted"]
            + 0.44 * team_df["free_throws_attempted"]
        ),
    )
    team_df["pace"] = safe_div(team_df["poss"] * 48.0, team_df["minutes"])

    opp_lookup = team_df[
        [
            "game_id",
            "team_id",
            "poss",
            "rebounds_defensive",
            "rebounds_offensive",
            "rebounds_total",
            "team_score",
            "field_goals_made",
            "field_goals_attempted",
            "field_goals_percentage",
            "three_pointers_made",
            "three_pointers_attempted",
            "three_pointers_percentage",
            "free_throws_made",
            "free_throws_attempted",
            "free_throws_percentage",
            "assists",
            "turnovers",
            "steals",
            "blocks",
            "fouls_personal",
            "efg",
            "ft_rate",
            "tov_pct",
        ]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "poss": "opp_poss",
            "rebounds_defensive": "opp_dreb",
            "rebounds_offensive": "opp_oreb",
            "rebounds_total": "opp_reb",
            "team_score": "opp_pts",
            "field_goals_made": "opp_fgm",
            "field_goals_attempted": "opp_fga",
            "field_goals_percentage": "opp_fg_pct",
            "three_pointers_made": "opp_fg3m",
            "three_pointers_attempted": "opp_fg3a",
            "three_pointers_percentage": "opp_fg3_pct",
            "free_throws_made": "opp_ftm",
            "free_throws_attempted": "opp_fta",
            "free_throws_percentage": "opp_ft_pct",
            "assists": "opp_ast",
            "turnovers": "opp_tov",
            "steals": "opp_stl",
            "blocks": "opp_blk",
            "fouls_personal": "opp_pf",
            "efg": "opp_efg_pct",
            "ft_rate": "opp_ft_rate",
            "tov_pct": "opp_tov_pct",
        }
    )
    team_df = team_df.merge(
        opp_lookup,
        on=["game_id", "opponent_team_id"],
        how="left",
    )
    team_df["oreb_pct"] = safe_div(
        team_df["rebounds_offensive"],
        team_df["rebounds_offensive"] + team_df["opp_dreb"],
    )
    team_df["dreb_pct"] = safe_div(
        team_df["rebounds_defensive"],
        team_df["rebounds_defensive"] + team_df["opp_oreb"],
    )
    team_df["reb_pct"] = safe_div(
        team_df["rebounds_total"],
        team_df["rebounds_total"] + team_df["opp_reb"],
    )
    team_df["opp_oreb_pct"] = safe_div(
        team_df["opp_oreb"], team_df["opp_oreb"] + team_df["rebounds_defensive"]
    )

    off_ratio = safe_div(team_df["team_score"], team_df["poss"]).astype(float)
    def_ratio = safe_div(
        team_df["opponent_score"], team_df["opp_poss"]
    ).astype(float)
    team_df["off_rtg"] = 100.0 * off_ratio
    team_df["def_rtg"] = 100.0 * def_ratio
    team_df["margin"] = team_df["team_score"] - team_df["opponent_score"]
    team_df["net_rtg"] = team_df["off_rtg"] - team_df["def_rtg"]

    team_df = add_head_to_head_history(team_df)

    if "team_id" in team_df.columns:
        sort_cols = [
            col
            for col in ["team_id", "season", "game_date", "game_id"]
            if col in team_df.columns
        ]
        if sort_cols:
            ordered = team_df.sort_values(sort_cols).copy()
        else:
            ordered = team_df.copy()
        group_cols = [col for col in ["team_id", "season"] if col in ordered.columns]
        if isinstance(ordered["game_date"].dtype, DatetimeTZDtype):
            ordered["_game_datetime"] = ordered["game_date"].dt.tz_convert(None)
        else:
            ordered["_game_datetime"] = pd.to_datetime(
                ordered["game_date"], errors="coerce"
            )
        travel_ready = False
        if "home" in ordered.columns:
            home_flag = ordered["home"].fillna(0).astype(int)
            team_city_series = ordered.get("team_city")
            opp_city_series = ordered.get("opponent_team_city")
            team_name_series = ordered.get("team_name")
            opp_name_series = ordered.get("opponent_team_name")
            if (team_city_series is not None or team_name_series is not None) and (
                opp_city_series is not None or opp_name_series is not None
            ):
                if team_city_series is not None:
                    location_city = team_city_series.copy()
                else:
                    location_city = pd.Series(np.nan, index=ordered.index, dtype=object)
                if opp_city_series is not None:
                    location_city = location_city.where(home_flag == 1, opp_city_series)

                if team_name_series is not None:
                    location_name = team_name_series.copy()
                else:
                    location_name = pd.Series(np.nan, index=ordered.index, dtype=object)
                if opp_name_series is not None:
                    location_name = location_name.where(
                        home_flag == 1, opp_name_series
                    )

                coords = [
                    lookup_location_coordinates(city, name)
                    for city, name in zip(location_city, location_name)
                ]
                latitudes = pd.Series(
                    [coord[0] if coord is not None else np.nan for coord in coords],
                    index=ordered.index,
                    dtype=float,
                )
                longitudes = pd.Series(
                    [coord[1] if coord is not None else np.nan for coord in coords],
                    index=ordered.index,
                    dtype=float,
                )
                ordered["_game_lat"] = latitudes
                ordered["_game_lon"] = longitudes
                travel_ready = True
        if group_cols:
            grouped = ordered.groupby(group_cols, sort=False)
            group_index = [ordered[col] for col in group_cols]

            rest_delta = grouped["_game_datetime"].diff()
            ordered["days_since_last_game"] = rest_delta.dt.total_seconds() / 86400.0
            if travel_ready:
                prev_lat = grouped["_game_lat"].shift(1)
                prev_lon = grouped["_game_lon"].shift(1)
                ordered["travel_distance_km"] = compute_haversine_series(
                    prev_lat, prev_lon, ordered["_game_lat"], ordered["_game_lon"]
                )
                team_df["travel_distance_km"] = ordered["travel_distance_km"].astype(float)
            for window in (7, 10, 14):
                counts = grouped.apply(
                    lambda g: compute_recent_game_counts(g["_game_datetime"], window)
                )
                ordered[f"games_last_{window}_days"] = counts.reset_index(
                    level=group_cols, drop=True
                )
            team_df["days_since_last_game"] = (
                ordered["days_since_last_game"].astype(float)
            )
            for window in (7, 10, 14):
                col_name = f"games_last_{window}_days"
                team_df[col_name] = ordered[col_name].astype(float)
            if travel_ready:
                for window in (7, 10, 14):
                    travel_window = grouped.apply(
                        lambda g, w=window: compute_recent_travel_sum(
                            g["_game_datetime"], g["travel_distance_km"], w
                        )
                    )
                    team_df[f"travel_km_last_{window}_days"] = travel_window.reset_index(
                        level=group_cols, drop=True
                    ).astype(float)
                ordered.drop(columns=["_game_lat", "_game_lon"], inplace=True, errors="ignore")

            ordered["season_games_played_total"] = grouped.cumcount() + 1
            win_series = ordered["win"].astype(float)
            ordered["season_wins_total"] = (
                win_series.fillna(0.0).groupby(group_index).cumsum()
            )
            ordered["season_win_pct"] = safe_div(
                ordered["season_wins_total"], ordered["season_games_played_total"]
            )
            team_df["season_win_pct"] = ordered["season_win_pct"].astype(float)

            streak_series = grouped["win"].apply(
                lambda s: compute_streak(s.astype(float))
            )
            team_df["win_loss_streak"] = streak_series.reset_index(
                level=group_cols, drop=True
            ).astype(float)

            if "home" in ordered.columns:
                home_flag = ordered["home"].fillna(0).astype(int)
                away_flag = 1 - home_flag
                ordered["_home_flag"] = home_flag
                ordered["_away_flag"] = away_flag

                home_streak = grouped.apply(
                    lambda g: compute_filtered_streak(
                        g["win"].astype(float), g["_home_flag"]
                    )
                )
                away_streak = grouped.apply(
                    lambda g: compute_filtered_streak(
                        g["win"].astype(float), g["_away_flag"]
                    )
                )
                team_df["home_win_loss_streak"] = home_streak.reset_index(
                    level=group_cols, drop=True
                ).astype(float)
                team_df["away_win_loss_streak"] = away_streak.reset_index(
                    level=group_cols, drop=True
                ).astype(float)

                team_df["season_home_games_played_total"] = (
                    home_flag.groupby(group_index).cumsum().astype(float)
                )
                team_df["season_away_games_played_total"] = (
                    away_flag.groupby(group_index).cumsum().astype(float)
                )

                home_wins = (win_series.fillna(0.0) * home_flag)
                away_wins = (win_series.fillna(0.0) * away_flag)
                team_df["season_home_wins_total"] = (
                    home_wins.groupby(group_index).cumsum().astype(float)
                )
                team_df["season_away_wins_total"] = (
                    away_wins.groupby(group_index).cumsum().astype(float)
                )

                team_df["season_home_win_pct"] = safe_div(
                    team_df["season_home_wins_total"],
                    team_df["season_home_games_played_total"],
                )
                team_df["season_away_win_pct"] = safe_div(
                    team_df["season_away_wins_total"],
                    team_df["season_away_games_played_total"],
                )
                team_df["home_win_pct"] = team_df["season_home_win_pct"]
                team_df["away_win_pct"] = team_df["season_away_win_pct"]

                margin_vals = ordered["margin"].astype(float)
                off_vals = ordered["off_rtg"].astype(float)
                def_vals = ordered["def_rtg"].astype(float)
                team_pts_vals = ordered["team_score"].astype(float)
                opp_pts_vals = ordered["opponent_score"].astype(float)

                team_df["season_home_margin_avg"] = masked_cumulative_average(
                    margin_vals, home_flag, group_index
                )
                team_df["season_away_margin_avg"] = masked_cumulative_average(
                    margin_vals, away_flag, group_index
                )
                team_df["season_home_off_rtg_avg"] = masked_cumulative_average(
                    off_vals, home_flag, group_index
                )
                team_df["season_away_off_rtg_avg"] = masked_cumulative_average(
                    off_vals, away_flag, group_index
                )
                team_df["season_home_def_rtg_avg"] = masked_cumulative_average(
                    def_vals, home_flag, group_index
                )
                team_df["season_away_def_rtg_avg"] = masked_cumulative_average(
                    def_vals, away_flag, group_index
                )
                team_df["season_home_points_avg"] = masked_cumulative_average(
                    team_pts_vals, home_flag, group_index
                )
                team_df["season_away_points_avg"] = masked_cumulative_average(
                    team_pts_vals, away_flag, group_index
                )
                team_df["season_home_points_allowed_avg"] = masked_cumulative_average(
                    opp_pts_vals, home_flag, group_index
                )
                team_df["season_away_points_allowed_avg"] = masked_cumulative_average(
                    opp_pts_vals, away_flag, group_index
                )

                away_consecutive = grouped.apply(
                    lambda g: compute_consecutive_true(g["_away_flag"])
                )
                team_df["consecutive_away_games"] = away_consecutive.reset_index(
                    level=group_cols, drop=True
                ).astype(float)

                ordered.drop(columns=["_home_flag", "_away_flag"], inplace=True, errors="ignore")
            else:
                for col in [
                    "home_win_loss_streak",
                    "away_win_loss_streak",
                    "season_home_games_played_total",
                    "season_away_games_played_total",
                    "season_home_wins_total",
                    "season_away_wins_total",
                    "season_home_win_pct",
                    "season_away_win_pct",
                    "home_win_pct",
                    "away_win_pct",
                    "season_home_margin_avg",
                    "season_away_margin_avg",
                    "season_home_off_rtg_avg",
                    "season_away_off_rtg_avg",
                    "season_home_def_rtg_avg",
                    "season_away_def_rtg_avg",
                    "season_home_points_avg",
                    "season_away_points_avg",
                    "season_home_points_allowed_avg",
                    "season_away_points_allowed_avg",
                    "consecutive_away_games",
                    "travel_distance_km",
                    "travel_km_last_7_days",
                    "travel_km_last_10_days",
                    "travel_km_last_14_days",
                ]:
                    team_df[col] = np.nan
        else:
            team_df["season_win_pct"] = np.nan
            team_df["win_loss_streak"] = np.nan
            for col in [
                "win_loss_streak",
                "home_win_loss_streak",
                "away_win_loss_streak",
                "season_home_games_played_total",
                "season_home_wins_total",
                "season_home_win_pct",
                "season_away_games_played_total",
                "season_away_wins_total",
                "season_away_win_pct",
                "home_win_pct",
                "away_win_pct",
                "days_since_last_game",
                "games_last_7_days",
                "games_last_10_days",
                "games_last_14_days",
                "season_home_points_avg",
                "season_home_points_allowed_avg",
                "season_away_points_avg",
                "season_away_points_allowed_avg",
                "season_home_margin_avg",
                "season_away_margin_avg",
                "season_home_off_rtg_avg",
                "season_home_def_rtg_avg",
                "season_away_off_rtg_avg",
                "season_away_def_rtg_avg",
                "consecutive_away_games",
                "travel_distance_km",
                "travel_km_last_7_days",
                "travel_km_last_10_days",
                "travel_km_last_14_days",
            ]:
                team_df[col] = np.nan
    else:
        team_df["season_win_pct"] = np.nan
        for col in [
            "win_loss_streak",
            "home_win_loss_streak",
            "away_win_loss_streak",
            "season_home_games_played_total",
            "season_home_wins_total",
            "season_home_win_pct",
            "season_away_games_played_total",
            "season_away_wins_total",
            "season_away_win_pct",
            "home_win_pct",
            "away_win_pct",
            "days_since_last_game",
            "games_last_7_days",
            "games_last_10_days",
            "games_last_14_days",
            "season_home_points_avg",
            "season_home_points_allowed_avg",
            "season_away_points_avg",
            "season_away_points_allowed_avg",
            "season_home_margin_avg",
            "season_away_margin_avg",
            "season_home_off_rtg_avg",
            "season_home_def_rtg_avg",
            "season_away_off_rtg_avg",
            "season_away_def_rtg_avg",
            "consecutive_away_games",
            "travel_distance_km",
            "travel_km_last_7_days",
            "travel_km_last_10_days",
            "travel_km_last_14_days",
        ]:
            team_df[col] = np.nan

    team_df = add_elo_ratings(team_df)

    team_df.drop(
        columns=[
            "pf_per_poss",
            "stl_per_poss",
            "blk_per_poss",
            "paint_share",
            "fb_share",
            "tov_points_share",
        ],
        inplace=True,
        errors="ignore",
    )

    if MISSING_LOCATION_KEYS:
        sample_missing = sorted(MISSING_LOCATION_KEYS)[:5]
        logging.info(
            "Missing arena coordinates for %d location keys; travel metrics set to NaN when unavailable: %s",
            len(MISSING_LOCATION_KEYS),
            sample_missing,
        )

    return team_df


def compute_player_group_features(group: pd.DataFrame) -> pd.Series:
    minutes = group["num_minutes_float"].astype(float)
    points = group.get("points", pd.Series(dtype=float)).fillna(0.0)
    assists = group.get("assists", pd.Series(dtype=float)).fillna(0.0)
    fgm = group.get("field_goals_made", pd.Series(dtype=float)).fillna(0.0)
    plus_minus = group.get("plus_minus_points", pd.Series(dtype=float)).fillna(0.0)
    off_reb = group.get("rebounds_offensive", pd.Series(dtype=float)).fillna(0.0)
    total_reb = group.get("rebounds_total", pd.Series(dtype=float)).fillna(0.0)

    total_points = points.sum()
    minutes_avg = minutes.mean()
    minutes_std = minutes.std()
    if pd.isna(minutes_std):
        minutes_std = 0.0
    points_std = points.std()
    if pd.isna(points_std):
        points_std = 0.0
    top_scorer_share = (
        float(points.max()) / (total_points + 1e-9) if len(points) else np.nan
    )
    bench_points_share = (
        points[minutes < 20].sum() / (total_points + 1e-9)
        if len(points)
        else np.nan
    )
    plusminus_avg = plus_minus.mean()
    plusminus_std = plus_minus.std()
    if pd.isna(plusminus_std):
        plusminus_std = 0.0
    assist_ratio = assists.sum() / (fgm.sum() + 1e-9)
    off_reb_share = off_reb.sum() / (total_reb.sum() + 1e-9)

    return pd.Series(
        {
            "ply_minutes_avg": minutes_avg,
            "ply_minutes_std": minutes_std,
            "ply_points_std": points_std,
            "ply_top_scorer_share": top_scorer_share,
            "ply_bench_points_share": bench_points_share,
            "ply_plusminus_avg": plusminus_avg,
            "ply_plusminus_std": plusminus_std,
            "ply_assist_ratio": assist_ratio,
            "ply_off_reb_share": off_reb_share,
        }
    )


def aggregate_player_features(player_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["game_id", "playerteam_city", "playerteam_name"]
    for col in required_cols:
        if col not in player_df.columns:
            raise KeyError(f"Missing column {col} in player statistics")
    group_cols = ["game_id", "playerteam_city", "playerteam_name"]
    agg = player_df.groupby(
        group_cols, group_keys=False
    ).apply(
        compute_player_group_features, include_groups=False
    )
    agg = agg.reset_index()
    agg = agg.rename(
        columns={
            "playerteam_city": "team_city",
            "playerteam_name": "team_name",
        }
    )
    return agg


def make_pre_features(
    team_features: pd.DataFrame, roll_windows: Sequence[int]
) -> pd.DataFrame:
    team_features = team_features.copy()
    team_features = team_features.sort_values(["team_id", "game_date", "game_id"])

    all_numeric_cols = [
        col
        for col in team_features.columns
        if col
        not in {
            "team_id",
            "game_id",
            "game_date",
            "team_city",
            "team_name",
            "opponent_team_id",
            "opponent_team_name",
            "opponent_team_city",
            "season",
        }
        and pd.api.types.is_numeric_dtype(team_features[col])
    ]

    allowed_numeric = [
        col
        for col in all_numeric_cols
        if col in TEAM_ALLOWED_BASE_COLUMNS and col not in TEAM_EXCLUDED_NUMERIC_FEATURES
    ]
    if not allowed_numeric:
        logging.warning("No allowed numeric columns available for rolling features")
        allowed_numeric = []

    observed_mask = (
        team_features[allowed_numeric].notna().any()
        if allowed_numeric
        else pd.Series(dtype=bool)
    )
    observed_cols = (
        observed_mask[observed_mask].index.tolist() if not observed_mask.empty else []
    )
    dropped_cols = sorted(set(allowed_numeric) - set(observed_cols))
    if dropped_cols:
        logging.info(
            "Skipping %d team feature bases with no data: %s",
            len(dropped_cols),
            dropped_cols[:10],
        )
    numeric_cols = observed_cols

    if not numeric_cols:
        logging.warning("No numeric columns survived filtering; returning identifiers only")
        return team_features[["game_id", "team_id", "game_date"]].copy()

    requested_windows = {int(window) for window in roll_windows if int(window) > 0}
    roll_windows = sorted(set(DEFAULT_ROLL_WINDOWS).union(requested_windows))
    if not roll_windows:
        roll_windows = list(DEFAULT_ROLL_WINDOWS)
    logging.info(
        "Using rolling windows (games): %s", ", ".join(str(w) for w in roll_windows)
    )

    pre_frames: list[pd.DataFrame] = []
    group_keys = ["team_id", "season"] if "season" in team_features.columns else ["team_id"]
    for _, group in team_features.groupby(group_keys, sort=False):
        group = group.sort_values(["game_date", "game_id"])
        base_cols = ["game_id", "team_id", "game_date"]
        if "season" in group.columns:
            base_cols.append("season")
        base = group[base_cols].copy().reset_index(drop=True)
        numeric_group = group[numeric_cols].reset_index(drop=True)
        shifted = numeric_group.shift(1)
        frames: list[pd.DataFrame] = [base]
        frames.append(
            pd.DataFrame({"tm_games_played_pre": np.arange(len(group))}, index=base.index)
        )

        if "elo_rating" in shifted.columns:
            frames.append(
                pd.DataFrame(
                    {"tm_elo_rating_pre": shifted["elo_rating"]}, index=base.index
                )
            )

        rolling_cache: dict[int, pd.DataFrame] = {}
        for window in roll_windows:
            rolling_cache[window] = shifted.rolling(window=window, min_periods=1).mean()

        ewm_cache: pd.DataFrame | None = None
        if EWM_COMBINATION_SPAN in rolling_cache:
            ewm_cache = shifted.ewm(
                span=EWM_COMBINATION_SPAN, adjust=False, min_periods=1
            ).mean()

        for window in roll_windows:
            rolling = rolling_cache[window]
            if window == EWM_COMBINATION_SPAN and ewm_cache is not None:
                combined = (
                    EWM_COMBINATION_WEIGHT * ewm_cache
                    + (1.0 - EWM_COMBINATION_WEIGHT) * rolling
                )
                data = combined
            else:
                data = rolling
            frames.append(
                pd.DataFrame(
                    {f"tm_{col}_r{window}_pre": data[col] for col in numeric_cols},
                    index=base.index,
                )
            )

        base_with_features = pd.concat(frames, axis=1)
        pre_frames.append(base_with_features)
    pre_df = pd.concat(pre_frames, ignore_index=True)
    pre_df.sort_values(["game_date", "game_id", "team_id"], inplace=True)
    pre_df.reset_index(drop=True, inplace=True)
    pre_df.drop(columns=["season"], inplace=True, errors="ignore")
    return pre_df


def drop_highly_correlated_pre_features(
    pre_df: pd.DataFrame, threshold: float = 0.999
) -> pd.DataFrame:
    """Remove ``tm_`` columns that are effectively duplicates.

    The experimental build intentionally limits itself to the user-specified
    metric list, but multiple rolling horizons can still produce
    near-identical series. Removing those columns keeps the exported feature
    set lean without violating the metric whitelist.
    """

    numeric_cols = [
        col
        for col in pre_df.columns
        if col.startswith("tm_") and pd.api.types.is_numeric_dtype(pre_df[col])
    ]
    if len(numeric_cols) < 2:
        return pre_df

    corr = pre_df[numeric_cols].corr().abs()
    if corr.empty:
        return pre_df

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = {
        column
        for column in upper.columns
        if upper[column].max(skipna=True) >= threshold
    }
    if to_drop:
        sample = sorted(to_drop)[:10]
        logging.info(
            "Dropping %d highly correlated pre-feature columns (>|%.3f|): %s",
            len(to_drop),
            threshold,
            sample,
        )
        pre_df = pre_df.drop(columns=list(to_drop), errors="ignore")
    return pre_df


def drop_highly_correlated_game_features(
    df: pd.DataFrame,
    target_col: str,
    feature_prefixes: Sequence[str] = ("home_tm_", "away_tm_", "delta_"),
    target_threshold: float = 0.999,
    feature_threshold: float = 0.999,
) -> pd.DataFrame:
    """Remove feature columns that mirror the target or each other."""

    # The high thresholds intentionally focus on near-duplicate signals so that
    # legitimate correlations remain available for modeling.

    if target_col not in df.columns:
        return df

    feature_cols = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in feature_prefixes)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(feature_cols) < 2:
        return df

    target = df[target_col]
    drop_cols: set[str] = set()

    target_drops: list[str] = []
    for col in feature_cols:
        corr_val = target.corr(df[col])
        if pd.notna(corr_val) and abs(float(corr_val)) >= target_threshold:
            drop_cols.add(col)
            target_drops.append(col)

    if target_drops:
        sample = sorted(target_drops)[:10]
        logging.info(
            "Dropping %d feature columns highly correlated with %s (>|%.3f|): %s",
            len(target_drops),
            target_col,
            target_threshold,
            sample,
        )

    remaining = [col for col in feature_cols if col not in drop_cols]
    if len(remaining) > 1:
        corr = df[remaining].corr().abs()
        if not corr.empty:
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            for column in upper.columns:
                if upper[column].max(skipna=True) >= feature_threshold:
                    drop_cols.add(column)

    if drop_cols:
        sample = sorted(drop_cols)[:10]
        logging.info(
            "Dropping %d highly correlated game-level feature columns: %s",
            len(drop_cols),
            sample,
        )
        df = df.drop(columns=list(drop_cols), errors="ignore")

    return df


def drop_highly_correlated_final_features(
    df: pd.DataFrame, threshold: float = 0.999
) -> pd.DataFrame:
    """Remove near-duplicate home/away/delta columns in the final table."""

    if len(df) < 2:
        return df

    candidate_cols = [
        col
        for col in df.columns
        if (
            col.startswith("home_tm_")
            or col.startswith("away_tm_")
            or col.startswith("delta_")
        )
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(candidate_cols) < 2:
        return df

    corr = df[candidate_cols].corr().abs()
    if corr.empty:
        return df

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = {
        column
        for column in upper.columns
        if upper[column].max(skipna=True) >= threshold
    }
    if to_drop:
        sample = sorted(to_drop)[:10]
        logging.info(
            "Dropping %d final-stage feature columns (>|%.3f| correlation): %s",
            len(to_drop),
            threshold,
            sample,
        )
        df = df.drop(columns=list(to_drop), errors="ignore")
    return df


def build_game_rows(
    games_df: pd.DataFrame,
    team_pre: pd.DataFrame,
    min_history: int,
) -> tuple[pd.DataFrame, float | None]:
    games_df = games_df.copy()
    games_df = games_df.sort_values(["game_date", "game_id"])
    games_df["home_margin"] = (
        games_df["home_score"].astype(float) - games_df["away_score"].astype(float)
    )
    games_df["label_home_margin"] = games_df["home_margin"]
    games_df["label_home_margin"] = games_df["label_home_margin"].astype(float)
    games_df["helper_home_win"] = (games_df["home_margin"] > 0).astype(int)

    home_pre = team_pre.drop(columns=["game_date"], errors="ignore").rename(
        columns={"team_id": "home_team_id"}
    )
    home_pre = home_pre.rename(
        columns={col: f"home_{col}" for col in home_pre.columns if col not in {"game_id", "home_team_id"}}
    )
    away_pre = team_pre.drop(columns=["game_date"], errors="ignore").rename(
        columns={"team_id": "away_team_id"}
    )
    away_pre = away_pre.rename(
        columns={col: f"away_{col}" for col in away_pre.columns if col not in {"game_id", "away_team_id"}}
    )

    merged = games_df.merge(home_pre, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_pre, on=["game_id", "away_team_id"], how="left")

    merged["home_is_home"] = 1
    merged["away_is_home"] = 0

    merged["season_recency_weight"] = 1.0

    if "game_date" in merged.columns and pd.api.types.is_datetime64_any_dtype(
        merged["game_date"]
    ):
        merged["season_month"] = merged["game_date"].dt.month.astype("Int64")
        month_dummies = pd.get_dummies(
            merged["season_month"], prefix="season_month", dtype=int
        )
        if not month_dummies.empty:
            month_dummies = month_dummies.reindex(
                columns=sorted(month_dummies.columns), fill_value=0
            )
            merged = pd.concat([merged, month_dummies], axis=1)

        max_date = merged["game_date"].max()
        if pd.notna(max_date):
            day_diffs = (max_date - merged["game_date"]).dt.days.astype(float)
            half_life_days = 365.0
            decay = np.log(2.0) / half_life_days
            recency_component = np.exp(-decay * day_diffs)
        else:
            recency_component = 1.0
    else:
        recency_component = 1.0

    if "season" in merged.columns:
        start_years = merged["season"].map(season_start_year)
        if start_years.notna().any():
            latest_year = start_years.max()
            if pd.notna(latest_year):
                half_life_seasons = 3.0
                decay = np.log(2.0) / half_life_seasons
                diffs = (latest_year - start_years).astype(float).clip(lower=0)
                weights = np.exp(-decay * diffs)
                merged.loc[start_years.notna(), "season_recency_weight"] = weights[
                    start_years.notna()
                ]

    if np.isscalar(recency_component):
        merged["recency_weight"] = float(recency_component)
    else:
        merged["recency_weight"] = np.asarray(recency_component, dtype=float)

    merged["season_recency_weight"] = merged["season_recency_weight"].astype(float)
    merged["recency_weight"] = (
        merged["recency_weight"].astype(float) * merged["season_recency_weight"]
    )

    if "home_tm_games_played_pre" in merged.columns:
        missing_home = merged["home_tm_games_played_pre"].isna().mean()
    else:
        missing_home = np.nan
    if "away_tm_games_played_pre" in merged.columns:
        missing_away = merged["away_tm_games_played_pre"].isna().mean()
    else:
        missing_away = np.nan
    home_pct = 100 * missing_home if not np.isnan(missing_home) else float("nan")
    away_pct = 100 * missing_away if not np.isnan(missing_away) else float("nan")
    logging.info(
        "Missing prefeatures — home: %.1f%%, away: %.1f%%",
        home_pct,
        away_pct,
    )

    kept_before = merged.shape[0]
    if min_history is not None and min_history > 0:
        merged = merged[
            (merged.get("home_tm_games_played_pre", 0) >= min_history)
            & (merged.get("away_tm_games_played_pre", 0) >= min_history)
        ]
        logging.info(
            "Dropped %d games due to min_history=%d",
            kept_before - merged.shape[0],
            min_history,
        )

    helper_rate: float | None = None
    if "helper_home_win" in merged.columns:
        helper_rate = merged["helper_home_win"].mean()

    feature_cols = [
        col
        for col in merged.columns
        if col.startswith("home_tm_") or col.startswith("away_tm_")
    ]
    delta_frames = {}
    for col in feature_cols:
        if not col.startswith("home_"):
            continue
        base = col[len("home_") :]
        other = f"away_{base}"
        if (
            other in merged.columns
            and pd.api.types.is_numeric_dtype(merged[col])
            and pd.api.types.is_numeric_dtype(merged[other])
        ):
            both_present = merged[[col, other]].dropna()
            if both_present.empty:
                continue
            delta_frames[f"delta_{base}"] = merged[col] - merged[other]
    if delta_frames:
        delta_df = pd.DataFrame(delta_frames, index=merged.index)
        merged = pd.concat([merged, delta_df], axis=1)

    relative_specs = [
        (
            "home_tm_win_pct_pre",
            "away_tm_win_pct_pre",
            "home_relative_victory_freq_diff",
            "away_relative_victory_freq_diff",
        ),
        (
            "home_tm_home_win_pct_pre",
            "away_tm_away_win_pct_pre",
            "home_home_relative_victory_freq_diff",
            "away_home_relative_victory_freq_diff",
        ),
        (
            "home_tm_away_win_pct_pre",
            "away_tm_home_win_pct_pre",
            "home_away_relative_victory_freq_diff",
            "away_away_relative_victory_freq_diff",
        ),
    ]
    for home_src, away_src, home_out, away_out in relative_specs:
        if home_src in merged.columns and away_src in merged.columns:
            diff = merged[home_src] - merged[away_src]
            merged[home_out] = diff
            merged[away_out] = -diff

    merged = drop_highly_correlated_game_features(
        merged,
        target_col="label_home_margin",
        feature_prefixes=("home_tm_", "away_tm_", "delta_"),
    )

    keep_cols = ["game_id", "game_date"]
    if "game_location" in merged.columns:
        keep_cols.append("game_location")
    keep_cols.extend([
        "home_team_id",
        "away_team_id",
    ])
    for extra in (
        "home_is_home",
        "away_is_home",
        "season_month",
        "season_recency_weight",
        "recency_weight",
    ):
        if extra in merged.columns:
            keep_cols.append(extra)
    month_dummy_cols = sorted(
        col for col in merged.columns if col.startswith("season_month_")
    )
    extra_cols = [
        col
        for col in merged.columns
        if col.startswith("home_tm_")
        or col.startswith("away_tm_")
        or col.startswith("delta_")
        or col.startswith("home_relative_")
        or col.startswith("away_relative_")
    ]
    ordered_cols = keep_cols + month_dummy_cols + sorted(extra_cols) + [
        "label_home_margin"
    ]
    merged = merged[ordered_cols]
    merged = merged.sort_values(["game_date", "game_id"])
    merged.reset_index(drop=True, inplace=True)
    return merged, helper_rate


def build_player_momentum_features(
    player_df: pd.DataFrame, team_df: pd.DataFrame, roll_windows: Sequence[int]
) -> pd.DataFrame:
    logging.info(
        "Skipping player-level momentum features in experimental build per requirements"
    )
    return pd.DataFrame(columns=["game_id", "team_id"])


def build_team_game_feature_table(
    team_df: pd.DataFrame, player_df: pd.DataFrame
) -> pd.DataFrame:
    if not player_df.empty:
        logging.info(
            "Player aggregation skipped in experimental build to honor team-only metric list"
        )
    return team_df.copy()


def limit_smoke(games_df: pd.DataFrame, team_df: pd.DataFrame, player_df: pd.DataFrame, smoke: int):
    if smoke is None:
        return games_df, team_df, player_df
    logging.info("Running in SMOKE mode, limiting to most recent %s games", smoke)
    ordered_games = games_df.sort_values("game_date", ascending=False)
    keep_game_ids = ordered_games["game_id"].drop_duplicates().head(smoke)
    games_df = ordered_games[ordered_games["game_id"].isin(keep_game_ids)].copy()
    team_df = team_df[team_df["game_id"].isin(keep_game_ids)].copy()
    player_df = player_df[player_df["game_id"].isin(keep_game_ids)].copy()
    return games_df, team_df, player_df


def write_outputs(
    df: pd.DataFrame, out_path: str, schema_path: str | None
) -> None:
    df.to_csv(out_path, index=False, float_format="%.6f")
    logging.info("Saved training table to %s", out_path)
    if schema_path:
        schema = build_schema_hint(df)
        schema.to_csv(schema_path, index=False)
        logging.info("Saved schema hint to %s", schema_path)


def build_schema_hint(df: pd.DataFrame) -> pd.DataFrame:
    roles = {}
    for col in df.columns:
        if col == "game_date":
            roles[col] = "time"
        elif col.startswith("label_"):
            roles[col] = "target"
        else:
            roles[col] = "feature"

    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            suggested = "TIMESTAMP"
        elif pd.api.types.is_integer_dtype(dtype):
            suggested = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            suggested = "FLOAT"
        else:
            suggested = "STRING"
        rows.append({"column_name": col, "suggested_type": suggested, "role": roles[col]})
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Vertex AI training table")
    parser.add_argument(
        "--games",
        default="Games.csv",
        help="Path to games CSV/TSV (defaults to 'Games.csv')",
    )
    parser.add_argument(
        "--team",
        default="TeamStatistics.csv",
        help="Path to team statistics CSV/TSV (defaults to 'TeamStatistics.csv')",
    )
    parser.add_argument(
        "--player",
        default="PlayerStatistics.csv",
        help="Path to player statistics CSV/TSV (defaults to 'PlayerStatistics.csv')",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--schema_out", help="Optional schema hint CSV path")
    parser.add_argument("--smoke", type=int, help="Limit to first N games for smoke test")
    parser.add_argument(
        "--min_history",
        type=int,
        default=3,
        help="Minimum number of prior games required",
    )
    parser.add_argument(
        "--roll_windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROLL_WINDOWS),
        help="Rolling window sizes in games (experimental build enforces 4 and 15)",
    )
    return parser.parse_args(argv)


def report(games_df: pd.DataFrame, out_df: pd.DataFrame) -> None:
    input_games = games_df["game_id"].nunique()
    output_games = out_df["game_id"].nunique()
    feature_cols = [
        col
        for col in out_df.columns
        if col.startswith("home_") or col.startswith("away_") or col.startswith("delta_")
    ]
    logging.info(
        "Games in input: %s | Games in output: %s", input_games, output_games
    )
    logging.info("Number of feature columns: %s", len(feature_cols))
    if not out_df.empty:
        logging.info("Date range: %s -> %s", out_df["game_date"].min(), out_df["game_date"].max())
        logging.info("Sample rows:\n%s", out_df.head())


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    games_df = load_table(args.games)
    games_df = games_df.rename(
        columns={
            "hometeam_id": "home_team_id",
            "awayteam_id": "away_team_id",
            "hometeam_name": "home_team_name",
            "awayteam_name": "away_team_name",
            "hometeam_city": "home_team_city",
            "awayteam_city": "away_team_city",
        }
    )
    games_df = games_df.rename(
        columns={
            "homescore": "home_score",
            "awayscore": "away_score",
            "home_score_total": "home_score",
            "away_score_total": "away_score",
        }
    )
    team_df = load_table(args.team)
    player_df = load_table(args.player)

    team_df = drop_preseason_rows(team_df, "team")
    player_df = drop_preseason_rows(player_df, "player")
    games_df = drop_preseason_rows(games_df, "games")

    relevant_sources: list[pd.Series] = []
    if "game_id" in team_df.columns:
        relevant_sources.append(team_df["game_id"].dropna())
    if "game_id" in player_df.columns:
        relevant_sources.append(player_df["game_id"].dropna())
    if relevant_sources:
        relevant_ids = pd.Index(pd.unique(pd.concat(relevant_sources, ignore_index=True)))
        if not relevant_ids.empty:
            before = len(games_df)
            games_df = games_df.loc[games_df["game_id"].isin(relevant_ids)].copy()
            removed = before - len(games_df)
            if removed:
                logging.info(
                    "Filtered %d schedule-only rows not present in box-score tables",
                    removed,
                )

    games_df = backfill_game_dates(
        games_df,
        (
            ("team", team_df),
            ("player", player_df),
        ),
    )

    games_df = add_season_column(games_df)
    games_df = filter_min_season(games_df, "games")
    bad_dates = games_df["game_date"].isna()
    if bad_dates.any():
        drop_count = int(bad_dates.sum())
        drop_pct = 100 * drop_count / max(len(games_df), 1)
        logging.warning(
            "Dropping %d rows with NaT game_date from games_df (%.2f%% of file)",
            drop_count,
            drop_pct,
        )
        games_df = games_df.loc[~bad_dates].copy()
    same_team = games_df["home_team_id"].eq(games_df["away_team_id"]).fillna(False)
    if same_team.any():
        logging.warning(
            "Dropping %d self-match rows (home==away)", same_team.sum()
        )
        games_df = games_df.loc[~same_team].copy()
    if games_df["game_id"].duplicated().any():
        logging.warning(
            "games_df has duplicated game_id; deduping by first occurrence"
        )
        games_df = (
            games_df.sort_values(["game_date", "game_id"])
            .drop_duplicates("game_id", keep="first")
        )
    required = {
        "game_id",
        "game_date",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    }
    missing = required - set(games_df.columns)
    if missing:
        raise ValueError(
            f"games file missing columns after normalization/rename: {missing}"
        )
    location_series = None
    if "home_team_city" in games_df.columns:
        location_series = games_df["home_team_city"]
    elif "arena_id" in games_df.columns:
        location_series = games_df["arena_id"].astype(str)
    if location_series is None:
        games_df["game_location"] = "unknown"
    else:
        games_df["game_location"] = location_series.fillna("unknown").astype(str)
        games_df.loc[
            games_df["game_location"].str.lower().isin({"nan", "none", ""}),
            "game_location",
        ] = "unknown"
    games_df["game_era"] = games_df["game_date"].map(determine_game_era)
    team_df = add_season_column(team_df)
    team_df = filter_min_season(team_df, "team")
    player_df = add_season_column(player_df)
    player_df = filter_min_season(player_df, "player")

    logging.info(
        "Loaded shapes — games=%s, team=%s, player=%s",
        games_df.shape,
        team_df.shape,
        player_df.shape,
    )

    for df in (games_df, team_df, player_df):
        for col in ("team_id", "opponent_team_id", "home_team_id", "away_team_id"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    games_df, team_df, player_df = limit_smoke(
        games_df, team_df, player_df, args.smoke
    )

    team_df = clean_team_table(team_df)
    player_df = clean_player_table(player_df)

    team_df = build_team_features(team_df)
    team_game_features = build_team_game_feature_table(team_df, player_df)

    requested_windows = {int(x) for x in args.roll_windows if int(x) > 0}
    enforced_windows = set(DEFAULT_ROLL_WINDOWS)
    if requested_windows and requested_windows != enforced_windows:
        logging.info(
            "Experimental build enforces rolling windows %s; ignoring custom request %s",
            DEFAULT_ROLL_WINDOWS,
            sorted(requested_windows),
        )
    roll_windows = tuple(sorted(enforced_windows))
    logging.info("Using rolling windows (games): %s", roll_windows)
    player_momentum_features = build_player_momentum_features(
        player_df, team_df, roll_windows
    )
    pre_features = make_pre_features(team_game_features, roll_windows)
    if not player_momentum_features.empty:
        pre_features = pre_features.merge(
            player_momentum_features, on=["game_id", "team_id"], how="left"
        )

    h2h_cols = [
        col
        for col in (
            "season_h2h_games_prior",
            "season_h2h_wins_prior",
            "season_h2h_win_pct_prior",
            "season_h2h_margin_avg_prior",
        )
        if col in team_game_features.columns
    ]
    if h2h_cols:
        h2h_frame = (
            team_game_features[["game_id", "team_id", *h2h_cols]]
            .drop_duplicates(subset=["game_id", "team_id"], keep="first")
            .copy()
        )
        h2h_frame = h2h_frame.rename(
            columns={col: f"tm_{col}_pre" for col in h2h_cols}
        )
        pre_features = pre_features.merge(h2h_frame, on=["game_id", "team_id"], how="left")

    pre_features = drop_highly_correlated_pre_features(pre_features)

    data_feature_cols = [
        col for col in pre_features.columns if col.startswith("tm_")
    ]
    drop_missing_data = [
        col for col in data_feature_cols if not pre_features[col].notna().any()
    ]
    if drop_missing_data:
        logging.info(
            "Dropping %d pre-feature columns with no historical data: %s",
            len(drop_missing_data),
            drop_missing_data[:10],
        )
        pre_features = pre_features.drop(columns=drop_missing_data)

    dups = pre_features.duplicated(subset=["game_id", "team_id"], keep=False)
    if dups.any():
        logging.warning(
            "Found %d duplicate (game_id, team_id) rows in pre_features; deduping by first",
            dups.sum(),
        )
        pre_features = (
            pre_features.sort_values(["game_date", "game_id", "team_id"])
            .drop_duplicates(subset=["game_id", "team_id"], keep="first")
        )

    assert not pre_features.duplicated(["game_id", "team_id"]).any(), (
        "pre_features still has duplicate (game_id, team_id)"
    )

    logging.info(
        "Prefeatures shape: %s | Player momentum shape: %s",
        pre_features.shape,
        player_momentum_features.shape,
    )

    out_df, helper_rate = build_game_rows(games_df, pre_features, args.min_history)

    if "label_home_margin" not in out_df.columns:
        raise ValueError(
            "Output table must include label_home_margin as the predictive target"
        )
    extra_label_cols = [
        col for col in out_df.columns if col.startswith("label_") and col != "label_home_margin"
    ]
    if extra_label_cols:
        logging.info(
            "Dropping helper label columns to keep Vertex target focused on home margin: %s",
            extra_label_cols,
        )
        out_df = out_df.drop(columns=extra_label_cols, errors="ignore")

    feature_subset = out_df.filter(like="tm_")
    if not feature_subset.empty:
        null_rate = feature_subset.isna().mean().mean()
        logging.info("Average feature NaN rate: %.2f%%", 100 * null_rate)
    else:
        logging.info("Average feature NaN rate: N/A (no tm_ features present)")

    if helper_rate is not None and not np.isnan(helper_rate):
        logging.info("Helper home-win positive rate (for reference): %.1f%%", 100 * helper_rate)
    else:
        logging.info("Helper home-win positive rate: N/A")

    if len(out_df) > 1:
        out_df = drop_highly_correlated_final_features(out_df)
        nunique = out_df.nunique(dropna=False)
        const_cols = [
            col
            for col in nunique[nunique <= 1].index.tolist()
            if col
            not in {
                "game_id",
                "game_date",
                "home_team_id",
                "away_team_id",
            }
            and not col.startswith("label_")
        ]
        if const_cols:
            logging.info(
                "Dropping %d constant cols: %s",
                len(const_cols),
                const_cols[:10],
            )
            out_df = out_df.drop(columns=const_cols, errors="ignore")

    for col in ("home_team_id", "away_team_id"):
        if col in out_df.columns:
            out_df[col] = out_df[col].astype("Int64")

    assert out_df["game_id"].is_unique, "Output must contain one row per game"

    logging.info("Final training table shape: %s", out_df.shape)

    write_outputs(out_df, args.out, args.schema_out)
    report(games_df, out_df)


if __name__ == "__main__":
    main()

