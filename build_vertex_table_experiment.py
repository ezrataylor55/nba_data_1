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


MIN_SEASON_START_YEAR = 2014
DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (3, 8, 15)

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
TEAM_ALLOWED_BASE_COLUMNS: set[str] = {
    "games_played",
    "win",
    "loss",
    "win_pct",
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
    "fouls_drawn",
    "blocks_against",
    "plus_minus_points",
    "off_rtg",
    "def_rtg",
    "net_rtg",
    "ast_ratio",
    "ast_to_ratio",
    "oreb_pct",
    "dreb_pct",
    "reb_pct",
    "tov_pct",
    "efg",
    "ts_pct",
    "pace",
    "poss",
    "ft_rate",
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
    "opp_pf",
    "opp_pts",
    "opp_plus_minus",
    "opp_pts_off_tov",
    "opp_pts_2nd_chance",
    "opp_pts_fast_break",
    "opp_pts_paint",
}

TIME_DECAY_FEATURE_CANDIDATES: set[str] = {
    "win",
    "loss",
    "win_pct",
    "minutes",
    "team_score",
    "field_goals_made",
    "field_goals_attempted",
    "three_pointers_made",
    "three_pointers_attempted",
    "free_throws_made",
    "free_throws_attempted",
    "rebounds_offensive",
    "rebounds_defensive",
    "rebounds_total",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "fouls_personal",
    "plus_minus_points",
    "off_rtg",
    "def_rtg",
    "net_rtg",
    "ast_ratio",
    "ast_to_ratio",
    "oreb_pct",
    "dreb_pct",
    "reb_pct",
    "tov_pct",
    "efg",
    "ts_pct",
    "pace",
    "poss",
    "ft_rate",
    "opp_pts",
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
    "opp_pf",
    "opp_pts_off_tov",
    "opp_pts_2nd_chance",
    "opp_pts_fast_break",
    "opp_pts_paint",
}.intersection(TEAM_ALLOWED_BASE_COLUMNS)


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
        "points_in_the_paint",
        "points_fast_break",
        "points_from_turnovers",
        "points_second_chance",
        "fouls_personal",
        "fouls_drawn",
        "fouls_personal_drawn",
        "rebounds_total",
        "plus_minus_points",
        "num_minutes",
        "blocks_against",
    ]
    for col in base_numeric:
        if col not in team_df.columns:
            team_df[col] = np.nan
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
    if "fouls_personal_drawn" in team_df.columns and "fouls_drawn" not in team_df.columns:
        team_df["fouls_drawn"] = pd.to_numeric(
            team_df["fouls_personal_drawn"], errors="coerce"
        )
    if "fouls_drawn" in team_df.columns:
        team_df["fouls_drawn"] = pd.to_numeric(
            team_df["fouls_drawn"], errors="coerce"
        )
    if "blocks_against" in team_df.columns:
        team_df["blocks_against"] = pd.to_numeric(
            team_df["blocks_against"], errors="coerce"
        )
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
            "plus_minus_points",
            "efg",
            "ts_pct",
            "ft_rate",
            "tov_pct",
            "points_from_turnovers",
            "points_second_chance",
            "points_fast_break",
            "points_in_the_paint",
            "pace",
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
            "plus_minus_points": "opp_plus_minus",
            "efg": "opp_efg_pct",
            "ts_pct": "opp_ts_pct",
            "ft_rate": "opp_ft_rate",
            "tov_pct": "opp_tov_pct",
            "points_from_turnovers": "opp_pts_off_tov",
            "points_second_chance": "opp_pts_2nd_chance",
            "points_fast_break": "opp_pts_fast_break",
            "points_in_the_paint": "opp_pts_paint",
            "pace": "opp_pace",
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
    team_df["opp_dreb_pct"] = safe_div(
        team_df["opp_dreb"], team_df["opp_dreb"] + team_df["rebounds_offensive"]
    )
    team_df["opp_reb_pct"] = safe_div(
        team_df["opp_reb"], team_df["opp_reb"] + team_df["rebounds_total"]
    )

    off_ratio = safe_div(team_df["team_score"], team_df["poss"]).astype(float)
    def_ratio = safe_div(
        team_df["opponent_score"], team_df["opp_poss"]
    ).astype(float)
    team_df["off_rtg"] = 100.0 * off_ratio
    team_df["def_rtg"] = 100.0 * def_ratio
    team_df["margin"] = team_df["team_score"] - team_df["opponent_score"]
    team_df["net_rtg"] = team_df["off_rtg"] - team_df["def_rtg"]

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

    roll_windows = sorted({int(window) for window in roll_windows if int(window) > 0})
    if not roll_windows:
        roll_windows = list(DEFAULT_ROLL_WINDOWS)

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
        expanding = shifted.expanding(min_periods=1).mean()

        frames: list[pd.DataFrame] = [base]
        frames.append(
            pd.DataFrame({"tm_games_played_pre": np.arange(len(group))}, index=base.index)
        )

        expanding_dict = {f"tm_{col}_exp_pre": expanding[col] for col in numeric_cols}
        frames.append(pd.DataFrame(expanding_dict, index=base.index))

        for window in roll_windows:
            rolling = shifted.rolling(window=window, min_periods=1).mean()
            rolling_dict = {
                f"tm_{col}_r{window}_pre": rolling[col] for col in numeric_cols
            }
            frames.append(pd.DataFrame(rolling_dict, index=base.index))

        ewm_cols = [col for col in numeric_cols if col in TIME_DECAY_FEATURE_CANDIDATES]
        if ewm_cols:
            ewm_dict: dict[str, pd.Series] = {}
            for halflife in (5, 15):
                ewm_frame = shifted[ewm_cols].ewm(
                    halflife=halflife, min_periods=1, adjust=False
                ).mean()
                for col in ewm_cols:
                    ewm_dict[f"tm_{col}_ewm_h{halflife}_pre"] = ewm_frame[col]
            if ewm_dict:
                frames.append(pd.DataFrame(ewm_dict, index=base.index))

        base_with_features = pd.concat(frames, axis=1)
        pre_frames.append(base_with_features)
    pre_df = pd.concat(pre_frames, ignore_index=True)
    pre_df.sort_values(["game_date", "game_id", "team_id"], inplace=True)
    pre_df.reset_index(drop=True, inplace=True)
    pre_df.drop(columns=["season"], inplace=True, errors="ignore")
    return pre_df


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
            delta_frames[f"delta_{base}"] = merged[col] - merged[other]
    if delta_frames:
        delta_df = pd.DataFrame(delta_frames, index=merged.index)
        merged = pd.concat([merged, delta_df], axis=1)

    keep_cols = ["game_id", "game_date"]
    if "game_location" in merged.columns:
        keep_cols.append("game_location")
    keep_cols.extend([
        "home_team_id",
        "away_team_id",
    ])
    extra_cols = [
        col
        for col in merged.columns
        if col.startswith("home_tm_")
        or col.startswith("away_tm_")
        or col.startswith("delta_")
    ]
    ordered_cols = keep_cols + sorted(extra_cols) + ["label_home_margin"]
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
        help="Rolling window sizes in games (experimental build enforces 3, 8, 15)",
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

