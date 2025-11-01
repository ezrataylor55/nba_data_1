#!/usr/bin/env python3
"""Build a Vertex AI AutoML ready dataset with season-to-date and rolling features."""

import argparse
import logging
from datetime import datetime
import math
import re
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Ensure columns exist by adding them in a single batch assignment."""

    ordered_missing = [
        col for col in dict.fromkeys(columns) if col not in df.columns
    ]
    if ordered_missing:
        df.loc[:, ordered_missing] = np.nan

# Alias mapping for resolving columns with flexible headers.
ALIASES: Dict[str, List[str]] = {
    "GameID": ["GameID", "GAME_ID", "game_id", "gid"],
    "HomeID": ["HomeID", "HOME_TEAM_ID", "home_id", "home_team_id"],
    "AwayID": ["AwayID", "AWAY_TEAM_ID", "away_id", "away_team_id"],
    "GameDate": ["GameDate", "GAME_DATE", "date"],
    "Location": ["Location", "VENUE", "Arena", "Site"],
    "TeamCity": ["teamCity", "TEAM_CITY", "City", "TEAMCITY"],
    "OpponentCity": [
        "opponentTeamCity",
        "OPPONENT_TEAM_CITY",
        "opponentCity",
        "OPPCITY",
    ],
    "HomeScore": ["HomeScore", "HOME_PTS", "home_pts", "HomePoints"],
    "AwayScore": ["AwayScore", "AWAY_PTS", "away_pts", "AwayPoints"],
    "WinnerStr": ["Winner", "WINNER", "Result", "result", "HomeWin"],
    "TeamID": ["TeamID", "teamId", "TEAM_ID", "team_id"],
    "OpponentID": ["OpponentID", "opponentTeamId", "opponent_team_id", "opponentTeamID"],
    "IsHome": ["home", "HOME", "is_home", "Home"],
    "TeamScore": ["teamScore", "TeamScore", "team_pts", "PTS"],
    "OpponentScore": ["opponentScore", "OpponentScore", "opp_pts", "OPP_PTS"],
    "SeasonWins": ["seasonWins", "SeasonWins", "wins", "Wins"],
    "SeasonLosses": ["seasonLosses", "SeasonLosses", "losses", "Losses"],
    "TeamWinFlag": ["win", "WIN", "teamWin", "TeamWin"],
    "HOME_SELF_Elo": ["HomeElo", "HOME_ELO", "home_elo"],
    "HOME_SELF_EFG%": ["Home_eFG%", "HOME_EFG_PCT", "home_efg_pct", "HomeEFG"],
    "HOME_SELF_TO%": ["Home_TOV%", "HOME_TOV_PCT", "home_to_pct", "HomeTO%"],
    "HOME_SELF_OR%": ["Home_ORB%", "HOME_OR_PCT", "home_or_pct", "HomeORB%"],
    "HOME_SELF_FTR": ["Home_FTR", "HOME_FTR", "home_ftr"],
    "HOME_SELF_DR%": ["Home_DRB%", "HOME_DR_PCT", "home_dr_pct", "HomeDRB%"],
    "HOME_SELF_OffRating": [
        "HomeOffRating",
        "HOME_OFF_RATING",
        "home_off_rating",
        "Home_Off_Rating",
        "HOME_SELF_OFF_RTG",
        "home_off_rtg",
    ],
    "HOME_SELF_DefRating": [
        "HomeDefRating",
        "HOME_DEF_RATING",
        "home_def_rating",
        "Home_Def_Rating",
        "HOME_SELF_DEF_RTG",
        "home_def_rtg",
    ],
    "HOME_SELF_NetRating": [
        "HomeNetRating",
        "HOME_NET_RATING",
        "home_net_rating",
        "Home_Net_Rating",
        "HOME_SELF_NET_RTG",
        "home_net_rtg",
    ],
    "HOME_OPP_EFG%": ["HomeOpp_eFG%", "HOME_OPP_EFG_PCT", "home_opp_efg_pct"],
    "HOME_OPP_TO%": ["HomeOpp_TOV%", "HOME_OPP_TOV_PCT", "home_opp_to_pct"],
    "HOME_OPP_FTR": ["HomeOpp_FTR", "HOME_OPP_FTR", "home_opp_ftr"],
    "HOME_OPP_OffRating": [
        "HomeOppOffRating",
        "HOME_OPP_OFF_RATING",
        "home_opp_off_rating",
        "HomeOpp_Off_Rating",
        "HOME_OPP_OFF_RTG",
        "home_opp_off_rtg",
    ],
    "HOME_OPP_DefRating": [
        "HomeOppDefRating",
        "HOME_OPP_DEF_RATING",
        "home_opp_def_rating",
        "HomeOpp_Def_Rating",
        "HOME_OPP_DEF_RTG",
        "home_opp_def_rtg",
    ],
    "HOME_OPP_NetRating": [
        "HomeOppNetRating",
        "HOME_OPP_NET_RATING",
        "home_opp_net_rating",
        "HomeOpp_Net_Rating",
        "HOME_OPP_NET_RTG",
        "home_opp_net_rtg",
    ],
    "AWAY_SELF_Elo": ["AwayElo", "AWAY_ELO", "away_elo"],
    "AWAY_SELF_EFG%": ["Away_eFG%", "AWAY_EFG_PCT", "away_efg_pct", "AwayEFG"],
    "AWAY_SELF_TO%": ["Away_TOV%", "AWAY_TOV_PCT", "away_to_pct", "AwayTO%"],
    "AWAY_SELF_OR%": ["Away_ORB%", "AWAY_OR_PCT", "away_or_pct", "AwayORB%"],
    "AWAY_SELF_FTR": ["Away_FTR", "AWAY_FTR", "away_ftr"],
    "AWAY_SELF_DR%": ["Away_DRB%", "AWAY_DR_PCT", "away_dr_pct", "AwayDRB%"],
    "AWAY_SELF_OffRating": [
        "AwayOffRating",
        "AWAY_OFF_RATING",
        "away_off_rating",
        "Away_Off_Rating",
        "AWAY_SELF_OFF_RTG",
        "away_off_rtg",
    ],
    "AWAY_SELF_DefRating": [
        "AwayDefRating",
        "AWAY_DEF_RATING",
        "away_def_rating",
        "Away_Def_Rating",
        "AWAY_SELF_DEF_RTG",
        "away_def_rtg",
    ],
    "AWAY_SELF_NetRating": [
        "AwayNetRating",
        "AWAY_NET_RATING",
        "away_net_rating",
        "Away_Net_Rating",
        "AWAY_SELF_NET_RTG",
        "away_net_rtg",
    ],
    "AWAY_OPP_EFG%": ["AwayOpp_eFG%", "AWAY_OPP_EFG_PCT", "away_opp_efg_pct"],
    "AWAY_OPP_TO%": ["AwayOpp_TOV%", "AWAY_OPP_TOV_PCT", "away_opp_to_pct"],
    "AWAY_OPP_FTR": ["AwayOpp_FTR", "AWAY_OPP_FTR", "away_opp_ftr"],
    "AWAY_OPP_OffRating": [
        "AwayOppOffRating",
        "AWAY_OPP_OFF_RATING",
        "away_opp_off_rating",
        "AwayOpp_Off_Rating",
        "AWAY_OPP_OFF_RTG",
        "away_opp_off_rtg",
    ],
    "AWAY_OPP_DefRating": [
        "AwayOppDefRating",
        "AWAY_OPP_DEF_RATING",
        "away_opp_def_rating",
        "AwayOpp_Def_Rating",
        "AWAY_OPP_DEF_RTG",
        "away_opp_def_rtg",
    ],
    "AWAY_OPP_NetRating": [
        "AwayOppNetRating",
        "AWAY_OPP_NET_RATING",
        "away_opp_net_rating",
        "AwayOpp_Net_Rating",
        "AWAY_OPP_NET_RTG",
        "away_opp_net_rtg",
    ],
}

SELF_METRICS: Sequence[str] = (
    "Elo",
    "EFG%",
    "TO%",
    "OR%",
    "FTR",
    "DR%",
    "OffRating",
    "DefRating",
    "NetRating",
    "Pace",
)
OPP_METRICS: Sequence[str] = (
    "EFG%",
    "TO%",
    "FTR",
    "OffRating",
    "DefRating",
    "NetRating",
    "Pace",
)

REST_FEATURES: Sequence[str] = (
    "days_since_last_game",
    "is_back_to_back",
    "three_in_four",
    "four_in_six",
    "days_until_next_game",
    "home_game_streak",
    "away_game_streak",
)
TRAVEL_BASE_FEATURES: Sequence[str] = (
    "travel_miles_since_last_game",
    "travel_timezone_change",
    "travel_altitude_change",
    "travel_fatigue_score",
)
TRAVEL_AGG_FEATURES: Sequence[str] = ("travel_fatigue_score",)
SCHEDULE_DIFF_PREFIX = "strength_of_schedule_diff"
HEAD_TO_HEAD_COLUMN = "head_to_head_index"
WIN_RATE_SUFFIXES: Sequence[str] = ("season", "r4", "r10", "r20")
WIN_RATE_DIFF_MAP = {
    suffix: f"HomeAwayWinRateDelta_{suffix}" for suffix in WIN_RATE_SUFFIXES
}
WIN_RATE_DIFF_COLS: Sequence[str] = tuple(WIN_RATE_DIFF_MAP.values())

RAW_STAT_ALIASES: Dict[str, List[str]] = {
    "FGM": [
        "FGM",
        "FGM_MADE",
        "FGM_MADE_TOTAL",
        "FIELD_GOALS_MADE",
        "FGM_TOTAL",
        "fieldGoalsMade",
    ],
    "FG3M": [
        "FG3M",
        "FG3_MADE",
        "3PM",
        "FG3M_TOTAL",
        "THREE_PM",
        "threePointersMade",
    ],
    "FGA": [
        "FGA",
        "FIELD_GOALS_ATTEMPTED",
        "FGA_TOTAL",
        "fieldGoalsAttempted",
    ],
    "FTA": [
        "FTA",
        "FREE_THROW_ATTEMPTS",
        "FTA_TOTAL",
        "freeThrowsAttempted",
    ],
    "TOV": ["TOV", "TURNOVERS", "TO", "TOV_TOTAL", "turnovers"],
    "ORB": [
        "ORB",
        "OFFENSIVE_REBOUNDS",
        "OREB",
        "ORB_TOTAL",
        "reboundsOffensive",
    ],
    "DRB": [
        "DRB",
        "DEFENSIVE_REBOUNDS",
        "DREB",
        "DRB_TOTAL",
        "reboundsDefensive",
    ],
}

TEAM_SIDE_PREFIXES = ("HOME", "AWAY")

PERCENT_PATTERN = re.compile(r"^\s*([+-]?[0-9]*\.?[0-9]+)\s*%\s*$")


# Elo decay factors to gradually dampen older results and heavily regress between seasons.
ELO_PER_GAME_DECAY = 0.85
ELO_OFFSEASON_DECAY = 0.3
ELO_BLOWOUT_DECAY_START = 20.0
ELO_BLOWOUT_DECAY_LAMBDA = 0.05
ELO_BLOWOUT_WEIGHT_FLOOR = 0.6
LEAGUE_AVG_POINTS_PER_POSSESSION = 1.05

TRAVEL_DISTANCE_NORMALIZER = 500.0
TRAVEL_TIMEZONE_WEIGHT = 0.75
TRAVEL_ALTITUDE_NORMALIZER = 1000.0


def _to_str_id(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        try:
            text = str(value).strip()
        except Exception:  # pragma: no cover - defensive
            return None
        return text if text else None


TEAM_TRAVEL_INFO: Dict[str, Dict[str, float]] = {
    "1610612737": {"lat": 33.7573, "lon": -84.3963, "tz": -5.0, "alt": 1050.0},
    "1610612738": {"lat": 42.3663, "lon": -71.0622, "tz": -5.0, "alt": 20.0},
    "1610612739": {"lat": 41.4965, "lon": -81.6882, "tz": -5.0, "alt": 650.0},
    "1610612740": {"lat": 29.9490, "lon": -90.0821, "tz": -6.0, "alt": 10.0},
    "1610612741": {"lat": 41.8807, "lon": -87.6742, "tz": -6.0, "alt": 600.0},
    "1610612742": {"lat": 32.7905, "lon": -96.8104, "tz": -6.0, "alt": 430.0},
    "1610612743": {"lat": 39.7487, "lon": -105.0077, "tz": -7.0, "alt": 5280.0},
    "1610612744": {"lat": 37.7680, "lon": -122.3877, "tz": -8.0, "alt": 30.0},
    "1610612745": {"lat": 29.7508, "lon": -95.3621, "tz": -6.0, "alt": 50.0},
    "1610612746": {"lat": 34.0430, "lon": -118.2673, "tz": -8.0, "alt": 305.0},
    "1610612747": {"lat": 34.0430, "lon": -118.2673, "tz": -8.0, "alt": 305.0},
    "1610612748": {"lat": 25.7814, "lon": -80.1870, "tz": -5.0, "alt": 10.0},
    "1610612749": {"lat": 43.0451, "lon": -87.9172, "tz": -6.0, "alt": 600.0},
    "1610612750": {"lat": 44.9795, "lon": -93.2760, "tz": -6.0, "alt": 840.0},
    "1610612751": {"lat": 40.6826, "lon": -73.9754, "tz": -5.0, "alt": 50.0},
    "1610612752": {"lat": 40.7505, "lon": -73.9934, "tz": -5.0, "alt": 60.0},
    "1610612753": {"lat": 28.5392, "lon": -81.3839, "tz": -5.0, "alt": 82.0},
    "1610612754": {"lat": 39.7639, "lon": -86.1555, "tz": -5.0, "alt": 715.0},
    "1610612755": {"lat": 39.9012, "lon": -75.1719, "tz": -5.0, "alt": 30.0},
    "1610612756": {"lat": 33.4457, "lon": -112.0712, "tz": -7.0, "alt": 1070.0},
    "1610612757": {"lat": 45.5316, "lon": -122.6668, "tz": -8.0, "alt": 50.0},
    "1610612758": {"lat": 38.5804, "lon": -121.4997, "tz": -8.0, "alt": 30.0},
    "1610612759": {"lat": 29.4271, "lon": -98.4375, "tz": -6.0, "alt": 650.0},
    "1610612760": {"lat": 35.4634, "lon": -97.5151, "tz": -6.0, "alt": 1200.0},
    "1610612761": {"lat": 43.6435, "lon": -79.3791, "tz": -5.0, "alt": 250.0},
    "1610612762": {"lat": 40.7683, "lon": -111.9011, "tz": -7.0, "alt": 4225.0},
    "1610612763": {"lat": 35.1382, "lon": -90.0506, "tz": -6.0, "alt": 337.0},
    "1610612764": {"lat": 38.8981, "lon": -77.0209, "tz": -5.0, "alt": 80.0},
    "1610612765": {"lat": 42.6978, "lon": -83.2399, "tz": -5.0, "alt": 980.0},
    "1610612766": {"lat": 35.2251, "lon": -80.8392, "tz": -5.0, "alt": 750.0},
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in miles between two coordinates."""

    radius_miles = 3958.8
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
    return radius_miles * c


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--out_csv", required=True, help="Path to write the output CSV")
    parser.add_argument(
        "--min_season",
        default="2010-2011",
        help="Minimum season to include in YYYY-YYYY format (inclusive)",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        type=int,
        default=[4, 10, 20],
        help="Rolling window sizes for features",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke-test mode to limit rows for a lightweight end-to-end run",
    )
    return parser.parse_args()


def normalize_pct(value: object, column: str) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return np.nan
        match = PERCENT_PATTERN.match(stripped)
        if match:
            try:
                return float(match.group(1)) / 100.0
            except ValueError:
                logging.warning("Unable to parse percentage '%s' in %s", value, column)
                return np.nan
        cleaned = stripped.replace(",", "")
        try:
            numeric = float(cleaned)
        except ValueError:
            logging.warning("Non-numeric percentage '%s' in %s", value, column)
            return np.nan
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            logging.warning("Non-numeric percentage '%s' in %s", value, column)
            return np.nan
    if numeric > 1.0000001:
        if numeric <= 100:
            logging.debug(
                "Scaling value %.3f in column %s assuming percentage in 0-100 scale",
                numeric,
                column,
            )
            return numeric / 100.0
        logging.warning(
            "Value %.3f in column %s exceeds expected bounds for a percentage",
            numeric,
            column,
        )
        return np.nan
    if numeric < 0:
        logging.warning("Negative percentage %.3f in column %s", numeric, column)
        return np.nan
    return numeric


def parse_game_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, (int, float)):
        try:
            if value > 10_000:
                parsed = datetime.fromordinal(datetime(1899, 12, 30).toordinal() + int(value))
            else:
                parsed = datetime.strptime(str(int(value)), "%Y%m%d")
            return pd.Timestamp(parsed)
        except Exception:  # noqa: BLE001
            return pd.NaT
    value_str = str(value).strip()
    if not value_str:
        return pd.NaT
    truncated = re.split(r"[T\s]", value_str)[0]
    if len(truncated) >= 8:
        parsed = pd.to_datetime(truncated, errors="coerce")
        if pd.notna(parsed):
            return parsed.tz_localize(None) if parsed.tzinfo else parsed
    parsed_full = pd.to_datetime(value_str, errors="coerce")
    if pd.notna(parsed_full):
        return parsed_full.tz_localize(None) if parsed_full.tzinfo else parsed_full
    return pd.NaT


def infer_season(game_date: datetime) -> str:
    year = game_date.year
    if game_date.month < 7:
        start_year = year - 1
    else:
        start_year = year
    return f"{start_year}-{start_year + 1}"


def season_start_year(season: str) -> int:
    try:
        return int(season.split("-")[0])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid season format: {season}") from None


def _normalize_column_name(name: str) -> str:
    """Normalize a column name for fuzzy alias matching."""

    return re.sub(r"[^a-z0-9]", "", name.lower())


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to their canonical aliases using flexible matching."""

    rename_map: Dict[str, str] = {}
    normalized_columns: Dict[str, str] = {
        _normalize_column_name(column): column for column in df.columns
    }

    for canonical, aliases in ALIASES.items():
        if canonical in df.columns:
            continue
        candidates = [canonical] + aliases
        for candidate in candidates:
            normalized = _normalize_column_name(candidate)
            if normalized in normalized_columns:
                source_column = normalized_columns[normalized]
                if source_column != canonical:
                    rename_map[source_column] = canonical
                break

    if rename_map:
        logging.info("Resolved column aliases: %s", rename_map)
        df = df.rename(columns=rename_map)

    for canonical in ALIASES:
        if canonical not in df.columns:
            df[canonical] = np.nan

    return df


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["GameID", "HomeID", "AwayID", "GameDate"]
    missing = [col for col in required if df[col].isna().all()]
    if missing:
        raise ValueError(
            "Missing required columns after alias resolution: " + ", ".join(missing)
        )


def _normalize_home_flag(value: object) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value) == 1
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"home", "h", "true", "t", "1", "yes"}:
            return True
        if normalized in {"away", "a", "false", "f", "0", "no"}:
            return False
    if isinstance(value, bool):
        return bool(value)
    return None


def _series_with_suffix(df: pd.DataFrame, base: str, suffix: str) -> pd.Series:
    column = f"{base}_{suffix}"
    if column in df.columns:
        return coerce_numeric(df[column])
    return pd.Series(np.nan, index=df.index, dtype=float)


def _stat_series_with_suffix(df: pd.DataFrame, alias_key: str, suffix: str) -> pd.Series:
    candidates = [alias_key] + RAW_STAT_ALIASES.get(alias_key, [])
    for base in candidates:
        column = f"{base}_{suffix}"
        if column in df.columns:
            return coerce_numeric(df[column])
    return pd.Series(np.nan, index=df.index, dtype=float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.mask((denominator == 0) | denominator.isna())


def apply_pre_game_elo(
    df: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
) -> None:
    """Assign pre-game Elo ratings with intra-season decay and offseason regression."""

    if df.empty:
        return

    def _apply_decay(
        rating: float,
        last_season_start: Optional[int],
        current_season_start: int,
    ) -> float:
        if last_season_start is None:
            return rating
        if current_season_start != last_season_start:
            seasons_apart = max(1, current_season_start - last_season_start)
            offseason_multiplier = ELO_OFFSEASON_DECAY ** seasons_apart
            return base_rating + (rating - base_rating) * offseason_multiplier
        return base_rating + (rating - base_rating) * ELO_PER_GAME_DECAY

    order_cols = ["SeasonStartYear", "GameDate_dt", "GameID", "HomeID", "AwayID"]
    ordered_index = df.sort_values(order_cols).index
    home_pre = pd.Series(base_rating, index=df.index, dtype=float)
    away_pre = pd.Series(base_rating, index=df.index, dtype=float)
    ratings: Dict[str, Dict[str, object]] = {}

    for idx in ordered_index:
        season_start = int(df.at[idx, "SeasonStartYear"])
        home_id = str(df.at[idx, "HomeID"])
        away_id = str(df.at[idx, "AwayID"])

        home_state = ratings.setdefault(
            home_id, {"rating": base_rating, "last_season_start": None}
        )
        away_state = ratings.setdefault(
            away_id, {"rating": base_rating, "last_season_start": None}
        )

        home_rating = float(
            _apply_decay(
                home_state["rating"],
                home_state["last_season_start"],
                season_start,
            )
        )
        away_rating = float(
            _apply_decay(
                away_state["rating"],
                away_state["last_season_start"],
                season_start,
            )
        )
        home_state["rating"] = home_rating
        away_state["rating"] = away_rating

        home_pre.at[idx] = home_rating
        away_pre.at[idx] = away_rating

        outcome = df.at[idx, "HomeTeamWin"]
        if pd.isna(outcome):
            home_state["last_season_start"] = season_start
            away_state["last_season_start"] = season_start
            continue
        outcome = float(outcome)
        expected_home = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))
        margin_multiplier = 1.0
        home_score = df.at[idx, "HomeScore"] if "HomeScore" in df.columns else np.nan
        away_score = df.at[idx, "AwayScore"] if "AwayScore" in df.columns else np.nan
        if pd.notna(home_score) and pd.notna(away_score):
            try:
                margin = abs(float(home_score) - float(away_score))
            except (TypeError, ValueError):
                margin = 0.0
            if margin > ELO_BLOWOUT_DECAY_START:
                excess = margin - ELO_BLOWOUT_DECAY_START
                decay = math.exp(-ELO_BLOWOUT_DECAY_LAMBDA * excess)
                margin_multiplier = max(ELO_BLOWOUT_WEIGHT_FLOOR, decay)
        adjustment = k_factor * margin_multiplier
        home_rating += adjustment * (outcome - expected_home)
        away_rating += adjustment * ((1.0 - outcome) - (1.0 - expected_home))
        home_state["rating"] = home_rating
        home_state["last_season_start"] = season_start
        away_state["rating"] = away_rating
        away_state["last_season_start"] = season_start

    df.loc[ordered_index, "HOME_SELF_Elo"] = home_pre.loc[ordered_index].values
    df.loc[ordered_index, "AWAY_SELF_Elo"] = away_pre.loc[ordered_index].values


def _compute_elo_from_record(
    wins: pd.Series,
    losses: pd.Series,
    opponent_wins: Optional[pd.Series] = None,
    opponent_losses: Optional[pd.Series] = None,
) -> pd.Series:
    wins = coerce_numeric(wins).fillna(0)
    losses = coerce_numeric(losses).fillna(0)
    total_games = wins + losses

    adjusted_wins = wins + 1.5
    adjusted_losses = losses + 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        win_strength = adjusted_wins / (adjusted_wins + adjusted_losses)
    win_strength = win_strength.clip(lower=1e-6, upper=1 - 1e-6)
    logistic_component = np.log(win_strength / (1 - win_strength))
    elo_from_record = 1500 + 173.7178 * logistic_component

    stabilization = (total_games / (total_games + 10)).fillna(0)
    elo = 1500 + (elo_from_record - 1500) * stabilization

    record_balance = wins - losses
    with np.errstate(divide="ignore", invalid="ignore"):
        balance_ratio = record_balance / total_games
    balance_ratio = balance_ratio.replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    elo += 50 * balance_ratio.fillna(0) * stabilization

    if opponent_wins is not None and opponent_losses is not None:
        opponent_wins = coerce_numeric(opponent_wins).fillna(0)
        opponent_losses = coerce_numeric(opponent_losses).fillna(0)
        opponent_games = opponent_wins + opponent_losses
        adjusted_opp_wins = opponent_wins + 1.5
        adjusted_opp_losses = opponent_losses + 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            opponent_strength = adjusted_opp_wins / (adjusted_opp_wins + adjusted_opp_losses)
        opponent_strength = opponent_strength.clip(lower=1e-6, upper=1 - 1e-6)
        opponent_component = np.log(opponent_strength / (1 - opponent_strength))
        schedule_adjustment = 25 * opponent_component * stabilization
        schedule_adjustment = schedule_adjustment.mask(opponent_games <= 0)
        elo += schedule_adjustment.fillna(0)

    baseline = 1500.0
    elo = elo.mask(total_games.isna(), baseline)
    elo = elo.where(total_games > 0, baseline)
    return elo


def maybe_convert_team_view_to_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """Transform team-level rows into game-level records when needed."""

    if df["HomeID"].notna().any() and df["AwayID"].notna().any():
        return df
    if "TeamID" not in df.columns or df["TeamID"].isna().all():
        return df
    if "OpponentID" not in df.columns or df["OpponentID"].isna().all():
        return df
    if "IsHome" not in df.columns or df["IsHome"].isna().all():
        return df

    normalized_home = df["IsHome"].apply(_normalize_home_flag)
    valid_mask = normalized_home.notna()
    if not valid_mask.all():
        logging.warning(
            "Dropping %d rows with indeterminable home/away flag during team-level conversion",
            (~valid_mask).sum(),
        )
    working = df[valid_mask].copy()
    working["IsHome"] = normalized_home[valid_mask]

    id_mask = working["TeamID"].notna() & working["OpponentID"].notna()
    if not id_mask.all():
        logging.warning(
            "Dropping %d rows missing team/opponent identifiers during team-level conversion",
            (~id_mask).sum(),
        )
    working = working[id_mask].copy()

    gid_mask = working["GameID"].notna()
    if not gid_mask.all():
        logging.warning(
            "Dropping %d rows with missing GameID during team-level conversion",
            (~gid_mask).sum(),
        )
    working = working[gid_mask].copy()

    if working.empty:
        raise ValueError("No rows remain after filtering team-level data for conversion")

    home_rows = working[working["IsHome"]].copy()
    away_rows = working[~working["IsHome"]].copy()
    if home_rows.empty or away_rows.empty:
        raise ValueError("Unable to locate both home and away rows for conversion")

    home_rows = home_rows.drop_duplicates(subset="GameID", keep="last")
    away_rows = away_rows.drop_duplicates(subset="GameID", keep="last")

    try:
        merged = home_rows.merge(
            away_rows,
            on="GameID",
            suffixes=("_home", "_away"),
            how="inner",
            validate="one_to_one",
        )
    except ValueError:
        logging.warning(
            "Encountered duplicate team rows; falling back to permissive merge during conversion",
        )
        merged = home_rows.merge(
            away_rows,
            on="GameID",
            suffixes=("_home", "_away"),
            how="inner",
        )

    if merged.empty:
        raise ValueError(
            "Failed to pair team-level rows into games; verify input file has both teams per game"
        )

    home_scores = coerce_numeric(merged.get("TeamScore_home", np.nan))
    away_scores = coerce_numeric(merged.get("TeamScore_away", np.nan))

    if "OpponentScore_home" in merged.columns:
        opponent_scores_home = coerce_numeric(merged["OpponentScore_home"])
        mismatch_mask = (
            home_scores.notna()
            & opponent_scores_home.notna()
            & (opponent_scores_home != away_scores)
        )
        if mismatch_mask.any():
            logging.warning(
                "Detected %d games where opponentScore did not match paired teamScore",
                mismatch_mask.sum(),
            )

    home_fgm = _stat_series_with_suffix(merged, "FGM", "home")
    home_fg3m = _stat_series_with_suffix(merged, "FG3M", "home")
    home_fga = _stat_series_with_suffix(merged, "FGA", "home")
    away_fgm = _stat_series_with_suffix(merged, "FGM", "away")
    away_fg3m = _stat_series_with_suffix(merged, "FG3M", "away")
    away_fga = _stat_series_with_suffix(merged, "FGA", "away")

    home_turnovers = _stat_series_with_suffix(merged, "TOV", "home")
    away_turnovers = _stat_series_with_suffix(merged, "TOV", "away")
    home_fta = _stat_series_with_suffix(merged, "FTA", "home")
    away_fta = _stat_series_with_suffix(merged, "FTA", "away")

    home_orb = _stat_series_with_suffix(merged, "ORB", "home")
    away_orb = _stat_series_with_suffix(merged, "ORB", "away")
    home_drb = _stat_series_with_suffix(merged, "DRB", "home")
    away_drb = _stat_series_with_suffix(merged, "DRB", "away")

    home_elo = _compute_elo_from_record(
        _series_with_suffix(merged, "SeasonWins", "home"),
        _series_with_suffix(merged, "SeasonLosses", "home"),
        opponent_wins=_series_with_suffix(merged, "SeasonWins", "away"),
        opponent_losses=_series_with_suffix(merged, "SeasonLosses", "away"),
    )
    away_elo = _compute_elo_from_record(
        _series_with_suffix(merged, "SeasonWins", "away"),
        _series_with_suffix(merged, "SeasonLosses", "away"),
        opponent_wins=_series_with_suffix(merged, "SeasonWins", "home"),
        opponent_losses=_series_with_suffix(merged, "SeasonLosses", "home"),
    )

    home_efg = _safe_ratio(home_fgm + 0.5 * home_fg3m, home_fga)
    away_efg = _safe_ratio(away_fgm + 0.5 * away_fg3m, away_fga)

    home_to = _safe_ratio(
        home_turnovers,
        home_fga + 0.44 * home_fta + home_turnovers,
    )
    away_to = _safe_ratio(
        away_turnovers,
        away_fga + 0.44 * away_fta + away_turnovers,
    )

    home_ftr = _safe_ratio(home_fta, home_fga)
    away_ftr = _safe_ratio(away_fta, away_fga)

    home_or = _safe_ratio(home_orb, home_orb + away_drb)
    home_dr = _safe_ratio(home_drb, home_drb + away_orb)
    away_or = _safe_ratio(away_orb, away_orb + home_drb)
    away_dr = _safe_ratio(away_drb, away_drb + home_orb)

    with np.errstate(divide="ignore", invalid="ignore"):
        home_possessions = home_fga - home_orb + home_turnovers + 0.44 * home_fta
        away_possessions = away_fga - away_orb + away_turnovers + 0.44 * away_fta
    home_possessions = home_possessions.replace(0, np.nan)
    away_possessions = away_possessions.replace(0, np.nan)
    home_pace = home_possessions.copy()
    away_pace = away_possessions.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        home_off_rating = 100.0 * home_scores / home_possessions
        away_off_rating = 100.0 * away_scores / away_possessions
        home_def_rating = 100.0 * away_scores / away_possessions
        away_def_rating = 100.0 * home_scores / home_possessions
    home_net_rating = home_off_rating - home_def_rating
    away_net_rating = away_off_rating - away_def_rating

    game_dates = merged.get("GameDate_home")
    if game_dates is None:
        game_dates = merged.get("GameDate_away")
    else:
        game_dates = game_dates.fillna(merged.get("GameDate_away"))

    location_series = merged.get("TeamCity_home")
    if location_series is None:
        location_series = merged.get("Location_home")
    if location_series is None:
        location_series = pd.Series("Unknown", index=merged.index, dtype=object)
    else:
        location_series = location_series.copy()
    location_series = location_series.astype(str)
    location_series = location_series.str.strip()
    location_series = location_series.where(
        ~location_series.str.lower().isin({"", "nan", "none"}),
        np.nan,
    )
    if "Location_home" in merged.columns:
        fallback_location = merged["Location_home"].astype(str).str.strip()
        fallback_location = fallback_location.where(
            ~fallback_location.str.lower().isin({"", "nan", "none"}),
            np.nan,
        )
        location_series = location_series.fillna(fallback_location)
    location_series = location_series.fillna("Unknown")

    converted = pd.DataFrame(
        {
            "GameID": merged["GameID"],
            "GameDate": game_dates,
            "Location": location_series,
            "HomeID": merged.get("TeamID_home"),
            "AwayID": merged.get("TeamID_away"),
            "HomeScore": home_scores,
            "AwayScore": away_scores,
            "HOME_SELF_Elo": home_elo,
            "AWAY_SELF_Elo": away_elo,
            "HOME_OPP_Elo": away_elo,
            "AWAY_OPP_Elo": home_elo,
            "HOME_SELF_EFG%": home_efg,
            "AWAY_SELF_EFG%": away_efg,
            "HOME_SELF_TO%": home_to,
            "AWAY_SELF_TO%": away_to,
            "HOME_SELF_OR%": home_or,
            "AWAY_SELF_OR%": away_or,
            "HOME_SELF_FTR": home_ftr,
            "AWAY_SELF_FTR": away_ftr,
            "HOME_SELF_DR%": home_dr,
            "AWAY_SELF_DR%": away_dr,
            "HOME_SELF_OffRating": home_off_rating,
            "HOME_SELF_DefRating": home_def_rating,
            "HOME_SELF_NetRating": home_net_rating,
            "HOME_SELF_Pace": home_pace,
            "AWAY_SELF_OffRating": away_off_rating,
            "AWAY_SELF_DefRating": away_def_rating,
            "AWAY_SELF_NetRating": away_net_rating,
            "AWAY_SELF_Pace": away_pace,
            "HOME_OPP_EFG%": away_efg,
            "AWAY_OPP_EFG%": home_efg,
            "HOME_OPP_TO%": away_to,
            "AWAY_OPP_TO%": home_to,
            "HOME_OPP_FTR": away_ftr,
            "AWAY_OPP_FTR": home_ftr,
            "HOME_OPP_OffRating": away_off_rating,
            "HOME_OPP_DefRating": away_def_rating,
            "HOME_OPP_NetRating": away_net_rating,
            "AWAY_OPP_OffRating": home_off_rating,
            "AWAY_OPP_DefRating": home_def_rating,
            "AWAY_OPP_NetRating": home_net_rating,
            "HOME_OPP_Pace": away_pace,
            "AWAY_OPP_Pace": home_pace,
        }
    )

    converted["GameDate"] = converted["GameDate"].astype(str).str.strip()
    converted.loc[converted["GameDate"].isin({"", "nan", "NaT", "None"}), "GameDate"] = np.nan
    converted.loc[converted["GameDate"].notna(), "GameDate"] = converted.loc[
        converted["GameDate"].notna(), "GameDate"
    ].str.replace(r"[T\s].*", "", regex=True)

    if "TeamWinFlag_home" in merged.columns:
        win_flag = merged["TeamWinFlag_home"].astype(str).str.strip().str.lower()
        winner = pd.Series(np.nan, index=converted.index, dtype=object)
        winner = winner.mask(win_flag.isin({"1", "true", "t", "win", "w"}), "Home")
        winner = winner.mask(win_flag.isin({"0", "false", "f", "loss", "l"}), "Away")
        converted["WinnerStr"] = winner

    logging.info(
        "Transformed %d team-level rows into %d game-level records",
        len(df),
        len(converted),
    )

    return converted


def determine_home_win(row: pd.Series) -> Optional[int]:
    home_score = row.get("HomeScore")
    away_score = row.get("AwayScore")
    if pd.notna(home_score) and pd.notna(away_score):
        try:
            h = float(home_score)
            a = float(away_score)
        except (TypeError, ValueError):
            h = a = None
        if h is not None and a is not None:
            if h > a:
                return 1
            if h < a:
                return 0
    winner = row.get("WinnerStr")
    if pd.isna(winner):
        return None
    if isinstance(winner, str):
        normalized = winner.strip().lower()
        if not normalized:
            return None
        if normalized in {"home", "h", "1", "true", "t", "win", "homewin"}:
            return 1
        if normalized in {"away", "a", "0", "false", "f", "loss", "awaywin"}:
            return 0
    if isinstance(winner, (int, float)):
        if int(winner) == 1:
            return 1
        if int(winner) == 0:
            return 0
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prefixed_candidates(prefix: str, alias: str) -> List[str]:
    variations = {
        prefix,
        prefix.lower(),
        prefix.capitalize(),
        prefix.title(),
    }
    alias_variations = {
        alias,
        alias.lower(),
        alias.upper(),
        alias.capitalize(),
    }
    candidates: List[str] = []
    for pref in variations:
        for ali in alias_variations:
            candidates.append(f"{pref}_{ali}")
            candidates.append(f"{pref}{ali}")
    return candidates


def find_raw_column(df: pd.DataFrame, prefix: str, alias_key: str) -> Optional[str]:
    aliases = RAW_STAT_ALIASES.get(alias_key, [])
    for alias in aliases:
        for candidate in prefixed_candidates(prefix, alias):
            if candidate in df.columns:
                return candidate
    return None


def try_derive_rate_columns(df: pd.DataFrame) -> None:
    for side in TEAM_SIDE_PREFIXES:
        opp = "AWAY" if side == "HOME" else "HOME"
        for metric in ("OffRating", "DefRating", "NetRating"):
            self_col = f"{side}_SELF_{metric}"
            if self_col in df.columns:
                df[self_col] = coerce_numeric(df[self_col])
            opp_col = f"{side}_OPP_{metric}"
            if opp_col in df.columns:
                df[opp_col] = coerce_numeric(df[opp_col])
        efg_col = f"{side}_SELF_EFG%"
        if df[efg_col].isna().all():
            fgm_col = find_raw_column(df, side, "FGM")
            fg3m_col = find_raw_column(df, side, "FG3M")
            fga_col = find_raw_column(df, side, "FGA")
            if fgm_col and fg3m_col and fga_col:
                numer = coerce_numeric(df[fgm_col]) + 0.5 * coerce_numeric(df[fg3m_col])
                denom = coerce_numeric(df[fga_col])
                with np.errstate(divide="ignore", invalid="ignore"):
                    computed = numer / denom
                df[efg_col] = df[efg_col].fillna(computed)
            else:
                logging.warning(
                    "Unable to derive %s due to missing components (FGM, FG3M, FGA)",
                    efg_col,
                )
        to_col = f"{side}_SELF_TO%"
        if df[to_col].isna().all():
            tov_col = find_raw_column(df, side, "TOV")
            fga_col = find_raw_column(df, side, "FGA")
            fta_col = find_raw_column(df, side, "FTA")
            if tov_col and fga_col and fta_col:
                tov = coerce_numeric(df[tov_col])
                fga = coerce_numeric(df[fga_col])
                fta = coerce_numeric(df[fta_col])
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = fga + 0.44 * fta + tov
                    computed = tov / denom
                df[to_col] = df[to_col].fillna(computed)
            else:
                logging.warning(
                    "Unable to derive %s due to missing components (TOV, FGA, FTA)",
                    to_col,
                )
        or_col = f"{side}_SELF_OR%"
        if df[or_col].isna().all():
            orb_col = find_raw_column(df, side, "ORB")
            opp_drb_col = find_raw_column(df, opp, "DRB")
            if orb_col and opp_drb_col:
                orb = coerce_numeric(df[orb_col])
                opp_drb = coerce_numeric(df[opp_drb_col])
                with np.errstate(divide="ignore", invalid="ignore"):
                    computed = orb / (orb + opp_drb)
                df[or_col] = df[or_col].fillna(computed)
            else:
                logging.warning(
                    "Unable to derive %s due to missing components (ORB, opponent DRB)",
                    or_col,
                )
        dr_col = f"{side}_SELF_DR%"
        if df[dr_col].isna().all():
            drb_col = find_raw_column(df, side, "DRB")
            opp_orb_col = find_raw_column(df, opp, "ORB")
            if drb_col and opp_orb_col:
                drb = coerce_numeric(df[drb_col])
                opp_orb = coerce_numeric(df[opp_orb_col])
                with np.errstate(divide="ignore", invalid="ignore"):
                    computed = drb / (drb + opp_orb)
                df[dr_col] = df[dr_col].fillna(computed)
            else:
                logging.warning(
                    "Unable to derive %s due to missing components (DRB, opponent ORB)",
                    dr_col,
                )
        ftr_col = f"{side}_SELF_FTR"
        if df[ftr_col].isna().all():
            fta_col = find_raw_column(df, side, "FTA")
            fga_col = find_raw_column(df, side, "FGA")
            if fta_col and fga_col:
                fta = coerce_numeric(df[fta_col])
                fga = coerce_numeric(df[fga_col])
                with np.errstate(divide="ignore", invalid="ignore"):
                    computed = fta / fga
                df[ftr_col] = df[ftr_col].fillna(computed)
            else:
                logging.warning(
                    "Unable to derive %s due to missing components (FTA, FGA)",
                    ftr_col,
                )
        off_col = f"{side}_SELF_OffRating"
        def_col = f"{side}_SELF_DefRating"
        need_off = off_col in df.columns and df[off_col].isna().all()
        need_def = def_col in df.columns and df[def_col].isna().all()
        if need_off or need_def:
            fga_col = find_raw_column(df, side, "FGA")
            fta_col = find_raw_column(df, side, "FTA")
            tov_col = find_raw_column(df, side, "TOV")
            orb_col = find_raw_column(df, side, "ORB")
            possessions = None
            if fga_col and fta_col and tov_col:
                fga = coerce_numeric(df[fga_col])
                fta = coerce_numeric(df[fta_col])
                tov = coerce_numeric(df[tov_col])
                orb = (
                    coerce_numeric(df[orb_col])
                    if orb_col
                    else pd.Series(0.0, index=df.index)
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    possessions = fga + 0.44 * fta + tov - orb
                possessions = possessions.replace(0, np.nan)
            opp_possessions = None
            if need_def:
                opp_fga_col = find_raw_column(df, opp, "FGA")
                opp_fta_col = find_raw_column(df, opp, "FTA")
                opp_tov_col = find_raw_column(df, opp, "TOV")
                opp_orb_col = find_raw_column(df, opp, "ORB")
                if opp_fga_col and opp_fta_col and opp_tov_col:
                    opp_fga = coerce_numeric(df[opp_fga_col])
                    opp_fta = coerce_numeric(df[opp_fta_col])
                    opp_tov = coerce_numeric(df[opp_tov_col])
                    opp_orb = (
                        coerce_numeric(df[opp_orb_col])
                        if opp_orb_col
                        else pd.Series(0.0, index=df.index)
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        opp_possessions = opp_fga + 0.44 * opp_fta + opp_tov - opp_orb
                    opp_possessions = opp_possessions.replace(0, np.nan)
                if opp_possessions is None:
                    opp_possessions = possessions
            if need_off and possessions is not None:
                score_col = "HomeScore" if side == "HOME" else "AwayScore"
                if score_col in df.columns:
                    points = coerce_numeric(df[score_col])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        off_rating = 100.0 * points / possessions
                    df[off_col] = df[off_col].fillna(off_rating)
            if need_def and opp_possessions is not None:
                opp_score_col = "AwayScore" if side == "HOME" else "HomeScore"
                if opp_score_col in df.columns:
                    opp_points = coerce_numeric(df[opp_score_col])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        def_rating = 100.0 * opp_points / opp_possessions
                    df[def_col] = df[def_col].fillna(def_rating)
        fallback_off_needed = off_col in df.columns and df[off_col].isna().all()
        fallback_def_needed = def_col in df.columns and df[def_col].isna().all()
        if (fallback_off_needed or fallback_def_needed) and {
            "HomeScore",
            "AwayScore",
        }.issubset(df.columns):
            score_col = "HomeScore" if side == "HOME" else "AwayScore"
            opp_score_col = "AwayScore" if side == "HOME" else "HomeScore"
            points = coerce_numeric(df[score_col])
            opp_points = coerce_numeric(df[opp_score_col])
            total_points = points + opp_points
            with np.errstate(divide="ignore", invalid="ignore"):
                possessions_est = total_points / (
                    2.0 * LEAGUE_AVG_POINTS_PER_POSSESSION
                )
            possessions_est = possessions_est.replace(0, np.nan)
            if fallback_off_needed:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fallback_off = 100.0 * points / possessions_est
                df[off_col] = df[off_col].fillna(fallback_off)
            if fallback_def_needed:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fallback_def = 100.0 * opp_points / possessions_est
                df[def_col] = df[def_col].fillna(fallback_def)
        net_col = f"{side}_SELF_NetRating"
        if net_col in df.columns:
            if off_col in df.columns and def_col in df.columns:
                off_rating = coerce_numeric(df[off_col])
                def_rating = coerce_numeric(df[def_col])
                df[net_col] = df[net_col].fillna(off_rating - def_rating)
        opp_net_col = f"{side}_OPP_NetRating"
        if opp_net_col in df.columns:
            opp_off_col = f"{side}_OPP_OffRating"
            opp_def_col = f"{side}_OPP_DefRating"
            if opp_off_col in df.columns and opp_def_col in df.columns:
                opp_off = coerce_numeric(df[opp_off_col])
                opp_def = coerce_numeric(df[opp_def_col])
                df[opp_net_col] = df[opp_net_col].fillna(opp_off - opp_def)
        opp_ftr_col = f"{side}_OPP_FTR"
        if opp_ftr_col in df.columns and df[opp_ftr_col].isna().all():
            # Attempt to compute opponent FTR from opponent stats when available.
            opp_fta_col = find_raw_column(df, opp, "FTA")
            opp_fga_col = find_raw_column(df, opp, "FGA")
            if opp_fta_col and opp_fga_col:
                opp_fta = coerce_numeric(df[opp_fta_col])
                opp_fga = coerce_numeric(df[opp_fga_col])
                with np.errstate(divide="ignore", invalid="ignore"):
                    computed = opp_fta / opp_fga
                df[opp_ftr_col] = df[opp_ftr_col].fillna(computed)
            else:
                logging.warning(
                    "Unable to derive %s due to missing opponent components (FTA, FGA)",
                    opp_ftr_col,
                )


def normalize_percent_columns(df: pd.DataFrame) -> None:
    percent_cols = [col for col in df.columns if "%" in col]
    for col in percent_cols:
        df[col] = df[col].apply(lambda x: normalize_pct(x, col))


def ensure_opponent_metric_parity(df: pd.DataFrame) -> None:
    for metric in OPP_METRICS:
        home_self = f"HOME_SELF_{metric}"
        away_self = f"AWAY_SELF_{metric}"
        home_opp = f"HOME_OPP_{metric}"
        away_opp = f"AWAY_OPP_{metric}"
        if home_opp in df.columns and away_self in df.columns:
            df[home_opp] = df[home_opp].fillna(df[away_self])
        if away_opp in df.columns and home_self in df.columns:
            df[away_opp] = df[away_opp].fillna(df[home_self])


def to_snake_case(name: str) -> str:
    name = name.replace("%", "_pct")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


def compute_team_long_frame(
    df: pd.DataFrame, windows: Sequence[int]
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        game_id = row["GameID"]
        game_date = row["GameDate_dt"]
        season = row["Season"]
        home_id = row["HomeID"]
        away_id = row["AwayID"]
        home_win = float(row["HomeTeamWin"])
        for is_home, team_id, opp_id, prefix in (
            (True, home_id, away_id, "HOME"),
            (False, away_id, home_id, "AWAY"),
        ):
            team_score = (
                row.get("HomeScore") if is_home else row.get("AwayScore")
            )
            opp_score = (
                row.get("AwayScore") if is_home else row.get("HomeScore")
            )
            record: Dict[str, object] = {
                "OriginalIndex": idx,
                "GameID": game_id,
                "TeamID": team_id,
                "OpponentID": opp_id,
                "is_home": is_home,
                "GameDate": game_date,
                "Season": season,
                "SeasonStartYear": row["SeasonStartYear"],
                "TeamWin": home_win if is_home else (1.0 - home_win if pd.notna(home_win) else np.nan),
                "TeamScore": team_score,
                "OpponentScore": opp_score,
                "LocationTeamID": team_id if is_home else opp_id,
            }
            for metric in SELF_METRICS:
                record[f"SELF_{metric}"] = row.get(f"{prefix}_SELF_{metric}")
            for metric in OPP_METRICS:
                record[f"OPP_{metric}"] = row.get(f"{prefix}_OPP_{metric}")
            records.append(record)
    team_df = pd.DataFrame(records)
    team_df.sort_values(
        by=["TeamID", "Season", "GameDate", "OriginalIndex"],
        inplace=True,
    )
    grouped = team_df.groupby(["TeamID", "Season"], sort=False)
    metrics_to_process = [f"SELF_{m}" for m in SELF_METRICS] + [f"OPP_{m}" for m in OPP_METRICS]
    for metric in metrics_to_process:
        if metric not in team_df.columns:
            continue
        team_df[f"{metric}_season"] = grouped[metric].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        for window in windows:
            team_df[f"{metric}_r{window}"] = grouped[metric].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
            )
    for feature in list(REST_FEATURES) + list(TRAVEL_BASE_FEATURES):
        team_df[feature] = np.nan
    missing_travel_ids: Set[str] = set()
    per_team = team_df.groupby("TeamID", sort=False)
    for _, group in per_team:
        if group.empty:
            continue
        group_sorted = group.sort_values(
            by=["GameDate", "OriginalIndex"], kind="mergesort"
        )
        prev_dates = group_sorted["GameDate"].shift(1)
        next_dates = group_sorted["GameDate"].shift(-1)
        days_since = (group_sorted["GameDate"] - prev_dates).dt.days.astype(float)
        days_until = (next_dates - group_sorted["GameDate"]).dt.days.astype(float)
        team_df.loc[group_sorted.index, "days_since_last_game"] = days_since
        team_df.loc[group_sorted.index, "days_until_next_game"] = days_until
        b2b = np.where(days_since.notna(), (days_since <= 1).astype(float), np.nan)
        team_df.loc[group_sorted.index, "is_back_to_back"] = b2b
        dates = group_sorted["GameDate"].values.astype("datetime64[D]")
        counts_4 = np.zeros(len(dates), dtype=int)
        counts_6 = np.zeros(len(dates), dtype=int)
        start4 = 0
        start6 = 0
        for idx_date, current in enumerate(dates):
            while current - dates[start4] > np.timedelta64(3, "D"):
                start4 += 1
            while current - dates[start6] > np.timedelta64(5, "D"):
                start6 += 1
            counts_4[idx_date] = idx_date - start4 + 1
            counts_6[idx_date] = idx_date - start6 + 1
        three_in_four = (counts_4 >= 3).astype(float)
        four_in_six = (counts_6 >= 4).astype(float)
        team_df.loc[group_sorted.index, "three_in_four"] = three_in_four
        team_df.loc[group_sorted.index, "four_in_six"] = four_in_six
        away_flags = (~group_sorted["is_home"]).astype(int)
        away_streak: List[float] = []
        running_away = 0
        for flag in away_flags:
            if flag:
                running_away += 1
            else:
                running_away = 0
            away_streak.append(float(running_away))
        team_df.loc[group_sorted.index, "away_game_streak"] = away_streak
        home_flags = group_sorted["is_home"].astype(int)
        home_streak: List[float] = []
        running_home = 0
        for flag in home_flags:
            if flag:
                running_home += 1
            else:
                running_home = 0
            home_streak.append(float(running_home))
        team_df.loc[group_sorted.index, "home_game_streak"] = home_streak
        location_ids = group_sorted["LocationTeamID"].apply(_to_str_id)
        travel_miles = np.full(len(group_sorted), np.nan, dtype=float)
        timezone_changes = np.full(len(group_sorted), np.nan, dtype=float)
        altitude_changes = np.full(len(group_sorted), np.nan, dtype=float)
        fatigue_scores = np.full(len(group_sorted), np.nan, dtype=float)
        prev_info: Optional[Dict[str, float]] = None
        for idx_pos, loc_id in enumerate(location_ids):
            info = TEAM_TRAVEL_INFO.get(loc_id) if loc_id is not None else None
            if info is None and loc_id and loc_id not in missing_travel_ids:
                logging.warning(
                    "Missing travel mapping for team/venue id %s; travel fatigue will be NaN",
                    loc_id,
                )
                missing_travel_ids.add(loc_id)
            if info is not None and prev_info is not None:
                distance = haversine_miles(
                    prev_info["lat"], prev_info["lon"], info["lat"], info["lon"]
                )
                tz_change = abs(info["tz"] - prev_info["tz"])
                alt_change = abs(info["alt"] - prev_info["alt"])
                fatigue = (
                    distance / TRAVEL_DISTANCE_NORMALIZER
                    + tz_change * TRAVEL_TIMEZONE_WEIGHT
                    + alt_change / TRAVEL_ALTITUDE_NORMALIZER
                )
                travel_miles[idx_pos] = distance
                timezone_changes[idx_pos] = tz_change
                altitude_changes[idx_pos] = alt_change
                fatigue_scores[idx_pos] = fatigue
            prev_info = info
        team_df.loc[group_sorted.index, "travel_miles_since_last_game"] = travel_miles
        team_df.loc[group_sorted.index, "travel_timezone_change"] = timezone_changes
        team_df.loc[group_sorted.index, "travel_altitude_change"] = altitude_changes
        team_df.loc[group_sorted.index, "travel_fatigue_score"] = fatigue_scores
    matchup_grouped = team_df.groupby(["TeamID", "OpponentID", "Season"], sort=False)
    team_df["head_to_head_games"] = matchup_grouped.cumcount().astype(float)
    wins_prior = matchup_grouped["TeamWin"].transform(
        lambda s: s.fillna(0).cumsum().shift(1)
    )
    wins_prior = wins_prior.fillna(0.0)
    team_df["head_to_head_wins"] = wins_prior
    score_margin = coerce_numeric(team_df["TeamScore"]) - coerce_numeric(
        team_df["OpponentScore"]
    )
    team_df["_score_margin"] = score_margin
    point_diff_prior = matchup_grouped["_score_margin"].transform(
        lambda s: s.fillna(0).cumsum().shift(1)
    )
    point_diff_prior = point_diff_prior.fillna(0.0)
    team_df["head_to_head_point_diff"] = point_diff_prior
    games_played = team_df["head_to_head_games"]
    denom = games_played.where(games_played > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_pct = wins_prior.divide(denom)
    team_df["head_to_head_win_pct"] = win_pct
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_point_diff = point_diff_prior.divide(denom)
    win_component = (win_pct.fillna(0.5) - 0.5) * 1.5
    point_component = avg_point_diff.fillna(0.0) / 15.0
    index_raw = win_component + point_component
    index_score = np.tanh(index_raw)
    index_score = index_score.where(games_played > 0)
    team_df["head_to_head_index"] = index_score
    team_df.drop(columns=["_score_margin"], inplace=True)
    drop_cols = [
        "head_to_head_games",
        "head_to_head_wins",
        "head_to_head_point_diff",
        "head_to_head_win_pct",
    ]
    existing = [col for col in drop_cols if col in team_df.columns]
    if existing:
        team_df.drop(columns=existing, inplace=True)
    travel_grouped = team_df.groupby(["TeamID", "Season"], sort=False)
    for feature in TRAVEL_AGG_FEATURES:
        if feature not in team_df.columns:
            continue
        team_df[f"{feature}_season"] = travel_grouped[feature].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        for window in windows:
            team_df[f"{feature}_r{window}"] = travel_grouped[feature].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
            )
    return team_df


def assign_features(
    base_df: pd.DataFrame, team_df: pd.DataFrame, windows: Sequence[int]
) -> None:
    home_features = team_df[team_df["is_home"]].set_index("OriginalIndex")
    away_features = team_df[~team_df["is_home"]].set_index("OriginalIndex")
    required_columns: List[str] = []
    for metric in SELF_METRICS:
        base = f"SELF_{metric}"
        required_columns.extend(
            [f"HOME_{base}_season", f"AWAY_{base}_season"]
        )
        for window in windows:
            required_columns.extend(
                [f"HOME_{base}_r{window}", f"AWAY_{base}_r{window}"]
            )
    for metric in OPP_METRICS:
        base = f"OPP_{metric}"
        required_columns.extend(
            [f"HOME_{base}_season", f"AWAY_{base}_season"]
        )
        for window in windows:
            required_columns.extend(
                [f"HOME_{base}_r{window}", f"AWAY_{base}_r{window}"]
            )
    for feature in REST_FEATURES:
        if feature == "home_game_streak":
            required_columns.append("HOME_home_game_streak")
        elif feature == "away_game_streak":
            required_columns.append("AWAY_away_game_streak")
        else:
            required_columns.extend([f"HOME_{feature}", f"AWAY_{feature}"])
    travel_suffixes = ["_season"] + [f"_r{window}" for window in windows]
    for feature in TRAVEL_AGG_FEATURES:
        for suffix in travel_suffixes:
            required_columns.extend(
                [f"HOME_{feature}{suffix}", f"AWAY_{feature}{suffix}"]
            )
    sos_suffixes = ["_season"] + [f"_r{window}" for window in windows]
    for suffix in sos_suffixes:
        required_columns.append(f"{SCHEDULE_DIFF_PREFIX}{suffix}")
    elo_suffixes = ["season"] + [f"r{window}" for window in windows]
    for suffix in elo_suffixes:
        required_columns.append(f"Elo_diff_{suffix}")
    for suffix in WIN_RATE_SUFFIXES:
        required_columns.extend(
            [f"HOME_team_win_rate_{suffix}", f"AWAY_team_win_rate_{suffix}"]
        )
    required_columns.extend(WIN_RATE_DIFF_COLS)
    required_columns.append(HEAD_TO_HEAD_COLUMN)
    required_columns.append("REST_DAY_DIFFERENTIAL")
    ensure_columns(base_df, required_columns)

    for metric in SELF_METRICS:
        home_base = f"HOME_SELF_{metric}"
        away_base = f"AWAY_SELF_{metric}"
        season_key = f"SELF_{metric}_season"
        if season_key in home_features.columns:
            base_df.loc[home_features.index, f"{home_base}_season"] = home_features[
                season_key
            ]
        if season_key in away_features.columns:
            base_df.loc[away_features.index, f"{away_base}_season"] = away_features[
                season_key
            ]
        for window in windows:
            key = f"SELF_{metric}_r{window}"
            if key in home_features.columns:
                base_df.loc[home_features.index, f"{home_base}_r{window}"] = home_features[
                    key
                ]
            if key in away_features.columns:
                base_df.loc[away_features.index, f"{away_base}_r{window}"] = away_features[
                    key
                ]
    for metric in OPP_METRICS:
        home_base = f"HOME_OPP_{metric}"
        away_base = f"AWAY_OPP_{metric}"
        season_key = f"OPP_{metric}_season"
        if season_key in home_features.columns:
            base_df.loc[home_features.index, f"{home_base}_season"] = home_features[
                season_key
            ]
        if season_key in away_features.columns:
            base_df.loc[away_features.index, f"{away_base}_season"] = away_features[
                season_key
            ]
        for window in windows:
            key = f"OPP_{metric}_r{window}"
            if key in home_features.columns:
                base_df.loc[home_features.index, f"{home_base}_r{window}"] = home_features[
                    key
                ]
            if key in away_features.columns:
                base_df.loc[away_features.index, f"{away_base}_r{window}"] = away_features[
                    key
                ]
    for feature in REST_FEATURES:
        if feature == "home_game_streak":
            if feature in home_features.columns:
                base_df.loc[
                    home_features.index, "HOME_home_game_streak"
                ] = home_features[feature]
            continue
        if feature == "away_game_streak":
            if feature in away_features.columns:
                base_df.loc[
                    away_features.index, "AWAY_away_game_streak"
                ] = away_features[feature]
            continue
        if feature in home_features.columns:
            base_df.loc[home_features.index, f"HOME_{feature}"] = home_features[
                feature
            ]
        if feature in away_features.columns:
            base_df.loc[away_features.index, f"AWAY_{feature}"] = away_features[
                feature
            ]
    for feature in TRAVEL_AGG_FEATURES:
        for suffix in travel_suffixes:
            source_col = f"{feature}{suffix}"
            if source_col in home_features.columns:
                base_df.loc[
                    home_features.index, f"HOME_{feature}{suffix}"
                ] = home_features[source_col]
            if source_col in away_features.columns:
                base_df.loc[
                    away_features.index, f"AWAY_{feature}{suffix}"
                ] = away_features[source_col]
    for suffix in sos_suffixes:
        source_col = f"strength_of_schedule{suffix}"
        if (
            source_col in home_features.columns
            and source_col in away_features.columns
        ):
            diff_values = home_features[source_col] - away_features[source_col]
            base_df.loc[home_features.index, f"{SCHEDULE_DIFF_PREFIX}{suffix}"] = (
                diff_values
            )
    if HEAD_TO_HEAD_COLUMN in home_features.columns and HEAD_TO_HEAD_COLUMN in away_features.columns:
        home_head = home_features[HEAD_TO_HEAD_COLUMN].fillna(0.0)
        away_head = away_features[HEAD_TO_HEAD_COLUMN].fillna(0.0)
        base_df.loc[home_features.index, HEAD_TO_HEAD_COLUMN] = home_head - away_head
    if (
        "HOME_days_since_last_game" in base_df.columns
        and "AWAY_days_since_last_game" in base_df.columns
    ):
        base_df["REST_DAY_DIFFERENTIAL"] = (
            base_df["HOME_days_since_last_game"]
            - base_df["AWAY_days_since_last_game"]
        )
    for suffix in elo_suffixes:
        key = f"SELF_Elo_{suffix}"
        if key in home_features.columns and key in away_features.columns:
            diff_values = home_features[key] - away_features[key]
            base_df.loc[home_features.index, f"Elo_diff_{suffix}"] = diff_values
    for suffix in WIN_RATE_SUFFIXES:
        source_col = f"TeamWin_{suffix}"
        if source_col in home_features.columns:
            base_df.loc[
                home_features.index, f"HOME_team_win_rate_{suffix}"
            ] = home_features[source_col]
        if source_col in away_features.columns:
            base_df.loc[
                away_features.index, f"AWAY_team_win_rate_{suffix}"
            ] = away_features[source_col]
    for suffix, col_name in WIN_RATE_DIFF_MAP.items():
        source_col = f"TeamWin_{suffix}"
        if source_col in home_features.columns and source_col in away_features.columns:
            diff = home_features[source_col] - away_features[source_col]
            base_df.loc[home_features.index, col_name] = diff
    # Mirror the differences to original DataFrame rows (home index only already aligns).


def drop_non_aggregated_metric_columns(df: pd.DataFrame) -> None:
    for prefix in ("HOME_SELF", "HOME_OPP", "AWAY_SELF", "AWAY_OPP"):
        metrics = SELF_METRICS if "SELF" in prefix else OPP_METRICS
        for metric in metrics:
            col = f"{prefix}_{metric}"
            if col in df.columns:
                df.drop(columns=col, inplace=True)


def compute_win_rate_features(
    team_df: pd.DataFrame, windows: Sequence[int]
) -> None:
    grouped = team_df.groupby(["TeamID", "Season"], sort=False)
    team_df["TeamWin_season"] = grouped["TeamWin"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    for window in windows:
        team_df[f"TeamWin_r{window}"] = grouped["TeamWin"].transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
        )


def compute_strength_of_schedule(
    team_df: pd.DataFrame, windows: Sequence[int]
) -> None:
    if team_df.empty:
        return
    suffixes = ["season"] + [f"r{window}" for window in windows]
    lookup = team_df.set_index(["GameID", "Season", "TeamID"])
    opp_index = pd.MultiIndex.from_arrays(
        [team_df["GameID"], team_df["Season"], team_df["OpponentID"]],
        names=["GameID", "Season", "TeamID"],
    )
    for suffix in suffixes:
        source_col = f"TeamWin_{suffix}"
        if source_col not in lookup.columns:
            continue
        sos_values = lookup[source_col].reindex(opp_index)
        team_df[f"strength_of_schedule_{suffix}"] = sos_values.values


RECENCY_SEASON_HALF_LIFE = 6.0
RECENCY_DAY_HALF_LIFE = 730.0
RECENCY_WEIGHT_MIN = 0.1
RECENCY_SEASON_COMPONENT = 0.7


def compute_recent_weightings(df: pd.DataFrame) -> pd.DataFrame:
    """Return a defragmented copy with recency weighting applied."""

    if df.empty:
        return df.copy()

    max_season_start = df["SeasonStartYear"].max()
    seasons_ago = (max_season_start - df["SeasonStartYear"]).clip(lower=0)

    max_date = df["GameDate_dt"].max()
    days_since = (max_date - df["GameDate_dt"]).dt.days.clip(lower=0)

    season_decay = 2.0 ** (-seasons_ago / RECENCY_SEASON_HALF_LIFE)
    day_decay = 2.0 ** (-days_since / RECENCY_DAY_HALF_LIFE)
    combined = (
        RECENCY_SEASON_COMPONENT * season_decay
        + (1.0 - RECENCY_SEASON_COMPONENT) * day_decay
    )
    weights = RECENCY_WEIGHT_MIN + (1.0 - RECENCY_WEIGHT_MIN) * combined
    clipped = weights.clip(lower=RECENCY_WEIGHT_MIN, upper=1.0)
    recency_df = clipped.to_frame(name="RecencyWeight")

    without_recency = df.drop(columns=["RecencyWeight"], errors="ignore").copy()
    return pd.concat([without_recency, recency_df], axis=1)


def validate_output(df: pd.DataFrame, windows: Sequence[int], min_start_year: int) -> None:
    if (df["SeasonStartYear"] < min_start_year).any():
        raise ValueError("Rows prior to min_season detected after filtering")
    for metric in SELF_METRICS:
        season_cols = [f"HOME_SELF_{metric}_season", f"AWAY_SELF_{metric}_season"]
        for col in season_cols:
            if col in df.columns and df[col].notna().sum() == 0:
                logging.warning("Season-to-date column %s contains only NaNs", col)
    for metric in OPP_METRICS:
        season_cols = [f"HOME_OPP_{metric}_season", f"AWAY_OPP_{metric}_season"]
        for col in season_cols:
            if col in df.columns and df[col].notna().sum() == 0:
                logging.warning("Season-to-date column %s contains only NaNs", col)
    percent_cols = [col for col in df.columns if "%" in col]
    for col in percent_cols:
        if ((df[col] < -1e-6) | (df[col] > 1 + 1e-6)).any():
            raise ValueError(f"Column {col} contains values outside [0, 1]")
    for window in windows:
        rolling_cols = [col for col in df.columns if col.endswith(f"_r{window}")]
        if rolling_cols:
            non_nan_counts = df[rolling_cols].notna().sum()
            if (non_nan_counts == 0).any():
                logging.warning(
                    "No non-NaN values detected for some rolling window features (r%s)",
                    window,
                )
    if "RecencyWeight" in df.columns:
        recency = df["RecencyWeight"]
        if recency.isna().any():
            raise ValueError("RecencyWeight column contains NaN values")
        if ((recency < -1e-6) | (recency > 1 + 1e-6)).any():
            raise ValueError("RecencyWeight column must be scaled to [0, 1]")
        most_recent_idx = df["GameDate_dt"].idxmax()
        if not np.isclose(recency.loc[most_recent_idx], recency.max()):
            raise ValueError(
                "RecencyWeight should be highest for the most recent game"
            )


def build_output_columns(windows: Sequence[int]) -> List[str]:
    windows = list(dict.fromkeys(windows))
    windows.sort()
    columns: List[str] = [
        "GameID",
        "HomeID",
        "AwayID",
        "GameDate",
        "Location",
    ]
    for prefix, metrics in (
        ("HOME_SELF", SELF_METRICS),
        ("HOME_OPP", OPP_METRICS),
        ("AWAY_SELF", SELF_METRICS),
        ("AWAY_OPP", OPP_METRICS),
    ):
        for metric in metrics:
            base = f"{prefix}_{metric}"
            columns.append(f"{base}_season")
            for window in windows:
                columns.append(f"{base}_r{window}")
    for feature in REST_FEATURES:
        if feature == "home_game_streak":
            columns.append("HOME_home_game_streak")
            continue
        if feature == "away_game_streak":
            columns.append("AWAY_away_game_streak")
            continue
        for prefix in ("HOME", "AWAY"):
            columns.append(f"{prefix}_{feature}")
    for prefix in ("HOME", "AWAY"):
        for feature in TRAVEL_AGG_FEATURES:
            columns.append(f"{prefix}_{feature}_season")
            for window in windows:
                columns.append(f"{prefix}_{feature}_r{window}")
    for suffix in ["_season"] + [f"_r{window}" for window in windows]:
        columns.append(f"{SCHEDULE_DIFF_PREFIX}{suffix}")
    for suffix in ["season"] + [f"r{window}" for window in windows]:
        columns.append(f"Elo_diff_{suffix}")
    for suffix in WIN_RATE_SUFFIXES:
        columns.append(f"HOME_team_win_rate_{suffix}")
        columns.append(f"AWAY_team_win_rate_{suffix}")
    columns.extend(WIN_RATE_DIFF_COLS)
    columns.append(HEAD_TO_HEAD_COLUMN)
    columns.append("REST_DAY_DIFFERENTIAL")
    columns.extend(["RecencyWeight", "HomeTeamWin"])
    return columns


def impute_numeric_columns(df: pd.DataFrame) -> None:
    protected_columns = set(WIN_RATE_DIFF_COLS)
    for suffix in WIN_RATE_SUFFIXES:
        protected_columns.add(f"HOME_team_win_rate_{suffix}")
        protected_columns.add(f"AWAY_team_win_rate_{suffix}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in protected_columns or col == "HomeTeamWin":
            continue
        median = df[col].median()
        if pd.notna(median):
            df[col] = df[col].fillna(median)


def main() -> None:
    args = parse_args()
    configure_logging()
    windows = sorted(set(args.windows))
    logging.info("==== Starting dataset build ====")
    logging.info("Reading input CSV from %s", args.in_csv)
    df = pd.read_csv(args.in_csv)
    logging.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    df = apply_aliases(df)
    df = maybe_convert_team_view_to_game_level(df)
    ensure_required_columns(df)
    try_derive_rate_columns(df)
    normalize_percent_columns(df)
    df["GameDate_dt"] = df["GameDate"].apply(parse_game_date)
    before_drop = len(df)
    df = df[df["GameDate_dt"].notna()].copy()
    dropped = before_drop - len(df)
    if dropped:
        logging.warning("Dropped %d rows due to unparseable GameDate", dropped)
    df["GameDate"] = df["GameDate_dt"].dt.strftime("%Y-%m-%d")
    df["Season"] = df["GameDate_dt"].apply(infer_season)
    df["SeasonStartYear"] = df["Season"].apply(season_start_year)
    min_start_year = season_start_year(args.min_season)
    before_filter = len(df)
    df = df[df["SeasonStartYear"] >= min_start_year].copy()
    logging.info("Filtered seasons before %s: kept %d of %d rows", args.min_season, len(df), before_filter)
    if args.smoke:
        smoke_limit = min(len(df), 1000)
        df = df.sort_values("GameDate_dt").tail(smoke_limit).copy()
        logging.info(
            "Smoke mode enabled: limiting dataset to %d most recent games after filtering",
            smoke_limit,
        )
    df["Location"] = df["Location"].astype(str).str.strip()
    df.loc[df["Location"].str.lower().isin({"", "nan", "none"}), "Location"] = np.nan
    df["Location"] = df["Location"].fillna("Unknown")
    df["HomeTeamWin"] = df.apply(determine_home_win, axis=1)
    missing_target = df["HomeTeamWin"].isna().sum()
    if missing_target:
        logging.warning("Dropping %d rows with indeterminable HomeTeamWin", missing_target)
    df = df[df["HomeTeamWin"].notna()].copy()
    essential_null = df[["GameID", "HomeID", "AwayID", "GameDate"]].isna().any(axis=1)
    if essential_null.any():
        logging.warning("Dropping %d rows with missing essential identifiers", essential_null.sum())
        df = df[~essential_null].copy()
    df.drop_duplicates(subset=["GameID", "HomeID", "AwayID"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["OriginalIndex"] = df.index
    apply_pre_game_elo(df)
    ensure_opponent_metric_parity(df)
    team_df = compute_team_long_frame(df, windows)
    compute_win_rate_features(team_df, windows)
    compute_strength_of_schedule(team_df, windows)
    assign_features(df, team_df, windows)
    drop_non_aggregated_metric_columns(df)
    df = compute_recent_weightings(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    output_columns = build_output_columns(windows)
    ensure_columns(df, output_columns)
    df["GameID"] = df["GameID"].astype(str)
    df["HomeID"] = df["HomeID"].astype(str)
    df["AwayID"] = df["AwayID"].astype(str)
    df["Location"] = df["Location"].astype(str)
    df = df[output_columns + ["SeasonStartYear", "GameDate_dt"]]
    df = df.sort_values("GameDate_dt").copy()
    impute_numeric_columns(df)
    df["HomeTeamWin"] = df["HomeTeamWin"].astype(int)
    validate_output(df, windows, min_start_year)
    df = df.drop(columns=["SeasonStartYear", "GameDate_dt"])
    rename_map = {col: to_snake_case(col) for col in df.columns}
    new_names = list(rename_map.values())
    if len(new_names) != len(set(new_names)):
        duplicates = sorted({name for name in new_names if new_names.count(name) > 1})
        raise ValueError(
            "Column rename collision detected for: " + ", ".join(duplicates)
        )
    df.rename(columns=rename_map, inplace=True)
    logging.info("Final dataset shape: %d rows x %d columns", df.shape[0], df.shape[1])
    df.to_csv(args.out_csv, index=False)
    logging.info(
        "Wrote %d rows x %d columns to %s", df.shape[0], df.shape[1], args.out_csv
    )
    logging.info("==== Dataset build complete ====")


if __name__ == "__main__":
    main()
