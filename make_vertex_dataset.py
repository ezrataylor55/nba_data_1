#!/usr/bin/env python3
"""Build a Vertex AI AutoML ready dataset with season-to-date and rolling features."""

import argparse
import logging
from datetime import datetime
import math
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

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
    "HOME_OPP_EFG%": ["HomeOpp_eFG%", "HOME_OPP_EFG_PCT", "home_opp_efg_pct"],
    "HOME_OPP_TO%": ["HomeOpp_TOV%", "HOME_OPP_TOV_PCT", "home_opp_to_pct"],
    "HOME_OPP_FTR": ["HomeOpp_FTR", "HOME_OPP_FTR", "home_opp_ftr"],
    "AWAY_SELF_Elo": ["AwayElo", "AWAY_ELO", "away_elo"],
    "AWAY_SELF_EFG%": ["Away_eFG%", "AWAY_EFG_PCT", "away_efg_pct", "AwayEFG"],
    "AWAY_SELF_TO%": ["Away_TOV%", "AWAY_TOV_PCT", "away_to_pct", "AwayTO%"],
    "AWAY_SELF_OR%": ["Away_ORB%", "AWAY_OR_PCT", "away_or_pct", "AwayORB%"],
    "AWAY_SELF_FTR": ["Away_FTR", "AWAY_FTR", "away_ftr"],
    "AWAY_SELF_DR%": ["Away_DRB%", "AWAY_DR_PCT", "away_dr_pct", "AwayDRB%"],
    "AWAY_OPP_EFG%": ["AwayOpp_eFG%", "AWAY_OPP_EFG_PCT", "away_opp_efg_pct"],
    "AWAY_OPP_TO%": ["AwayOpp_TOV%", "AWAY_OPP_TOV_PCT", "away_opp_to_pct"],
    "AWAY_OPP_FTR": ["AwayOpp_FTR", "AWAY_OPP_FTR", "away_opp_ftr"],
}

SELF_METRICS: Sequence[str] = ("Elo", "EFG%", "TO%", "OR%", "FTR", "DR%")
OPP_METRICS: Sequence[str] = ("EFG%", "TO%", "FTR")
WIN_RATE_SUFFIX_MAP = {
    "season": "HomeAwayWinRateDelta_season",
    "r4": "HomeAwayWinRateDelta_r4",
    "r10": "HomeAwayWinRateDelta_r10",
    "r20": "HomeAwayWinRateDelta_r20",
}
WIN_RATE_COLS: Sequence[str] = tuple(WIN_RATE_SUFFIX_MAP.values())

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
        default="2013-2014",
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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.mask((denominator == 0) | denominator.isna())


def apply_pre_game_elo(df: pd.DataFrame, base_rating: float = 1500.0, k_factor: float = 20.0) -> None:
    """Assign pre-game Elo ratings that reset each season."""

    if df.empty:
        return

    order_cols = ["SeasonStartYear", "GameDate_dt", "GameID", "HomeID", "AwayID"]
    ordered_index = df.sort_values(order_cols).index
    home_pre = pd.Series(base_rating, index=df.index, dtype=float)
    away_pre = pd.Series(base_rating, index=df.index, dtype=float)
    ratings: Dict[tuple, float] = {}

    for idx in ordered_index:
        season = df.at[idx, "Season"]
        home_id = str(df.at[idx, "HomeID"])
        away_id = str(df.at[idx, "AwayID"])
        home_key = (season, home_id)
        away_key = (season, away_id)
        home_rating = ratings.get(home_key, base_rating)
        away_rating = ratings.get(away_key, base_rating)
        home_pre.at[idx] = home_rating
        away_pre.at[idx] = away_rating

        outcome = df.at[idx, "HomeTeamWin"]
        if pd.isna(outcome):
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
            if margin > 0:
                margin_multiplier = math.log(margin + 1.0) * (
                    2.2 / (abs(home_rating - away_rating) * 0.001 + 2.2)
                )
        adjustment = k_factor * margin_multiplier
        home_rating += adjustment * (outcome - expected_home)
        away_rating += adjustment * ((1.0 - outcome) - (1.0 - expected_home))
        ratings[home_key] = home_rating
        ratings[away_key] = away_rating

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

    home_fgm = _series_with_suffix(merged, "fieldGoalsMade", "home")
    home_fg3m = _series_with_suffix(merged, "threePointersMade", "home")
    home_fga = _series_with_suffix(merged, "fieldGoalsAttempted", "home")
    away_fgm = _series_with_suffix(merged, "fieldGoalsMade", "away")
    away_fg3m = _series_with_suffix(merged, "threePointersMade", "away")
    away_fga = _series_with_suffix(merged, "fieldGoalsAttempted", "away")

    home_turnovers = _series_with_suffix(merged, "turnovers", "home")
    away_turnovers = _series_with_suffix(merged, "turnovers", "away")
    home_fta = _series_with_suffix(merged, "freeThrowsAttempted", "home")
    away_fta = _series_with_suffix(merged, "freeThrowsAttempted", "away")

    home_orb = _series_with_suffix(merged, "reboundsOffensive", "home")
    away_orb = _series_with_suffix(merged, "reboundsOffensive", "away")
    home_drb = _series_with_suffix(merged, "reboundsDefensive", "home")
    away_drb = _series_with_suffix(merged, "reboundsDefensive", "away")

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
            "HOME_OPP_EFG%": away_efg,
            "AWAY_OPP_EFG%": home_efg,
            "HOME_OPP_TO%": away_to,
            "AWAY_OPP_TO%": home_to,
            "HOME_OPP_FTR": away_ftr,
            "AWAY_OPP_FTR": home_ftr,
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
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
            )
    return team_df


def assign_features(
    base_df: pd.DataFrame, team_df: pd.DataFrame, windows: Sequence[int]
) -> None:
    home_features = team_df[team_df["is_home"]].set_index("OriginalIndex")
    away_features = team_df[~team_df["is_home"]].set_index("OriginalIndex")
    for metric in SELF_METRICS:
        home_base = f"HOME_SELF_{metric}"
        away_base = f"AWAY_SELF_{metric}"
        home_season_col = f"{home_base}_season"
        away_season_col = f"{away_base}_season"
        if home_season_col not in base_df.columns:
            base_df[home_season_col] = np.nan
        if away_season_col not in base_df.columns:
            base_df[away_season_col] = np.nan
        home_season_values = home_features[f"SELF_{metric}_season"]
        away_season_values = away_features[f"SELF_{metric}_season"]
        base_df.loc[home_features.index, home_season_col] = home_season_values
        base_df.loc[away_features.index, away_season_col] = away_season_values
        for window in windows:
            home_window_col = f"{home_base}_r{window}"
            away_window_col = f"{away_base}_r{window}"
            if home_window_col not in base_df.columns:
                base_df[home_window_col] = np.nan
            if away_window_col not in base_df.columns:
                base_df[away_window_col] = np.nan
            base_df.loc[home_features.index, home_window_col] = home_features[
                f"SELF_{metric}_r{window}"
            ]
            base_df.loc[away_features.index, away_window_col] = away_features[
                f"SELF_{metric}_r{window}"
            ]
    for metric in OPP_METRICS:
        home_base = f"HOME_OPP_{metric}"
        away_base = f"AWAY_OPP_{metric}"
        home_season_col = f"{home_base}_season"
        away_season_col = f"{away_base}_season"
        if home_season_col not in base_df.columns:
            base_df[home_season_col] = np.nan
        if away_season_col not in base_df.columns:
            base_df[away_season_col] = np.nan
        home_season_values = home_features[f"OPP_{metric}_season"]
        away_season_values = away_features[f"OPP_{metric}_season"]
        base_df.loc[home_features.index, home_season_col] = home_season_values
        base_df.loc[away_features.index, away_season_col] = away_season_values
        for window in windows:
            home_window_col = f"{home_base}_r{window}"
            away_window_col = f"{away_base}_r{window}"
            if home_window_col not in base_df.columns:
                base_df[home_window_col] = np.nan
            if away_window_col not in base_df.columns:
                base_df[away_window_col] = np.nan
            base_df.loc[home_features.index, home_window_col] = home_features[
                f"OPP_{metric}_r{window}"
            ]
            base_df.loc[away_features.index, away_window_col] = away_features[
                f"OPP_{metric}_r{window}"
            ]
    for suffix, col_name in WIN_RATE_SUFFIX_MAP.items():
        home_col = f"TeamWin_{suffix}"
        if home_col not in home_features.columns or home_col not in away_features.columns:
            continue
        diff = home_features[home_col] - away_features[home_col]
        if col_name not in base_df.columns:
            base_df[col_name] = np.nan
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
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
        )


RECENCY_SEASON_HALF_LIFE = 3.0
RECENCY_DAY_HALF_LIFE = 365.0
RECENCY_WEIGHT_MIN = 0.1


def compute_recent_weightings(df: pd.DataFrame) -> None:
    """Compute a gentle recency prior for Vertex AI sample weighting."""

    max_season_start = df["SeasonStartYear"].max()
    seasons_ago = (max_season_start - df["SeasonStartYear"]).clip(lower=0)

    max_date = df["GameDate_dt"].max()
    days_since = (max_date - df["GameDate_dt"]).dt.days.clip(lower=0)

    decay = np.exp(
        -(seasons_ago / RECENCY_SEASON_HALF_LIFE)
        - (days_since / RECENCY_DAY_HALF_LIFE)
    )
    weights = RECENCY_WEIGHT_MIN + (1.0 - RECENCY_WEIGHT_MIN) * decay
    df["RecencyWeight"] = weights.clip(lower=RECENCY_WEIGHT_MIN, upper=1.0)


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
    columns.extend(WIN_RATE_COLS)
    columns.extend(["RecencyWeight", "HomeTeamWin"])
    return columns


def impute_numeric_columns(df: pd.DataFrame) -> None:
    diff_columns = set(WIN_RATE_COLS)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in diff_columns or col == "HomeTeamWin":
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
        smoke_limit = min(len(df), 500)
        df = df.sort_values("GameDate_dt").head(smoke_limit).copy()
        logging.info(
            "Smoke mode enabled: limiting dataset to %d earliest games after filtering", smoke_limit
        )
    df["Location"] = df["Location"].astype(str).str.strip()
    df.loc[df["Location"].str.lower().isin({"", "nan", "none"}), "Location"] = np.nan
    df["Location"] = df["Location"].fillna("Unknown")
    df["HomeTeamWin"] = df.apply(determine_home_win, axis=1)
    missing_target = df["HomeTeamWin"].isna().sum()
    if missing_target:
        logging.warning("Dropping %d rows with indeterminable HomeTeamWin", missing_target)
    df = df[df["HomeTeamWin"].notna()].copy()
    df["HomeTeamWin"] = df["HomeTeamWin"].astype(int)
    essential_null = df[["GameID", "HomeID", "AwayID", "GameDate"]].isna().any(axis=1)
    if essential_null.any():
        logging.warning("Dropping %d rows with missing essential identifiers", essential_null.sum())
        df = df[~essential_null].copy()
    df.drop_duplicates(subset=["GameID", "HomeID", "AwayID"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["OriginalIndex"] = df.index
    apply_pre_game_elo(df)
    team_df = compute_team_long_frame(df, windows)
    compute_win_rate_features(team_df, windows)
    assign_features(df, team_df, windows)
    drop_non_aggregated_metric_columns(df)
    compute_recent_weightings(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    output_columns = build_output_columns(windows)
    for col in output_columns:
        if col not in df.columns:
            df[col] = np.nan
    df["GameID"] = df["GameID"].astype(str)
    df["HomeID"] = df["HomeID"].astype(str)
    df["AwayID"] = df["AwayID"].astype(str)
    df["Location"] = df["Location"].astype(str)
    df["HomeTeamWin"] = df["HomeTeamWin"].astype(int)
    df = df[output_columns + ["SeasonStartYear", "GameDate_dt"]]
    df = df.sort_values("GameDate_dt").copy()
    impute_numeric_columns(df)
    df["HomeTeamWin"] = df["HomeTeamWin"].astype(int)
    validate_output(df, windows, min_start_year)
    df = df.drop(columns=["SeasonStartYear", "GameDate_dt"])
    logging.info("Final dataset shape: %d rows x %d columns", df.shape[0], df.shape[1])
    df.to_csv(args.out_csv, index=False)
    logging.info(
        "Wrote %d rows x %d columns to %s", df.shape[0], df.shape[1], args.out_csv
    )
    logging.info("==== Dataset build complete ====")


if __name__ == "__main__":
    main()
