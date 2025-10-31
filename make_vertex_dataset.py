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
    "HomeScore": ["HomeScore", "HOME_PTS", "home_pts", "HomePoints"],
    "AwayScore": ["AwayScore", "AWAY_PTS", "away_pts", "AwayPoints"],
    "WinnerStr": ["Winner", "WINNER", "Result", "result", "HomeWin"],
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
WIN_RATE_BASE = "Difference in the relative frequency of victories"
WIN_RATE_COLS: Sequence[str] = (
    WIN_RATE_BASE,
    "DiffRelFreqVictories_r4",
    "DiffRelFreqVictories_r10",
    "DiffRelFreqVictories_r20",
)

RAW_STAT_ALIASES: Dict[str, List[str]] = {
    "FGM": ["FGM", "FGM_MADE", "FGM_MADE_TOTAL", "FIELD_GOALS_MADE", "FGM_TOTAL"],
    "FG3M": ["FG3M", "FG3_MADE", "3PM", "FG3M_TOTAL", "THREE_PM"],
    "FGA": ["FGA", "FIELD_GOALS_ATTEMPTED", "FGA_TOTAL"],
    "FTA": ["FTA", "FREE_THROW_ATTEMPTS", "FTA_TOTAL"],
    "TOV": ["TOV", "TURNOVERS", "TO", "TOV_TOTAL"],
    "ORB": ["ORB", "OFFENSIVE_REBOUNDS", "OREB", "ORB_TOTAL"],
    "DRB": ["DRB", "DEFENSIVE_REBOUNDS", "DREB", "DRB_TOTAL"],
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


def parse_game_date(value: object) -> Optional[datetime]:
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Attempt to parse Excel serial dates or year-month-day numbers.
        try:
            if value > 10_000:
                return datetime.fromordinal(datetime(1899, 12, 30).toordinal() + int(value))
            return datetime.strptime(str(int(value)), "%Y%m%d")
        except Exception:  # noqa: BLE001
            return None
    value_str = str(value).strip()
    if not value_str:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(value_str, fmt)
        except ValueError:
            continue
    return None


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


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    if rename_map:
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
        base_df.loc[home_features.index, home_base] = home_features[f"SELF_{metric}_season"]
        base_df.loc[away_features.index, away_base] = away_features[f"SELF_{metric}_season"]
        base_df.loc[home_features.index, f"{home_base}_season"] = home_features[
            f"SELF_{metric}_season"
        ]
        base_df.loc[away_features.index, f"{away_base}_season"] = away_features[
            f"SELF_{metric}_season"
        ]
        for window in windows:
            base_df.loc[home_features.index, f"{home_base}_r{window}"] = home_features[
                f"SELF_{metric}_r{window}"
            ]
            base_df.loc[away_features.index, f"{away_base}_r{window}"] = away_features[
                f"SELF_{metric}_r{window}"
            ]
    for metric in OPP_METRICS:
        home_base = f"HOME_OPP_{metric}"
        away_base = f"AWAY_OPP_{metric}"
        base_df.loc[home_features.index, home_base] = home_features[f"OPP_{metric}_season"]
        base_df.loc[away_features.index, away_base] = away_features[f"OPP_{metric}_season"]
        base_df.loc[home_features.index, f"{home_base}_season"] = home_features[
            f"OPP_{metric}_season"
        ]
        base_df.loc[away_features.index, f"{away_base}_season"] = away_features[
            f"OPP_{metric}_season"
        ]
        for window in windows:
            base_df.loc[home_features.index, f"{home_base}_r{window}"] = home_features[
                f"OPP_{metric}_r{window}"
            ]
            base_df.loc[away_features.index, f"{away_base}_r{window}"] = away_features[
                f"OPP_{metric}_r{window}"
            ]
    for suffix, col_name in (
        ("season", WIN_RATE_BASE),
        ("r4", "DiffRelFreqVictories_r4"),
        ("r10", "DiffRelFreqVictories_r10"),
        ("r20", "DiffRelFreqVictories_r20"),
    ):
        home_col = f"TeamWin_{suffix}"
        if home_col not in home_features.columns or home_col not in away_features.columns:
            continue
        diff = home_features[home_col] - away_features[home_col]
        base_df.loc[home_features.index, col_name] = diff
    # Mirror the differences to original DataFrame rows (home index only already aligns).


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


def compute_recent_weightings(df: pd.DataFrame) -> None:
    max_season_start = df["SeasonStartYear"].max()
    df["RecentSeasonWeighting"] = df["SeasonStartYear"].apply(
        lambda start: 1.0 / (1 + max(0, max_season_start - start))
    )
    max_date = df["GameDate_dt"].max()
    days_since = (max_date - df["GameDate_dt"]).dt.days
    df["RecentGameWeighting"] = np.exp(-days_since / 365.0)


def validate_output(df: pd.DataFrame, windows: Sequence[int], min_start_year: int) -> None:
    if (df["SeasonStartYear"] < min_start_year).any():
        raise ValueError("Rows prior to min_season detected after filtering")
    if not (df["HomeMarker"] == 1).all():
        raise ValueError("HomeMarker column must be constant 1")
    for metric in SELF_METRICS:
        base = f"HOME_SELF_{metric}"
        season = f"{base}_season"
        if not np.allclose(df[base], df[season], equal_nan=True):
            raise ValueError(f"Base column {base} must equal {season}")
        base = f"AWAY_SELF_{metric}"
        season = f"{base}_season"
        if not np.allclose(df[base], df[season], equal_nan=True):
            raise ValueError(f"Base column {base} must equal {season}")
    for metric in OPP_METRICS:
        base = f"HOME_OPP_{metric}"
        season = f"{base}_season"
        if not np.allclose(df[base], df[season], equal_nan=True):
            raise ValueError(f"Base column {base} must equal {season}")
        base = f"AWAY_OPP_{metric}"
        season = f"{base}_season"
        if not np.allclose(df[base], df[season], equal_nan=True):
            raise ValueError(f"Base column {base} must equal {season}")
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


def build_output_columns(windows: Sequence[int]) -> List[str]:
    windows = list(dict.fromkeys(windows))
    windows.sort()
    columns: List[str] = [
        "GameID",
        "HomeID",
        "AwayID",
        "GameDate",
        "Location",
        "HomeMarker",
    ]
    for prefix in ("HOME_SELF",):
        for metric in SELF_METRICS:
            base = f"{prefix}_{metric}"
            columns.append(base)
            columns.append(f"{base}_season")
            for window in windows:
                columns.append(f"{base}_r{window}")
    for prefix in ("HOME_OPP",):
        for metric in OPP_METRICS:
            base = f"{prefix}_{metric}"
            columns.append(base)
            columns.append(f"{base}_season")
            for window in windows:
                columns.append(f"{base}_r{window}")
    for prefix in ("AWAY_SELF",):
        for metric in SELF_METRICS:
            base = f"{prefix}_{metric}"
            columns.append(base)
            columns.append(f"{base}_season")
            for window in windows:
                columns.append(f"{base}_r{window}")
    for prefix in ("AWAY_OPP",):
        for metric in OPP_METRICS:
            base = f"{prefix}_{metric}"
            columns.append(base)
            columns.append(f"{base}_season")
            for window in windows:
                columns.append(f"{base}_r{window}")
    columns.extend(WIN_RATE_COLS)
    columns.extend(["RecentSeasonWeighting", "RecentGameWeighting", "HomeTeamWin"])
    return columns


def impute_numeric_columns(df: pd.DataFrame) -> None:
    diff_columns = set(WIN_RATE_COLS)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in diff_columns or col == "HomeTeamWin":
            continue
        median = df[col].median()
        if pd.notna(median):
            df[col].fillna(median, inplace=True)


def main() -> None:
    args = parse_args()
    configure_logging()
    windows = sorted(set(args.windows))
    logging.info("==== Starting dataset build ====")
    logging.info("Reading input CSV from %s", args.in_csv)
    df = pd.read_csv(args.in_csv)
    logging.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    df = apply_aliases(df)
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
    df["Location"] = df["Location"].fillna("Home")
    df["HomeMarker"] = 1
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
    team_df = compute_team_long_frame(df, windows)
    compute_win_rate_features(team_df, windows)
    assign_features(df, team_df, windows)
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
    df["HomeMarker"] = df["HomeMarker"].astype(int)
    df["HomeTeamWin"] = df["HomeTeamWin"].astype(int)
    df = df[output_columns + ["SeasonStartYear", "GameDate_dt"]]
    df = df.sort_values("GameDate_dt").copy()
    impute_numeric_columns(df)
    df["HomeMarker"] = df["HomeMarker"].astype(int)
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
