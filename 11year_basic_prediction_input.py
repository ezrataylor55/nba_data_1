#!/usr/bin/env python3
"""Generate current-season feature rows for today's NBA matchups."""

import argparse
import importlib.util
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _load_training_module():
    """Load the shared feature engineering helpers from the training script."""

    script_path = Path(__file__).with_name("11year_basic.py")
    if not script_path.exists():
        raise FileNotFoundError(
            "Expected training script '11year_basic.py' alongside "
            "11year_basic_prediction_input.py, but it was not found."
        )

    spec = importlib.util.spec_from_file_location("vertex_training", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to load feature helpers from 11year_basic.py"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mvd = _load_training_module()

DEFAULT_WINDOWS: Sequence[int] = (4, 10, 20)

SCHEDULE_ALIASES: Dict[str, Sequence[str]] = {
    "GameID": ("gameId", "GAME_ID", "Game Id", "game_id"),
    "GameDate": ("gameDate", "GAME_DATE", "Date", "date"),
    "HomeID": (
        "HomeTeamID",
        "homeTeamId",
        "HOME_TEAM_ID",
        "HomeID",
        "home_id",
    ),
    "AwayID": (
        "AwayTeamID",
        "awayTeamId",
        "AWAY_TEAM_ID",
        "AwayID",
        "away_id",
    ),
    "HomeCity": (
        "HomeTeamCity",
        "homeTeamCity",
        "HOME_TEAM_CITY",
        "HomeCity",
    ),
    "AwayCity": (
        "AwayTeamCity",
        "awayTeamCity",
        "AWAY_TEAM_CITY",
        "AwayCity",
    ),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate season-to-date and rolling features for today's NBA games "
            "by combining the team statistics CSV with the league schedule."
        )
    )
    parser.add_argument(
        "--in_csv",
        required=True,
        help="Path to the input CSV containing team statistics.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Path to write the snapshot CSV.",
    )
    parser.add_argument(
        "--schedule_csv",
        required=True,
        help="Path to the league schedule CSV (e.g., LeagueSchedule25_26).",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        type=int,
        default=list(DEFAULT_WINDOWS),
        help=(
            "Rolling window sizes to compute alongside season-to-date features. "
            "Defaults to 4, 10, and 20 games."
        ),
    )
    parser.add_argument(
        "--snapshot_date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help=(
            "ISO date string (YYYY-MM-DD) representing the games to predict. "
            "Defaults to today's UTC date."
        ),
    )
    return parser.parse_args()


def build_team_city_map(df: pd.DataFrame) -> Dict[str, str]:
    if "TeamID" not in df.columns or "TeamCity" not in df.columns:
        return {}
    subset = df[["TeamID", "TeamCity"]].dropna()
    if subset.empty:
        return {}
    subset["TeamID"] = subset["TeamID"].astype(str)
    subset["TeamCity"] = subset["TeamCity"].astype(str).str.strip()
    subset = subset[subset["TeamCity"] != ""]
    return dict(zip(subset["TeamID"], subset["TeamCity"]))


def resolve_schedule_aliases(df: pd.DataFrame) -> pd.DataFrame:
    resolved: Dict[str, str] = {}
    for canonical, aliases in SCHEDULE_ALIASES.items():
        candidates = (canonical,) + tuple(aliases)
        for candidate in candidates:
            if candidate in df.columns:
                resolved[canonical] = candidate
                break
    required = {"GameDate", "HomeID", "AwayID"}
    missing = [field for field in required if field not in resolved]
    if missing:
        raise ValueError(
            "Missing required schedule columns: " + ", ".join(sorted(missing))
        )
    renamed = df.rename(columns={orig: canon for canon, orig in resolved.items()})
    optional = set(SCHEDULE_ALIASES.keys()) - required
    for field in optional:
        if field in resolved:
            renamed[field] = renamed[field]
        elif field in renamed.columns:
            continue
    return renamed


def load_schedule_for_date(path: str, snapshot_ts: pd.Timestamp) -> pd.DataFrame:
    schedule = pd.read_csv(path)
    if schedule.empty:
        raise ValueError("Schedule CSV is empty; cannot determine today's games.")
    schedule = resolve_schedule_aliases(schedule)
    schedule["GameDate_dt"] = schedule["GameDate"].apply(mvd.parse_game_date)
    schedule = schedule[schedule["GameDate_dt"].notna()].copy()
    if schedule.empty:
        raise ValueError("Schedule CSV has no parsable game dates.")
    normalized_date = snapshot_ts.normalize()
    schedule = schedule[schedule["GameDate_dt"].dt.normalize() == normalized_date].copy()
    if schedule.empty:
        raise ValueError(
            f"No games found in schedule for snapshot date {normalized_date.date()}"
        )
    schedule["HomeID"] = schedule["HomeID"].astype(str)
    schedule["AwayID"] = schedule["AwayID"].astype(str)
    if "GameID" in schedule.columns:
        schedule["GameID"] = schedule["GameID"].astype(str)
    return schedule


def compute_current_team_features(
    df: pd.DataFrame,
    windows: Sequence[int],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        home_win = row.get("HomeTeamWin")
        home_id = row.get("HomeID")
        away_id = row.get("AwayID")
        for is_home, prefix in ((True, "HOME"), (False, "AWAY")):
            team_id = home_id if is_home else away_id
            opp_id = away_id if is_home else home_id
            record: Dict[str, object] = {
                "OriginalIndex": idx,
                "TeamID": team_id,
                "OpponentID": opp_id,
                "Season": row["Season"],
                "GameDate": row["GameDate_dt"],
                "TeamWin": home_win if is_home else (1.0 - home_win if pd.notna(home_win) else np.nan),
            }
            for metric in mvd.SELF_METRICS:
                column = f"{prefix}_SELF_{metric}"
                if column in row and pd.notna(row[column]):
                    record[f"SELF_{metric}"] = row[column]
                else:
                    record[f"SELF_{metric}"] = np.nan
            for metric in mvd.OPP_METRICS:
                column = f"{prefix}_OPP_{metric}"
                if column in row and pd.notna(row[column]):
                    record[f"OPP_{metric}"] = row[column]
                else:
                    record[f"OPP_{metric}"] = np.nan
            records.append(record)
    team_df = pd.DataFrame(records)
    if team_df.empty:
        return team_df
    team_df.sort_values(
        by=["TeamID", "Season", "GameDate", "OriginalIndex"], inplace=True
    )
    grouped = team_df.groupby(["TeamID", "Season"], sort=False)
    for metric in [f"SELF_{m}" for m in mvd.SELF_METRICS] + [f"OPP_{m}" for m in mvd.OPP_METRICS]:
        if metric not in team_df.columns:
            continue
        team_df[f"{metric}_season"] = grouped[metric].transform(
            lambda s: s.expanding().mean()
        )
        for window in windows:
            team_df[f"{metric}_r{window}"] = grouped[metric].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )
    team_df["TeamWin_season"] = grouped["TeamWin"].transform(
        lambda s: s.expanding().mean()
    )
    for window in windows:
        team_df[f"TeamWin_r{window}"] = grouped["TeamWin"].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
        )
    return team_df


def latest_team_features(
    team_df: pd.DataFrame, snapshot_ts: pd.Timestamp
) -> pd.DataFrame:
    if team_df.empty:
        return team_df
    filtered = team_df[team_df["GameDate"].notna()].copy()
    filtered = filtered[filtered["GameDate"] <= snapshot_ts]
    if filtered.empty:
        return filtered
    filtered["TeamID"] = filtered["TeamID"].astype(str)
    filtered.sort_values(
        by=["TeamID", "Season", "GameDate", "OriginalIndex"], inplace=True
    )
    latest = (
        filtered.groupby(["TeamID", "Season"], sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def _extract_location(
    schedule_row: pd.Series, team_city_map: Dict[str, str]
) -> str:
    location_fields = [
        schedule_row.get("HomeCity"),
        schedule_row.get("Location"),
        team_city_map.get(schedule_row.get("HomeID", ""), None),
    ]
    for value in location_fields:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unknown"


def build_game_rows(
    schedule: pd.DataFrame,
    team_features: pd.DataFrame,
    windows: Sequence[int],
    team_city_map: Dict[str, str],
    snapshot_ts: pd.Timestamp,
) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    if team_features.empty:
        raise ValueError(
            "Team features DataFrame is empty; cannot build prediction rows."
        )

    latest_lookup: Dict[str, pd.Series] = {
        str(row["TeamID"]): row for _, row in team_features.iterrows()
    }

    windows = sorted(set(int(w) for w in windows))

    records: List[Dict[str, object]] = []
    for _, sched_row in schedule.iterrows():
        home_id = str(sched_row["HomeID"])
        away_id = str(sched_row["AwayID"])
        home_features = latest_lookup.get(home_id)
        away_features = latest_lookup.get(away_id)
        if home_features is None or away_features is None:
            logging.warning(
                "Missing historical features for matchup %s vs %s",
                home_id,
                away_id,
            )
        record: Dict[str, object] = {
            "GameID": sched_row.get("GameID")
            if pd.notna(sched_row.get("GameID"))
            else f"{snapshot_ts.date()}_{home_id}_{away_id}",
            "HomeID": home_id,
            "AwayID": away_id,
            "GameDate": snapshot_ts.date().isoformat(),
            "Location": _extract_location(sched_row, team_city_map),
            "RecencyWeight": 1.0,
        }

        for metric in mvd.SELF_METRICS:
            home_key_base = f"HOME_SELF_{metric}"
            away_key_base = f"AWAY_SELF_{metric}"
            record[f"{home_key_base}_season"] = (
                home_features.get(f"SELF_{metric}_season")
                if home_features is not None
                else np.nan
            )
            record[f"{away_key_base}_season"] = (
                away_features.get(f"SELF_{metric}_season")
                if away_features is not None
                else np.nan
            )
            for window in windows:
                home_window_col = f"{home_key_base}_r{window}"
                away_window_col = f"{away_key_base}_r{window}"
                record[home_window_col] = (
                    home_features.get(f"SELF_{metric}_r{window}")
                    if home_features is not None
                    else np.nan
                )
                record[away_window_col] = (
                    away_features.get(f"SELF_{metric}_r{window}")
                    if away_features is not None
                    else np.nan
                )

        for metric in mvd.OPP_METRICS:
            home_key_base = f"HOME_OPP_{metric}"
            away_key_base = f"AWAY_OPP_{metric}"
            record[f"{home_key_base}_season"] = (
                home_features.get(f"OPP_{metric}_season")
                if home_features is not None
                else np.nan
            )
            record[f"{away_key_base}_season"] = (
                away_features.get(f"OPP_{metric}_season")
                if away_features is not None
                else np.nan
            )
            for window in windows:
                home_window_col = f"{home_key_base}_r{window}"
                away_window_col = f"{away_key_base}_r{window}"
                record[home_window_col] = (
                    home_features.get(f"OPP_{metric}_r{window}")
                    if home_features is not None
                    else np.nan
                )
                record[away_window_col] = (
                    away_features.get(f"OPP_{metric}_r{window}")
                    if away_features is not None
                    else np.nan
                )

        for suffix, col_name in mvd.WIN_RATE_SUFFIX_MAP.items():
            home_col = f"TeamWin_{suffix}"
            away_col = f"TeamWin_{suffix}"
            home_val = home_features.get(home_col) if home_features is not None else np.nan
            away_val = away_features.get(away_col) if away_features is not None else np.nan
            if pd.notna(home_val) and pd.notna(away_val):
                record[col_name] = home_val - away_val
            else:
                record[col_name] = np.nan

        records.append(record)

    output = pd.DataFrame(records)
    if output.empty:
        return output

    ordered_cols = [col for col in mvd.build_output_columns(windows) if col != "HomeTeamWin"]
    for col in ordered_cols:
        if col not in output.columns:
            output[col] = np.nan
    output = output[ordered_cols]
    output.sort_values(["GameDate", "GameID"], inplace=True)
    output.reset_index(drop=True, inplace=True)
    return output


def main() -> None:
    args = parse_args()
    configure_logging()
    windows = sorted(set(args.windows))
    logging.info("Loading data from %s", args.in_csv)
    raw_df = pd.read_csv(args.in_csv)
    logging.info("Loaded %d rows and %d columns", len(raw_df), len(raw_df.columns))

    alias_df = mvd.apply_aliases(raw_df)
    team_city_map = build_team_city_map(alias_df.copy())

    snapshot_ts = pd.to_datetime(args.snapshot_date, errors="coerce")
    if pd.isna(snapshot_ts):
        raise ValueError(f"Unable to parse snapshot_date {args.snapshot_date}")
    snapshot_ts = snapshot_ts.tz_localize(None) if getattr(snapshot_ts, "tzinfo", None) else snapshot_ts

    df = mvd.maybe_convert_team_view_to_game_level(alias_df)
    mvd.ensure_required_columns(df)
    mvd.try_derive_rate_columns(df)
    mvd.normalize_percent_columns(df)

    df["GameDate_dt"] = df["GameDate"].apply(mvd.parse_game_date)
    df = df[df["GameDate_dt"].notna()].copy()
    df["GameDate"] = df["GameDate_dt"].dt.strftime("%Y-%m-%d")
    df["Season"] = df["GameDate_dt"].apply(mvd.infer_season)
    df["SeasonStartYear"] = df["Season"].apply(mvd.season_start_year)

    if df.empty:
        raise ValueError("No valid games available after parsing dates.")

    current_start_year = df["SeasonStartYear"].max()
    current_season = df.loc[df["SeasonStartYear"] == current_start_year].copy()
    logging.info(
        "Filtering to current season starting %d: %d games", current_start_year, len(current_season)
    )
    if current_season.empty:
        raise ValueError("Current season subset is empty; check the input CSV contents.")

    current_season["HomeTeamWin"] = current_season.apply(mvd.determine_home_win, axis=1)
    current_season = current_season[current_season["HomeTeamWin"].notna()].copy()
    current_season["HomeTeamWin"] = current_season["HomeTeamWin"].astype(float)

    current_season = current_season[current_season["GameDate_dt"] <= snapshot_ts].copy()
    if current_season.empty:
        raise ValueError(
            "No completed games on or before the snapshot date for the current season."
        )

    current_season = current_season.sort_values(
        ["SeasonStartYear", "GameDate_dt", "GameID", "HomeID", "AwayID"],
        kind="mergesort",
    ).reset_index(drop=True)
    current_season["OriginalIndex"] = current_season.index

    mvd.apply_pre_game_elo(current_season)

    team_features = compute_current_team_features(current_season, windows)
    if team_features.empty:
        raise ValueError("Unable to compute team features for the current season.")

    latest = latest_team_features(team_features, snapshot_ts)
    if latest.empty:
        raise ValueError(
            "No historical team features available on or before the snapshot date."
        )

    schedule_df = load_schedule_for_date(args.schedule_csv, snapshot_ts)
    logging.info(
        "Found %d scheduled games for %s",
        len(schedule_df),
        snapshot_ts.date().isoformat(),
    )
    if "HomeCity" in schedule_df.columns:
        schedule_df["HomeCity"] = schedule_df["HomeCity"].astype(str)
        for _, row in schedule_df.iterrows():
            city = str(row.get("HomeCity", "")).strip()
            if city and city.lower() not in {"nan", "none"}:
                team_city_map[str(row["HomeID"])] = city
    if "AwayCity" in schedule_df.columns:
        schedule_df["AwayCity"] = schedule_df["AwayCity"].astype(str)

    snapshot_df = build_game_rows(
        schedule_df,
        latest,
        windows,
        team_city_map,
        snapshot_ts,
    )

    snapshot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    logging.info(
        "Writing %d rows x %d columns to %s",
        snapshot_df.shape[0],
        snapshot_df.shape[1],
        args.out_csv,
    )
    snapshot_df.to_csv(args.out_csv, index=False)
    logging.info("Snapshot build complete")


if __name__ == "__main__":
    main()
