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


def add_season_column(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns and "season" not in df.columns:
        df["season"] = df["game_date"].map(determine_season)
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


def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if isinstance(parsed.dtype, DatetimeTZDtype):
        try:
            return parsed.dt.tz_localize(None)
        except TypeError:
            return parsed.dt.tz_convert(None)
    return parsed


def ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns:
        parsed = _parse_datetime(df["game_date"])
        if parsed.notna().any():
            df["game_date"] = parsed
            return df

    fallback_columns = [
        "game_date_est",
        "game_date_time",
        "game_date_time_est",
        "game_date_time_utc",
        "game_date_utc",
    ]
    for col in fallback_columns:
        if col in df.columns:
            parsed = _parse_datetime(df[col])
            if parsed.notna().any():
                df["game_date"] = parsed
                logging.info("Filled game_date from %s", col)
                return df
    if "game_date" in df.columns:
        df["game_date"] = _parse_datetime(df["game_date"])
    return df


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
        "fouls_personal",
        "rebounds_total",
    ]
    for col in base_numeric:
        if col not in team_df.columns:
            team_df[col] = np.nan
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
    team_df["pf_per_poss"] = safe_div(team_df["fouls_personal"], team_df["poss"])
    team_df["stl_per_poss"] = safe_div(team_df["steals"], team_df["poss"])
    team_df["blk_per_poss"] = safe_div(team_df["blocks"], team_df["poss"])
    team_df["paint_share"] = safe_div(team_df["points_in_the_paint"], team_df["team_score"])
    team_df["fb_share"] = safe_div(team_df["points_fast_break"], team_df["team_score"])
    team_df["tov_points_share"] = safe_div(
        team_df["points_from_turnovers"], team_df["team_score"]
    )

    opp_lookup = team_df[
        [
            "game_id",
            "team_id",
            "poss",
            "rebounds_defensive",
            "rebounds_offensive",
            "rebounds_total",
        ]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "poss": "opponent_poss",
            "rebounds_defensive": "opponent_rebounds_defensive",
            "rebounds_offensive": "opponent_rebounds_offensive",
            "rebounds_total": "opponent_rebounds_total",
        }
    )
    team_df = team_df.merge(
        opp_lookup,
        on=["game_id", "opponent_team_id"],
        how="left",
    )
    team_df["oreb_pct"] = team_df["rebounds_offensive"] / (
        team_df["rebounds_offensive"]
        + team_df["opponent_rebounds_defensive"].fillna(0)
        + 1e-9
    )
    team_df["dreb_pct"] = team_df["rebounds_defensive"] / (
        team_df["rebounds_defensive"]
        + team_df["opponent_rebounds_offensive"].fillna(0)
        + 1e-9
    )
    off_ratio = safe_div(team_df["team_score"], team_df["poss"]).astype(float)
    def_ratio = safe_div(
        team_df["opponent_score"], team_df["opponent_poss"]
    ).astype(float)
    team_df["off_rtg"] = 100.0 * off_ratio
    team_df["def_rtg"] = 100.0 * def_ratio
    team_df["margin"] = team_df["team_score"] - team_df["opponent_score"]
    team_df["net_rtg"] = team_df["off_rtg"] - team_df["def_rtg"]

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

    numeric_cols = [
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

    pre_frames = []
    lookback_days = (3, 5, 7)
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

        basic_features = pd.DataFrame(
            {
                "tm_games_played_pre": np.arange(len(group)),
                "tm_rest_days_pre": group["game_date"].diff()
                .dt.days.astype(float)
                .reset_index(drop=True),
            },
            index=base.index,
        )
        frames.append(basic_features)

        history_dates = group["game_date"].reset_index(drop=True)
        lookback_counts = {window: [] for window in lookback_days}
        for idx, current in enumerate(history_dates):
            if pd.isna(current):
                for window in lookback_days:
                    lookback_counts[window].append(np.nan)
                continue
            prior = history_dates.iloc[:idx]
            for window in lookback_days:
                cutoff = current - pd.Timedelta(days=window)
                count = int((prior >= cutoff).sum()) if idx else 0
                lookback_counts[window].append(count)
        lookback_df = pd.DataFrame(
            {
                f"tm_games_in_last_{window}_days_pre": values
                for window, values in lookback_counts.items()
            },
            index=base.index,
        )
        frames.append(lookback_df)
        win_series = group.get("win")
        if win_series is not None:
            wins = win_series.fillna(np.nan).astype(float)
            win_streak = []
            loss_streak = []
            current_win = 0
            current_loss = 0
            for result in wins:
                win_streak.append(current_win)
                loss_streak.append(current_loss)
                if pd.isna(result):
                    current_win = 0
                    current_loss = 0
                elif result >= 1:
                    current_win += 1
                    current_loss = 0
                else:
                    current_loss += 1
                    current_win = 0
            streak_df = pd.DataFrame(
                {
                    "tm_win_streak_pre": win_streak,
                    "tm_loss_streak_pre": loss_streak,
                },
                index=base.index,
            )
            frames.append(streak_df)
        momentum_windows = sorted(
            set(int(window) for window in roll_windows if window > 0).union({3, 5, 10})
        )
        win_shifted = shifted.get("win")
        if win_shifted is not None:
            win_shifted = win_shifted.astype(float)
            win_filled = win_shifted.fillna(0.0)
            played_indicator = (~win_shifted.isna()).astype(float)
            momentum_dict = {}
            for window in momentum_windows:
                rolling_games = played_indicator.rolling(
                    window=window, min_periods=1
                ).sum()
                rolling_win_sum = win_filled.rolling(window=window, min_periods=1).sum()
                rolling_loss_sum = rolling_games - rolling_win_sum
                rolling_win_pct = safe_div(rolling_win_sum, rolling_games).fillna(0.0)
                momentum_dict[f"tm_momentum_win_pct_r{window}_pre"] = rolling_win_pct
                momentum_dict[f"tm_recent_wins_r{window}_pre"] = rolling_win_sum
                momentum_dict[f"tm_recent_losses_r{window}_pre"] = rolling_loss_sum
            if momentum_dict:
                frames.append(pd.DataFrame(momentum_dict, index=base.index))
        margin_shifted = shifted.get("margin")
        if margin_shifted is not None:
            margin_shifted = margin_shifted.astype(float)
            margin_dict = {}
            for window in momentum_windows:
                margin_dict[f"tm_momentum_margin_r{window}_pre"] = margin_shifted.rolling(
                    window=window, min_periods=1
                ).mean()
            if margin_dict:
                margin_df = pd.DataFrame(margin_dict, index=base.index)
                frames.append(margin_df)
                if {
                    "tm_momentum_margin_r3_pre",
                    "tm_momentum_margin_r10_pre",
                }.issubset(margin_df.columns):
                    trend_df = pd.DataFrame(
                        {
                            "tm_momentum_margin_trend_pre": margin_df[
                                "tm_momentum_margin_r3_pre"
                            ]
                            - margin_df["tm_momentum_margin_r10_pre"],
                        },
                        index=base.index,
                    )
                    frames.append(trend_df)

        expanding_dict = {
            f"tm_{col}_exp_pre": expanding[col]
            for col in numeric_cols
        }
        frames.append(pd.DataFrame(expanding_dict, index=base.index))

        for window in roll_windows:
            rolling = shifted.rolling(window=window, min_periods=1).mean()
            rolling_dict = {
                f"tm_{col}_r{window}_pre": rolling[col] for col in numeric_cols
            }
            frames.append(pd.DataFrame(rolling_dict, index=base.index))

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
    if "game_era" in merged.columns:
        keep_cols.append("game_era")
    keep_cols.extend(
        [
            "home_team_id",
            "away_team_id",
            "label_home_margin",
        ]
    )
    extra_cols = [
        col
        for col in merged.columns
        if col.startswith("home_tm_")
        or col.startswith("away_tm_")
        or col.startswith("delta_")
    ]
    merged = merged[keep_cols + sorted(extra_cols)]
    merged = merged.sort_values(["game_date", "game_id"])
    merged.reset_index(drop=True, inplace=True)
    return merged, helper_rate


def build_player_momentum_features(
    player_df: pd.DataFrame, team_df: pd.DataFrame, roll_windows: Sequence[int]
) -> pd.DataFrame:
    if player_df.empty:
        return pd.DataFrame(columns=["game_id", "team_id"])
    if "person_id" not in player_df.columns:
        logging.warning(
            "player file missing person_id; skipping player momentum features"
        )
        return pd.DataFrame(columns=["game_id", "team_id"])

    lookup_cols = [
        "game_id",
        "team_id",
        "team_city",
        "team_name",
        "game_date",
        "season",
    ]
    team_lookup = team_df[lookup_cols].drop_duplicates()
    merge_keys_left = ["game_id", "playerteam_city", "playerteam_name"]
    missing_keys = [key for key in merge_keys_left if key not in player_df.columns]
    if missing_keys:
        raise KeyError(f"Player statistics missing required columns: {missing_keys}")

    player_extended = player_df.merge(
        team_lookup,
        left_on=merge_keys_left,
        right_on=["game_id", "team_city", "team_name"],
        how="left",
        suffixes=("", "_team"),
    )
    if "season_team" in player_extended.columns:
        if "season" in player_extended.columns:
            player_extended["season"] = player_extended["season"].fillna(
                player_extended["season_team"]
            )
        else:
            player_extended["season"] = player_extended["season_team"]
        player_extended = player_extended.drop(columns=["season_team"])
    if "game_date_team" in player_extended.columns:
        if "game_date" in player_extended.columns:
            player_extended["game_date"] = player_extended["game_date"].fillna(
                player_extended["game_date_team"]
            )
        else:
            player_extended["game_date"] = player_extended["game_date_team"]
        player_extended = player_extended.drop(columns=["game_date_team"])
    if "season" not in player_extended.columns:
        player_extended["season"] = player_extended["game_date"].map(determine_season)
    map_rate = (~player_extended["team_id"].isna()).mean()
    if map_rate < 0.6:
        logging.warning(
            "Only %.0f%% player rows mapped to team_id; skipping player momentum",
            100 * map_rate,
        )
        return pd.DataFrame(columns=["game_id", "team_id"])
    player_extended = player_extended.dropna(subset=["team_id", "person_id"])
    if player_extended.empty:
        return pd.DataFrame(columns=["game_id", "team_id"])

    player_extended["team_id"] = player_extended["team_id"].astype("Int64")
    player_extended["person_id"] = player_extended["person_id"].astype(str)
    player_extended["season"] = player_extended["season"].fillna("unknown")

    stat_cols = [
        col
        for col in [
            "points",
            "assists",
            "num_minutes_float",
            "plus_minus_points",
            "rebounds_total",
            "rebounds_offensive",
            "rebounds_defensive",
        ]
        if col in player_extended.columns
    ]

    momentum_windows = sorted(
        set(int(window) for window in roll_windows if int(window) > 0).union({3, 5, 10})
    )

    records: list[pd.DataFrame] = []
    group_keys = ["person_id", "season"]
    for _, group in player_extended.groupby(group_keys, sort=False):
        group = group.sort_values(["game_date", "game_id"])
        base = group[["game_id", "team_id", "game_date"]].copy()
        base["person_id"] = group["person_id"].iloc[0]
        base["season"] = group["season"].iloc[0]
        base["ply_games_played_pre"] = np.arange(len(group))
        base["ply_rest_days_pre"] = group["game_date"].diff().dt.days.astype(float)

        win_series = group.get("win")
        if win_series is not None:
            wins = win_series.astype(float)
            win_streak = []
            loss_streak = []
            current_win = 0
            current_loss = 0
            for result in wins:
                win_streak.append(current_win)
                loss_streak.append(current_loss)
                if pd.isna(result):
                    current_win = 0
                    current_loss = 0
                elif result >= 1:
                    current_win += 1
                    current_loss = 0
                else:
                    current_loss += 1
                    current_win = 0
            base["ply_win_streak_pre"] = win_streak
            base["ply_loss_streak_pre"] = loss_streak

        if stat_cols:
            shifted = group[stat_cols].shift(1)
            expanding = shifted.expanding(min_periods=1).mean()
            for col in stat_cols:
                base[f"ply_{col}_exp_pre"] = expanding[col]
            for window in momentum_windows:
                rolling = shifted.rolling(window=window, min_periods=1).mean()
                for col in stat_cols:
                    base[f"ply_{col}_r{window}_pre"] = rolling[col]

        if win_series is not None:
            win_shifted = win_series.shift(1).astype(float)
            played_indicator = (~win_shifted.isna()).astype(float)
            win_values = win_shifted.fillna(0.0)
            for window in momentum_windows:
                games = played_indicator.rolling(window=window, min_periods=1).sum()
                win_sum = win_values.rolling(window=window, min_periods=1).sum()
                loss_sum = games - win_sum
                win_pct = safe_div(win_sum, games).fillna(0.0)
                base[f"ply_win_pct_r{window}_pre"] = win_pct
                base[f"ply_recent_wins_r{window}_pre"] = win_sum
                base[f"ply_recent_losses_r{window}_pre"] = loss_sum

        records.append(base)

    if not records:
        return pd.DataFrame(columns=["game_id", "team_id"])

    player_pre = pd.concat(records, ignore_index=True)
    numeric_cols = [
        col
        for col in player_pre.columns
        if col
        not in {"game_id", "team_id", "game_date", "season", "person_id"}
        and pd.api.types.is_numeric_dtype(player_pre[col])
    ]

    if not numeric_cols:
        return pd.DataFrame(columns=["game_id", "team_id"])

    agg_dict = {col: ["mean", "max"] for col in numeric_cols}
    team_agg = player_pre.groupby(["game_id", "team_id"]).agg(agg_dict)
    team_agg.columns = [f"tm_{col}_{stat}" for col, stat in team_agg.columns]
    team_agg.reset_index(inplace=True)
    team_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    return team_agg


def build_team_game_feature_table(
    team_df: pd.DataFrame, player_df: pd.DataFrame
) -> pd.DataFrame:
    player_agg = aggregate_player_features(player_df)
    lookup_cols = ["game_id", "team_id", "team_city", "team_name"]
    team_lookup = team_df[lookup_cols].drop_duplicates()
    player_agg = player_agg.merge(team_lookup, on=["game_id", "team_city", "team_name"], how="left")
    team_features = team_df.merge(
        player_agg,
        on=["game_id", "team_id", "team_city", "team_name"],
        how="left",
    )
    team_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return team_features


def limit_smoke(games_df: pd.DataFrame, team_df: pd.DataFrame, player_df: pd.DataFrame, smoke: int):
    if smoke is None:
        return games_df, team_df, player_df
    logging.info("Running in SMOKE mode, limiting to first %s games", smoke)
    ordered_games = games_df.sort_values("game_date")
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
        default=[3, 10],
        help="Rolling window sizes in games",
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
    games_df = add_season_column(games_df)
    bad_dates = games_df["game_date"].isna()
    if bad_dates.any():
        logging.warning(
            "Dropping %d rows with NaT game_date from games_df", bad_dates.sum()
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
    team_df = load_table(args.team)
    player_df = load_table(args.player)
    team_df = add_season_column(team_df)
    player_df = add_season_column(player_df)

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

    roll_windows = sorted(set(int(x) for x in args.roll_windows if int(x) > 0))
    player_momentum_features = build_player_momentum_features(
        player_df, team_df, roll_windows
    )
    pre_features = make_pre_features(team_game_features, roll_windows)
    if not player_momentum_features.empty:
        pre_features = pre_features.merge(
            player_momentum_features, on=["game_id", "team_id"], how="left"
        )

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

