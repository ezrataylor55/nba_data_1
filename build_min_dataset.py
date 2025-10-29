"""
Smoke test (fast):
  python build_min_dataset.py --smoke --sleep 1.0
Full (may take a while):
  python build_min_dataset.py --start_season 2014 --end_season auto --include_playoffs --sleep 1.0
"""

import argparse
import datetime as _dt
import logging
import sys
import time
from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("Missing dependency: pandas. Install via `pip install pandas`.", file=sys.stderr)
    sys.exit(1)

try:
    from nba_api.stats.endpoints import (
        BoxScoreAdvancedV2,
        BoxScoreFourFactorsV2,
        BoxScoreSummaryV2,
        BoxScoreTraditionalV2,
        LeagueGameLog,
    )
    from nba_api.stats.library.http import NBAStatsHTTP
except ImportError:  # pragma: no cover
    print("Missing dependency: nba_api. Install via `pip install nba_api`.", file=sys.stderr)
    sys.exit(1)

import numpy as np

LOGGER = logging.getLogger("team_games_min")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CUSTOM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
}


def apply_custom_headers() -> None:
    """Ensure all NBAStatsHTTP requests include our custom headers safely."""

    updated_existing = False
    session = getattr(NBAStatsHTTP, "_SESSION", None)
    if session is not None and hasattr(session, "headers"):
        session.headers.update(CUSTOM_HEADERS)
        updated_existing = True

    try:
        original_init = NBAStatsHTTP.__init__
    except AttributeError:  # pragma: no cover - unexpected library changes
        LOGGER.warning("NBAStatsHTTP.__init__ missing; cannot apply custom headers patch.")
        return

    if getattr(original_init, "_custom_headers_patched", False):
        return

    def patched_init(self, *args, **kwargs):  # type: ignore[misc]
        original_init(self, *args, **kwargs)
        session_obj = getattr(self, "session", None)
        if session_obj is not None and hasattr(session_obj, "headers"):
            session_obj.headers.update(CUSTOM_HEADERS)

    patched_init._custom_headers_patched = True  # type: ignore[attr-defined]
    NBAStatsHTTP.__init__ = patched_init  # type: ignore[assignment]

    if not updated_existing:
        try:
            client = NBAStatsHTTP()
            session_obj = getattr(client, "session", None)
            if session_obj is not None and hasattr(session_obj, "headers"):
                session_obj.headers.update(CUSTOM_HEADERS)
        except Exception as exc:  # pragma: no cover - best-effort
            LOGGER.warning("Failed to instantiate NBAStatsHTTP to seed custom headers: %s", exc)


apply_custom_headers()


def parse_args() -> argparse.Namespace:
    today = _dt.date.today()
    default_end = today.year - 1
    parser = argparse.ArgumentParser(description="Build minimal NBA team-game dataset.")
    parser.add_argument("--start_season", type=int, default=2014, help="Season start year (e.g., 2014 for 2014-15).")
    parser.add_argument(
        "--end_season",
        type=str,
        default=str(default_end),
        help="Season end year (e.g., 2023). Use 'auto' for latest completed season.",
    )
    parser.add_argument(
        "--include_playoffs",
        action="store_true",
        default=True,
        help="Include playoff games (default: True).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Limit to the most recent 30 games across seasons for a smoke test.",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep before each stats request.")
    parser.add_argument("--out", type=str, default="team_games_min.csv", help="Output CSV path.")
    parser.add_argument(
        "--no_playoffs",
        dest="include_playoffs",
        action="store_false",
        help="Exclude playoff games.",
    )
    args = parser.parse_args()
    if isinstance(args.end_season, str) and args.end_season.lower() == "auto":
        args.end_season = default_end
    else:
        args.end_season = int(args.end_season)
    if args.end_season < args.start_season:
        parser.error("--end_season must be >= --start_season")
    return args


def season_id_to_str(season_id: str) -> str:
    start_year = int(season_id[-4:])
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def sleep_then(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def call_endpoint(endpoint_cls, sleep_seconds: float, **params):
    backoffs = [0.5, 1.0, 2.0]
    attempts = len(backoffs) + 1
    for attempt in range(attempts):
        delay = sleep_seconds if attempt == 0 else backoffs[attempt - 1]
        sleep_then(delay)
        try:
            return endpoint_cls(**params)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("%s failed for %s (attempt %s/%s): %s", endpoint_cls.__name__, params, attempt + 1, attempts, exc)
    return None


def fetch_game_logs(start_season: int, end_season: int, include_playoffs: bool, sleep_seconds: float) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    season_types = ["Regular Season"] + (["Playoffs"] if include_playoffs else [])
    for season_start in range(start_season, end_season + 1):
        season_str = f"{season_start}-{str(season_start + 1)[-2:]}"
        for season_type in season_types:
            endpoint = call_endpoint(
                LeagueGameLog,
                sleep_seconds,
                season=season_str,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
            )
            if endpoint is None:
                LOGGER.warning("Skipping %s %s due to repeated failures.", season_str, season_type)
                continue
            df = endpoint.get_data_frames()[0]
            if df.empty:
                continue
            df = df.copy()
            df["SEASON"] = season_str
            df["SEASON_TYPE"] = season_type
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    logs = pd.concat(frames, ignore_index=True)
    logs = logs.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["IS_HOME"] = logs["MATCHUP"].str.contains(" vs", regex=False)
    logs["SEASON"] = logs.get("SEASON", logs["SEASON_ID"].apply(season_id_to_str))
    return logs


def extract_team_table(frames: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    for frame in frames:
        if "TEAM_ID" in frame.columns and "PLAYER_ID" not in frame.columns:
            return frame.copy()
    return None


def compute_advanced(row: Dict[str, float]) -> Dict[str, Optional[float]]:
    fga = row.get("FGA", 0) or 0
    fg3m = row.get("FG3M", 0) or 0
    fgm = row.get("FGM", 0) or 0
    fta = row.get("FTA", 0) or 0
    tov = row.get("TOV", 0) or 0
    pts = row.get("PTS", 0) or 0
    denom_efg = fga
    efg = (fgm + 0.5 * fg3m) / denom_efg if denom_efg else np.nan
    denom_ts = 2 * (fga + 0.44 * fta)
    ts = pts / denom_ts if denom_ts else np.nan
    denom_ftr = fga
    ftr = fta / denom_ftr if denom_ftr else np.nan
    denom_tov = fga + 0.44 * fta + tov
    tov_pct = tov / denom_tov if denom_tov else np.nan
    return {"EFG%": efg, "TS%": ts, "FTR": ftr, "TOV%": tov_pct}


def build_rows(game_ids: List[str], logs: pd.DataFrame, sleep_seconds: float) -> List[Dict[str, object]]:
    meta = logs[[
        "GAME_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "IS_HOME",
        "PTS",
        "WL",
    ]].drop_duplicates(subset=["GAME_ID", "TEAM_ID"]).set_index(["GAME_ID", "TEAM_ID"])

    rows: List[Dict[str, object]] = []
    for gid in game_ids:
        trad_ep = call_endpoint(BoxScoreTraditionalV2, sleep_seconds, game_id=gid)
        adv_ep = call_endpoint(BoxScoreAdvancedV2, sleep_seconds, game_id=gid)
        four_ep = call_endpoint(BoxScoreFourFactorsV2, sleep_seconds, game_id=gid)
        summary_ep = call_endpoint(BoxScoreSummaryV2, sleep_seconds, game_id=gid)
        if None in (trad_ep, adv_ep, four_ep, summary_ep):
            LOGGER.warning("Skipping game %s due to missing data endpoints.", gid)
            continue
        trad_df = extract_team_table(trad_ep.get_data_frames())
        adv_df = extract_team_table(adv_ep.get_data_frames())
        four_df = extract_team_table(four_ep.get_data_frames())
        if trad_df is None or adv_df is None or four_df is None:
            LOGGER.warning("Skipping game %s due to incomplete team tables.", gid)
            continue
        trad_df = trad_df[[
            "GAME_ID",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "FGM",
            "FGA",
            "FG3M",
            "FG3A",
            "FTM",
            "FTA",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
        ]]
        adv_cols = {"PACE": "PACE", "OFF_RATING": "ORtg", "DEF_RATING": "DRtg"}
        adv_df = adv_df[["GAME_ID", "TEAM_ID"] + list(adv_cols.keys())].rename(columns=adv_cols)
        four_df = four_df[["GAME_ID", "TEAM_ID", "EFG_PCT", "FTA_RATE", "TM_TOV_PCT"]]
        combined = trad_df.merge(adv_df, on=["GAME_ID", "TEAM_ID"], how="left").merge(
            four_df, on=["GAME_ID", "TEAM_ID"], how="left"
        )
        summary_frames = summary_ep.get_data_frames()
        summary_game = summary_frames[0] if summary_frames else pd.DataFrame()
        season_type_override: Optional[str] = None
        if not summary_game.empty and "GAMECODE" in summary_game.columns:
            # GameCode prefix indicates season type (001 preseason, 002 regular, 003 playoffs)
            code = str(summary_game.iloc[0]["GAMECODE"]) if not summary_game.empty else ""
            if code.startswith("003"):
                season_type_override = "Playoffs"
            elif code.startswith("002"):
                season_type_override = "Regular Season"
        for _, team_row in combined.iterrows():
            key = (gid, int(team_row["TEAM_ID"]))
            if key not in meta.index:
                LOGGER.warning("Metadata missing for game %s team %s; skipping game.", gid, team_row["TEAM_ID"])
                rows = [r for r in rows if r["game_id"] != gid]
                break
            meta_row = meta.loc[key]
            base = {
                "game_id": gid,
                "team_id": int(team_row["TEAM_ID"]),
                "team_abbrev": meta_row["TEAM_ABBREVIATION"],
                "game_date": pd.Timestamp(meta_row["GAME_DATE"]),
                "season": meta_row["SEASON"],
                "season_type": season_type_override or meta_row["SEASON_TYPE"],
                "is_home": bool(meta_row["IS_HOME"]),
                "team_pts": int(meta_row["PTS"]),
                "FGM": float(team_row["FGM"]),
                "FGA": float(team_row["FGA"]),
                "FG3M": float(team_row["FG3M"]),
                "FG3A": float(team_row["FG3A"]),
                "FTM": float(team_row["FTM"]),
                "FTA": float(team_row["FTA"]),
                "OREB": float(team_row["OREB"]),
                "DREB": float(team_row["DREB"]),
                "REB": float(team_row["REB"]),
                "AST": float(team_row["AST"]),
                "STL": float(team_row["STL"]),
                "BLK": float(team_row["BLK"]),
                "TOV": float(team_row["TOV"]),
                "PF": float(team_row["PF"]),
            }
            adv_vals = compute_advanced({
                "FGM": base["FGM"],
                "FG3M": base["FG3M"],
                "FGA": base["FGA"],
                "FTA": base["FTA"],
                "TOV": base["TOV"],
                "PTS": base["team_pts"],
            })
            base.update(adv_vals)
            for metric in ("EFG%", "TS%", "FTR", "TOV%"):
                if not pd.isna(base.get(metric)):
                    base[metric] = float(base[metric])
            replacements = {
                "EFG%": team_row.get("EFG_PCT"),
                "FTR": team_row.get("FTA_RATE"),
                "TOV%": team_row.get("TM_TOV_PCT"),
            }
            for metric, value in replacements.items():
                if not pd.isna(value):
                    base[metric] = float(value)
            base["PACE"] = float(team_row.get("PACE")) if not pd.isna(team_row.get("PACE")) else np.nan
            base["ORtg"] = float(team_row.get("ORtg")) if not pd.isna(team_row.get("ORtg")) else np.nan
            base["DRtg"] = float(team_row.get("DRtg")) if not pd.isna(team_row.get("DRtg")) else np.nan
            net_from_box = team_row.get("NET_RATING")
            if not pd.isna(net_from_box):
                base["NET_RTG"] = float(net_from_box)
            elif not pd.isna(base.get("ORtg")) and not pd.isna(base.get("DRtg")):
                base["NET_RTG"] = float(base["ORtg"] - base["DRtg"])
            else:
                base["NET_RTG"] = np.nan
            if not pd.isna(base["NET_RTG"]):
                base["NET_RTG"] = float(base["NET_RTG"])
            rows.append(base)
    return rows


def finalize_dataset(rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    opp = df[
        [
            "game_id",
            "team_id",
            "team_abbrev",
            "team_pts",
            "EFG%",
            "TS%",
            "TOV%",
            "FTR",
            "PACE",
            "ORtg",
            "DRtg",
            "NET_RTG",
        ]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "team_abbrev": "opponent_abbrev",
            "team_pts": "opp_pts",
            "EFG%": "opp_EFG%",
            "TS%": "opp_TS%",
            "TOV%": "opp_TOV%",
            "FTR": "opp_FTR",
            "PACE": "opp_PACE",
            "ORtg": "opp_ORtg",
            "DRtg": "opp_DRtg",
            "NET_RTG": "opp_NET_RTG",
        }
    )
    merged = df.merge(opp, on="game_id", how="left")
    merged = merged[merged["team_id"] != merged["opponent_id"]]
    merged = merged.drop_duplicates(subset=["game_id", "team_id"])
    merged["opp_pts"] = merged["opp_pts"].astype(float)
    merged["win"] = (merged["team_pts"] > merged["opp_pts"]).astype(int)
    merged["margin"] = merged["team_pts"] - merged["opp_pts"]
    merged["diff_EFG%"] = merged["EFG%"] - merged["opp_EFG%"]
    merged["diff_TS%"] = merged["TS%"] - merged["opp_TS%"]
    merged["diff_TOV%"] = merged["TOV%"] - merged["opp_TOV%"]
    merged["diff_FTR"] = merged["FTR"] - merged["opp_FTR"]
    merged["diff_PACE"] = merged["PACE"] - merged["opp_PACE"]
    merged["diff_NET_RTG"] = merged["NET_RTG"] - merged["opp_NET_RTG"]
    merged = merged.sort_values(["game_date", "game_id", "team_id"])
    merged["team_id"] = merged["team_id"].astype(int)
    merged["opponent_id"] = merged["opponent_id"].astype(int)
    merged["game_date"] = merged["game_date"].dt.strftime("%Y-%m-%d")
    merged["team_pts"] = merged["team_pts"].astype(int)
    merged["opp_pts"] = merged["opp_pts"].round().astype(int)
    merged["margin"] = merged["margin"].astype(int)
    ordered_cols = [
        "game_id",
        "game_date",
        "season",
        "season_type",
        "team_id",
        "team_abbrev",
        "opponent_id",
        "opponent_abbrev",
        "is_home",
        "team_pts",
        "opp_pts",
        "win",
        "margin",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "EFG%",
        "TS%",
        "FTR",
        "TOV%",
        "PACE",
        "ORtg",
        "DRtg",
        "NET_RTG",
        "opp_EFG%",
        "opp_TS%",
        "opp_TOV%",
        "opp_FTR",
        "opp_PACE",
        "opp_ORtg",
        "opp_DRtg",
        "opp_NET_RTG",
        "diff_EFG%",
        "diff_TS%",
        "diff_TOV%",
        "diff_FTR",
        "diff_PACE",
        "diff_NET_RTG",
    ]
    merged = merged[ordered_cols]
    return merged


def main() -> None:
    args = parse_args()
    start_time = _dt.datetime.now()
    print(f"=== START ({start_time:%Y-%m-%d %H:%M}) ===")
    logs = fetch_game_logs(args.start_season, args.end_season, args.include_playoffs, args.sleep)
    if logs.empty:
        LOGGER.error("No games found for given filters.")
        sys.exit(1)
    unique_games = logs[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values("GAME_DATE")
    if args.smoke:
        unique_games = unique_games.tail(30)
    game_ids = unique_games["GAME_ID"].tolist()
    print(f"Enumerated {len(game_ids)} games (after filters)")
    rows = build_rows(game_ids, logs, args.sleep)
    dataset = finalize_dataset(rows)
    if dataset.empty:
        LOGGER.error("No rows compiled; exiting.")
        sys.exit(1)
    processed_games = dataset["game_id"].nunique()
    print(f"Processed {processed_games} games → rows: {len(dataset)}")
    dataset.to_csv(args.out, index=False)
    print(f"=== DONE. Wrote: {args.out} (rows={len(dataset)}) ===")


if __name__ == "__main__":
    main()
