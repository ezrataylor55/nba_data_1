#!/usr/bin/env python3
"""Build a Vertex AI AutoML ready dataset with season-to-date and rolling features."""

import argparse
import logging
from contextlib import suppress
from datetime import datetime
import math
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Ensure columns exist by adding them in a single batch assignment."""

    ordered_missing = [
        col for col in dict.fromkeys(columns) if col not in df.columns
    ]
    if ordered_missing:
        df.loc[:, ordered_missing] = np.nan


def mask_by_games_played(
    series: pd.Series, season_counts: pd.Series, required_games: int
) -> pd.Series:
    """Blank out values until the requisite number of games have been played."""

    if series.empty:
        return series
    mask = season_counts >= required_games
    return series.where(mask)

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

PER_POSSESSION_METRICS: Sequence[str] = (
    "ThreePmPerPoss",
    "ThreePaPerPoss",
    "ReboundsPerPoss",
    "AssistsPerPoss",
    "BlocksPerPoss",
    "StealsPerPoss",
    "PfPerPoss",
    "TurnoversPerPoss",
)

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
    "Points",
    "ThreePmPerPoss",
    "ThreePaPerPoss",
    "ThreePPercentage",
    "TrueShootingPercentage",
    "TotalReboundPercentage",
    "ReboundsPerPoss",
    "AssistPercentage",
    "AssistToTurnover",
    "AssistsPerPoss",
    "BlocksPerPoss",
    "StealsPerPoss",
    "FtaPerFga",
    "PfPerPoss",
    "TurnoversPerPoss",
)
OPP_METRICS: Sequence[str] = (
    "EFG%",
    "TO%",
    "FTR",
    "OffRating",
    "DefRating",
    "NetRating",
    "Pace",
    "Points",
    "ThreePmPerPoss",
    "ThreePaPerPoss",
    "ThreePPercentage",
    "TrueShootingPercentage",
    "TotalReboundPercentage",
    "ReboundsPerPoss",
    "AssistPercentage",
    "AssistToTurnover",
    "AssistsPerPoss",
    "BlocksPerPoss",
    "StealsPerPoss",
    "FtaPerFga",
    "PfPerPoss",
    "TurnoversPerPoss",
)

OUTPUT_SELF_METRICS: Sequence[str] = tuple(
    metric for metric in SELF_METRICS if metric not in PER_POSSESSION_METRICS
)
OUTPUT_OPP_METRICS: Sequence[str] = tuple(
    metric for metric in OPP_METRICS if metric not in PER_POSSESSION_METRICS
)

SCHEDULE_DENSITY_FEATURE = "schedule_density_index"
SCHEDULE_PRESSURE_FEATURE = "schedule_pressure_index"
REST_FEATURES: Sequence[str] = (
    "days_since_last_game",
    "is_back_to_back",
    SCHEDULE_DENSITY_FEATURE,
    "rest_stress_score",
    "days_until_next_game",
    "home_game_streak",
    "away_game_streak",
    SCHEDULE_PRESSURE_FEATURE,
)
TRAVEL_BASE_FEATURES: Sequence[str] = (
    "travel_miles_since_last_game",
    "travel_timezone_change",
    "travel_altitude_change",
    "travel_fatigue_score",
)
TRAVEL_AGG_FEATURES: Sequence[str] = ("travel_fatigue_score",)
INJURY_FEATURES: Sequence[str] = ("injury_proxy",)
INJURY_WINDOWS: Sequence[int] = (3, 5, 10)
SUPPORTED_WINDOWS: Sequence[int] = (4, 10, 20)
NUMERIC_IMPUTE_ALLOWLIST: Set[str] = frozenset({"RecencyWeight"})
SCHEDULE_DIFF_PREFIX = "strength_of_schedule_diff"
HEAD_TO_HEAD_COLUMN = "head_to_head_index"
WIN_RATE_SUFFIXES: Sequence[str] = ("season", "r4", "r10", "r20")
WIN_RATE_DIFF_MAP = {
    suffix: f"HomeAwayWinRateDelta_{suffix}" for suffix in WIN_RATE_SUFFIXES
}
WIN_RATE_DIFF_COLS: Sequence[str] = tuple(WIN_RATE_DIFF_MAP.values())

EXCLUDED_FINAL_COLUMNS: Sequence[str] = (
    "home_self_fta_per_fga_season",
    "home_self_fta_per_fga_r10",
    "home_self_fta_per_fga_r4",
    "home_self_fta_per_fga_r20",
    "home_self_efg_pct_season",
    "home_self_efg_pct_r10",
    "home_self_efg_pct_r4",
    "home_self_efg_pct_r20",
    "home_self_ftr_season",
    "home_self_ftr_r10",
    "home_self_ftr_r4",
    "home_self_ftr_r20",
    "home_self_total_rebound_percentage_season",
    "home_self_total_rebound_percentage_r10",
    "home_self_total_rebound_percentage_r4",
    "home_self_total_rebound_percentage_r20",
    "home_self_assist_to_turnover_season",
    "home_self_assist_to_turnover_r10",
    "home_self_assist_to_turnover_r4",
    "home_self_assist_to_turnover_r20",
    "home_self_off_rating_season",
    "home_self_off_rating_r10",
    "home_self_off_rating_r4",
    "home_self_off_rating_r20",
    "home_self_def_rating_season",
    "home_self_def_rating_r10",
    "home_self_def_rating_r4",
    "home_self_def_rating_r20",
    "home_opp_def_rating_season",
    "home_opp_def_rating_r4",
    "home_opp_def_rating_r10",
    "home_opp_def_rating_r20",
    "home_opp_off_rating_r10",
    "home_opp_off_rating_season",
    "home_opp_off_rating_r4",
    "home_opp_off_rating_r20",
    "home_opp_efg_pct_season",
    "home_opp_efg_pct_r10",
    "home_opp_efg_pct_r4",
    "home_opp_efg_pct_r20",
    "home_opp_ftr_season",
    "home_opp_ftr_r10",
    "home_opp_ftr_r4",
    "home_opp_ftr_r20",
    "home_opp_total_rebound_percentage_season",
    "home_opp_fta_per_fga_r10",
    "home_opp_fta_per_fga_r4",
    "home_opp_fta_per_fga_r20",
    "home_opp_fta_per_fga_season",
    "home_opp_assist_to_turnover_season",
    "home_opp_assist_to_turnover_r10",
    "home_opp_assist_to_turnover_r4",
    "home_opp_assist_to_turnover_r20",
    "away_self_fta_per_fga_r20",
    "away_self_fta_per_fga_season",
    "away_self_fta_per_fga_r10",
    "away_self_fta_per_fga_r4",
    "away_self_efg_pct_season",
    "away_self_efg_pct_r10",
    "away_self_efg_pct_r4",
    "away_self_efg_pct_r20",
    "away_self_ftr_season",
    "away_self_ftr_r10",
    "away_self_ftr_r4",
    "away_self_ftr_r20",
    "away_self_total_rebound_percentage_season",
    "away_self_total_rebound_percentage_r10",
    "away_self_total_rebound_percentage_r4",
    "away_self_total_rebound_percentage_r20",
    "away_self_assist_to_turnover_season",
    "away_self_assist_to_turnover_r10",
    "away_self_assist_to_turnover_r4",
    "away_self_assist_to_turnover_r20",
    "away_self_off_rating_season",
    "away_self_off_rating_r10",
    "away_self_off_rating_r4",
    "away_self_off_rating_r20",
    "away_self_def_rating_season",
    "away_self_def_rating_r10",
    "away_self_def_rating_r4",
    "away_self_def_rating_r20",
    "away_opp_def_rating_season",
    "away_opp_def_rating_r4",
    "away_opp_def_rating_r10",
    "away_opp_def_rating_r20",
    "away_opp_off_rating_season",
    "away_opp_off_rating_r4",
    "away_opp_off_rating_r20",
    "away_opp_off_rating_r10",
    "away_opp_efg_pct_season",
    "away_opp_efg_pct_r10",
    "away_opp_efg_pct_r4",
    "away_opp_efg_pct_r20",
    "away_opp_ftr_season",
    "away_opp_ftr_r10",
    "away_opp_ftr_r4",
    "away_opp_ftr_r20",
    "away_opp_fta_per_fga_season",
    "away_opp_fta_per_fga_r4",
    "away_opp_fta_per_fga_r10",
    "away_opp_fta_per_fga_r20",
    "home_away_win_rate_delta_season",
    "home_away_win_rate_delta_r4",
    "home_away_win_rate_delta_r10",
    "home_away_win_rate_delta_r20",
    "away_opp_total_rebound_percentage_season",
    "home_opp_total_rebound_percentage_r4",
    "home_opp_total_rebound_percentage_r20",
    "away_opp_total_rebound_percentage_r20",
    "away_opp_total_rebound_percentage_r10",
    "away_opp_total_rebound_percentage_r4",
    "home_opp_total_rebound_percentage_r10",
    "away_opp_assist_to_turnover_season",
    "away_opp_assist_to_turnover_r10",
    "away_opp_assist_to_turnover_r4",
    "away_opp_assist_to_turnover_r20",
    "away_opp_pace_season",
    "home_opp_pace_season",
    "away_opp_pace_r4",
    "home_opp_pace_r4",
    "away_opp_pace_r20",
    "home_opp_pace_r20",
    "away_opp_pace_r10",
    "home_opp_pace_r10",
    "away_opp_net_rating_season",
    "away_opp_net_rating_r4",
    "away_opp_net_rating_r10",
    "away_opp_net_rating_r20",
    "home_opp_net_rating_r4",
    "home_opp_net_rating_r10",
    "home_opp_net_rating_r20",
    "home_opp_net_rating_season",
    "home_self_pace_season",
    "home_self_pace_r4",
    "home_self_pace_r10",
    "home_self_pace_r20",
    "away_self_pace_season",
    "away_self_pace_r4",
    "away_self_pace_r10",
    "away_self_pace_r20",
    "home_days_since_last_game",
    "away_days_since_last_game",
    "home_days_until_next_game",
    "away_days_until_next_game",
    "home_schedule_density_index",
    "away_schedule_density_index",
    "home_rest_stress_score",
    "away_rest_stress_score",
    "home_home_game_streak",
    "away_away_game_streak",
    "home_travel_miles_since_last_game",
    "away_travel_miles_since_last_game",
    "home_travel_timezone_change",
    "away_travel_timezone_change",
    "home_travel_altitude_change",
    "away_travel_altitude_change",
)

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
    "FG3A": [
        "FG3A",
        "FG3_ATTEMPTED",
        "3PA",
        "FG3A_TOTAL",
        "threePointersAttempted",
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
    "AST": [
        "AST",
        "ASSISTS",
        "AST_TOTAL",
        "assists",
    ],
    "STL": [
        "STL",
        "STEALS",
        "STL_TOTAL",
        "steals",
    ],
    "BLK": [
        "BLK",
        "BLOCKS",
        "BLK_TOTAL",
        "blocks",
    ],
    "PF": [
        "PF",
        "FOULS_PERSONAL",
        "PF_TOTAL",
        "foulsPersonal",
    ],
}

TEAM_LOOKUP_CANDIDATES: Sequence[Tuple[str, Sequence[str], Sequence[str]]] = (
    ("1610612737", ("atlanta",), ("hawks",)),
    ("1610612738", ("boston",), ("celtics",)),
    ("1610612739", ("cleveland",), ("cavaliers", "cavs")),
    (
        "1610612740",
        ("new orleans", "nola"),
        ("pelicans", "pels", "hornets"),
    ),
    ("1610612741", ("chicago",), ("bulls",)),
    ("1610612742", ("dallas",), ("mavericks", "mavs")),
    ("1610612743", ("denver",), ("nuggets",)),
    (
        "1610612744",
        ("golden state", "goldenstate", "san francisco", "sanfrancisco"),
        ("warriors", "dubs"),
    ),
    ("1610612745", ("houston",), ("rockets",)),
    ("1610612746", ("los angeles", "la"), ("clippers", "clips")),
    ("1610612747", ("los angeles", "la"), ("lakers",)),
    ("1610612748", ("miami",), ("heat",)),
    ("1610612749", ("milwaukee",), ("bucks",)),
    ("1610612750", ("minnesota", "minneapolis"), ("timberwolves", "wolves")),
    (
        "1610612751",
        ("brooklyn", "new jersey", "newjersey", "nj"),
        ("nets",),
    ),
    ("1610612752", ("new york", "ny", "nyc"), ("knicks",)),
    ("1610612753", ("orlando",), ("magic",)),
    ("1610612754", ("indiana", "indianapolis"), ("pacers",)),
    ("1610612755", ("philadelphia", "philly"), ("76ers", "sixers")),
    ("1610612756", ("phoenix",), ("suns",)),
    ("1610612757", ("portland",), ("trail blazers", "trailblazers", "blazers")),
    ("1610612758", ("sacramento",), ("kings",)),
    ("1610612759", ("san antonio", "sanantonio"), ("spurs",)),
    ("1610612760", ("oklahoma city", "okc"), ("thunder",)),
    ("1610612761", ("toronto",), ("raptors",)),
    (
        "1610612762",
        ("utah", "salt lake city", "saltlakecity"),
        ("jazz",),
    ),
    ("1610612763", ("memphis",), ("grizzlies", "grizz")),
    ("1610612764", ("washington", "dc", "d.c."), ("wizards",)),
    ("1610612765", ("detroit",), ("pistons",)),
    (
        "1610612766",
        ("charlotte",),
        ("hornets", "bobcats"),
    ),
)


TEAM_ABBREVIATIONS: Dict[str, Sequence[str]] = {
    "1610612737": ("atl",),
    "1610612738": ("bos",),
    "1610612739": ("cle",),
    "1610612740": ("nop", "no"),
    "1610612741": ("chi",),
    "1610612742": ("dal",),
    "1610612743": ("den",),
    "1610612744": ("gsw", "gs"),
    "1610612745": ("hou",),
    "1610612746": ("lac", "la clippers", "laclippers"),
    "1610612747": ("lal", "la lakers", "lalakers"),
    "1610612748": ("mia",),
    "1610612749": ("mil",),
    "1610612750": ("min", "minn"),
    "1610612751": ("bkn", "bro"),
    "1610612752": ("nyk", "ny", "nyknicks"),
    "1610612753": ("orl",),
    "1610612754": ("ind",),
    "1610612755": ("phi", "phl", "sixers"),
    "1610612756": ("phx",),
    "1610612757": ("por", "ptb"),
    "1610612758": ("sac",),
    "1610612759": ("sas", "sa", "sat"),
    "1610612760": ("okc",),
    "1610612761": ("tor",),
    "1610612762": ("uta", "utahjazz"),
    "1610612763": ("mem",),
    "1610612764": ("was", "wsh", "dc"),
    "1610612765": ("det",),
    "1610612766": ("cha", "cho"),
}


def _normalize_team_token(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _build_team_lookup() -> Tuple[
    Dict[Tuple[str, str], str],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
]:
    city_counts: Dict[str, int] = {}
    for _, cities, _ in TEAM_LOOKUP_CANDIDATES:
        for city in cities:
            normalized = _normalize_team_token(city)
            if not normalized:
                continue
            city_counts[normalized] = city_counts.get(normalized, 0) + 1

    lookup: Dict[Tuple[str, str], str] = {}
    city_token_map: Dict[str, Set[str]] = {}
    name_token_map: Dict[str, Set[str]] = {}
    team_city_tokens: Dict[str, Set[str]] = {}
    team_name_tokens: Dict[str, Set[str]] = {}

    for team_id, cities, names in TEAM_LOOKUP_CANDIDATES:
        normalized_cities = {
            token
            for token in (_normalize_team_token(city) for city in cities)
            if token
        }
        team_city_tokens[team_id] = normalized_cities

        base_name_tokens: Set[str] = {
            token
            for token in (
                _normalize_team_token(name)
                for name in (*names, *TEAM_ABBREVIATIONS.get(team_id, ()))
            )
            if token
        }
        combo_tokens: Set[str] = set()
        for city_token in normalized_cities:
            for name_token in base_name_tokens:
                combo_tokens.add(city_token + name_token)
                combo_tokens.add(name_token + city_token)
        normalized_names = base_name_tokens | {token for token in combo_tokens if token}
        team_name_tokens[team_id] = normalized_names

        for city_token in normalized_cities:
            for name_token in normalized_names:
                lookup[(city_token, name_token)] = team_id
                name_token_map.setdefault(name_token, set()).add(team_id)
                city_token_map.setdefault(city_token, set()).add(team_id)
        for name_token in normalized_names:
            lookup[("", name_token)] = team_id
            name_token_map.setdefault(name_token, set()).add(team_id)
        for city_token in normalized_cities:
            if city_counts.get(city_token, 0) == 1:
                lookup[(city_token, "")] = team_id
            city_token_map.setdefault(city_token, set()).add(team_id)

    unique_city_map = {
        token: next(iter(team_ids))
        for token, team_ids in city_token_map.items()
        if len(team_ids) == 1
    }
    unique_name_map = {
        token: next(iter(team_ids))
        for token, team_ids in name_token_map.items()
        if len(team_ids) == 1
    }

    return (
        lookup,
        city_token_map,
        name_token_map,
        unique_city_map,
        unique_name_map,
        team_city_tokens,
        team_name_tokens,
    )


(
    TEAM_LOOKUP,
    TEAM_CITY_TOKEN_SETS,
    TEAM_NAME_TOKEN_SETS,
    TEAM_CITY_TOKEN_UNIQUE,
    TEAM_NAME_TOKEN_UNIQUE,
    TEAM_CITY_TOKENS_BY_TEAM,
    TEAM_NAME_TOKENS_BY_TEAM,
) = _build_team_lookup()


TEAM_SIDE_PREFIXES = ("HOME", "AWAY")

PERCENT_PATTERN = re.compile(r"^\s*([+-]?[0-9]*\.?[0-9]+)\s*%\s*$")


# Elo decay factors to gradually dampen older results and heavily regress between seasons.
ELO_PER_GAME_DECAY = 0.965
ELO_OFFSEASON_DECAY = 0.3
ELO_BLOWOUT_DECAY_START = 20.0
ELO_BLOWOUT_DECAY_LAMBDA = 0.05
ELO_BLOWOUT_WEIGHT_FLOOR = 0.6
ELO_K_FACTOR = 36.0
LEAGUE_AVG_POINTS_PER_POSSESSION = 1.05

TRAVEL_DISTANCE_NORMALIZER = 500.0
TRAVEL_TIMEZONE_WEIGHT = 0.75
TRAVEL_ALTITUDE_NORMALIZER = 1000.0
TRAVEL_FATIGUE_RESPONSE = 1.0

REST_IDEAL_DAYS_BETWEEN_GAMES = 2.5
REST_ROAD_STREAK_NORMALIZER = 4.0
REST_COMPONENT_WEIGHTS = {
    "short_rest": 0.45,
    "b2b": 0.25,
    "density": 0.2,
    "road_streak": 0.1,
}
SCHEDULE_PRESSURE_FUTURE_WINDOW = 3.5
SCHEDULE_PRESSURE_HOME_RELIEF = 0.5
SCHEDULE_PRESSURE_STREAK_SCALE = 4.0
SCHEDULE_PRESSURE_WEIGHTS = {
    "rest_stress": 0.28,
    "density": 0.18,
    "prior_rest": 0.15,
    "future_congestion": 0.14,
    "location_balance": 0.1,
    "travel_load": 0.15,
}
REST_SCORE_MIN_SPREAD = 0.05
SCHEDULE_PRESSURE_MIN_SPREAD = 0.05
FATIGUE_MIN_SPREAD = 0.05
INJURY_PROXY_MIN_SPREAD = 0.05
FULL_GAME_MINUTES = 240.0

HEAD_TO_HEAD_DECAY_DAYS = 365.0
HEAD_TO_HEAD_PRIOR_WEIGHT = 1.25
HEAD_TO_HEAD_MARGIN_SCALE = 12.0
HEAD_TO_HEAD_STREAK_SCALE = 3.0
HEAD_TO_HEAD_VOLUME_SCALE = 3.5
HEAD_TO_HEAD_WIN_SCALER = 1.6
HEAD_TO_HEAD_LOCATION_SCALE = 2.5
HEAD_TO_HEAD_DECISIVENESS_NEUTRAL = 8.0
HEAD_TO_HEAD_DECISIVENESS_SCALE = 6.0
HEAD_TO_HEAD_COMPONENT_WEIGHTS = {
    "win": 0.4,
    "margin": 0.2,
    "streak": 0.12,
    "volume": 0.1,
    "location": 0.1,
    "decisiveness": 0.08,
}


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
        default=list(SUPPORTED_WINDOWS),
        help=(
            "Rolling window sizes for features (values outside 4, 10, 20 will be ignored)"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke-test mode to limit rows for a lightweight end-to-end run",
    )
    parser.add_argument(
        "--player_csv",
        help="Optional path to a player-level statistics CSV for injury proxies",
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


def parse_minutes_value(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    upper = text.upper()
    iso_match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", upper)
    if iso_match:
        hours = int(iso_match.group(1) or 0)
        minutes = int(iso_match.group(2) or 0)
        seconds = int(iso_match.group(3) or 0)
        total_minutes = hours * 60 + minutes + seconds / 60.0
        return float(total_minutes)
    if ":" in text:
        parts = text.split(":")
        try:
            parts = [float(part) for part in parts]
        except ValueError:
            return np.nan
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes + seconds / 60.0)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours * 60 + minutes + seconds / 60.0)
    try:
        return float(text)
    except ValueError:
        return np.nan


def resolve_player_team_id(city: object, name: object) -> Optional[str]:
    city_token = _normalize_team_token(city)
    name_token = _normalize_team_token(name)
    if not (city_token or name_token):
        return None

    for key in (
        (city_token, name_token),
        ("", name_token),
        (city_token, ""),
    ):
        if key in TEAM_LOOKUP:
            return TEAM_LOOKUP[key]

    if name_token and name_token in TEAM_NAME_TOKEN_UNIQUE:
        team_id = TEAM_NAME_TOKEN_UNIQUE[name_token]
        if not city_token or city_token in TEAM_CITY_TOKENS_BY_TEAM.get(team_id, set()):
            return team_id

    if city_token and city_token in TEAM_CITY_TOKEN_UNIQUE:
        team_id = TEAM_CITY_TOKEN_UNIQUE[city_token]
        if not name_token or team_id in TEAM_NAME_TOKENS_BY_TEAM:
            if not name_token or any(
                token and token in name_token
                for token in TEAM_NAME_TOKENS_BY_TEAM.get(team_id, set())
            ):
                return team_id

    candidates: Set[str] = set()
    if name_token:
        candidates.update(TEAM_NAME_TOKEN_SETS.get(name_token, set()))
        if not candidates:
            for team_id, tokens in TEAM_NAME_TOKENS_BY_TEAM.items():
                if any(token and token in name_token for token in tokens):
                    candidates.add(team_id)

    if city_token:
        city_candidates = TEAM_CITY_TOKEN_SETS.get(city_token, set())
        if candidates:
            if city_candidates:
                candidates &= city_candidates
        elif city_candidates:
            candidates.update(city_candidates)

    if len(candidates) == 1:
        return next(iter(candidates))

    return None


def rolling_unique_players(rosters: pd.Series, window: int) -> pd.Series:
    values: List[float] = []
    roster_list = rosters.tolist()
    for idx in range(len(roster_list)):
        if idx < window:
            values.append(np.nan)
            continue
        window_sets = roster_list[idx - window : idx]
        if any(not isinstance(r, frozenset) for r in window_sets):
            values.append(np.nan)
            continue
        combined: Set[str] = set()
        for roster_set in window_sets:
            combined.update(roster_set)
        values.append(float(len(combined)))
    return pd.Series(values, index=rosters.index, dtype=float)


def min_max_scale_series(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    min_val = valid.min()
    max_val = valid.max()
    if np.isclose(max_val, min_val):
        return pd.Series(0.0, index=series.index)
    scaled = (series - min_val) / (max_val - min_val)
    return scaled.clip(0.0, 1.0)


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


def parse_game_date_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype="datetime64[ns]")

    result = pd.Series(pd.NaT, index=series.index)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        numeric_strings = numeric.loc[numeric_mask].astype(int).astype(str).str.zfill(8)
        parsed_numeric = pd.to_datetime(
            numeric_strings, format="%Y%m%d", errors="coerce"
        )
        result.loc[numeric_mask] = parsed_numeric.values

    remainder_mask = ~numeric_mask
    if remainder_mask.any():
        text_values = series.loc[remainder_mask].astype(str).str.strip()
        parsed_strings = pd.to_datetime(text_values, errors="coerce")
        with suppress(AttributeError, TypeError):
            parsed_strings = parsed_strings.dt.tz_convert(None)
        with suppress(AttributeError, TypeError):
            parsed_strings = parsed_strings.dt.tz_localize(None)
        result.loc[remainder_mask] = parsed_strings.values

    return pd.to_datetime(result, errors="coerce")


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
    k_factor: float = ELO_K_FACTOR,
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

    home_fg3a = _stat_series_with_suffix(merged, "FG3A", "home")
    away_fg3a = _stat_series_with_suffix(merged, "FG3A", "away")
    home_ast = _stat_series_with_suffix(merged, "AST", "home")
    away_ast = _stat_series_with_suffix(merged, "AST", "away")
    home_stl = _stat_series_with_suffix(merged, "STL", "home")
    away_stl = _stat_series_with_suffix(merged, "STL", "away")
    home_blk = _stat_series_with_suffix(merged, "BLK", "home")
    away_blk = _stat_series_with_suffix(merged, "BLK", "away")
    home_pf = _stat_series_with_suffix(merged, "PF", "home")
    away_pf = _stat_series_with_suffix(merged, "PF", "away")

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

    home_total_reb = home_orb + home_drb
    away_total_reb = away_orb + away_drb

    home_three_pm_per_poss = _safe_ratio(home_fg3m, home_possessions)
    away_three_pm_per_poss = _safe_ratio(away_fg3m, away_possessions)

    home_three_pa_per_poss = _safe_ratio(home_fg3a, home_possessions)
    away_three_pa_per_poss = _safe_ratio(away_fg3a, away_possessions)

    home_three_p_percentage = _safe_ratio(home_fg3m, home_fg3a)
    away_three_p_percentage = _safe_ratio(away_fg3m, away_fg3a)

    home_ts_denom = 2.0 * (home_fga + 0.44 * home_fta)
    away_ts_denom = 2.0 * (away_fga + 0.44 * away_fta)
    home_true_shooting = _safe_ratio(home_scores, home_ts_denom)
    away_true_shooting = _safe_ratio(away_scores, away_ts_denom)

    home_total_reb_pct = _safe_ratio(home_total_reb, home_total_reb + away_total_reb)
    away_total_reb_pct = _safe_ratio(away_total_reb, home_total_reb + away_total_reb)

    home_reb_per_poss = _safe_ratio(home_total_reb, home_possessions)
    away_reb_per_poss = _safe_ratio(away_total_reb, away_possessions)

    home_assist_pct = _safe_ratio(home_ast, home_fgm)
    away_assist_pct = _safe_ratio(away_ast, away_fgm)

    home_assist_to = _safe_ratio(home_ast, home_turnovers)
    away_assist_to = _safe_ratio(away_ast, away_turnovers)

    home_assist_per_poss = _safe_ratio(home_ast, home_possessions)
    away_assist_per_poss = _safe_ratio(away_ast, away_possessions)

    home_blocks_per_poss = _safe_ratio(home_blk, home_possessions)
    away_blocks_per_poss = _safe_ratio(away_blk, away_possessions)

    home_steals_per_poss = _safe_ratio(home_stl, home_possessions)
    away_steals_per_poss = _safe_ratio(away_stl, away_possessions)

    home_pf_per_poss = _safe_ratio(home_pf, home_possessions)
    away_pf_per_poss = _safe_ratio(away_pf, away_possessions)

    home_turnovers_per_poss = _safe_ratio(home_turnovers, home_possessions)
    away_turnovers_per_poss = _safe_ratio(away_turnovers, away_possessions)

    home_fta_per_fga = _safe_ratio(home_fta, home_fga)
    away_fta_per_fga = _safe_ratio(away_fta, away_fga)

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
            "HOME_SELF_Points": home_scores,
            "AWAY_SELF_Points": away_scores,
            "HOME_OPP_Points": away_scores,
            "AWAY_OPP_Points": home_scores,
            "HOME_SELF_ThreePmPerPoss": home_three_pm_per_poss,
            "AWAY_SELF_ThreePmPerPoss": away_three_pm_per_poss,
            "HOME_OPP_ThreePmPerPoss": away_three_pm_per_poss,
            "AWAY_OPP_ThreePmPerPoss": home_three_pm_per_poss,
            "HOME_SELF_ThreePaPerPoss": home_three_pa_per_poss,
            "AWAY_SELF_ThreePaPerPoss": away_three_pa_per_poss,
            "HOME_OPP_ThreePaPerPoss": away_three_pa_per_poss,
            "AWAY_OPP_ThreePaPerPoss": home_three_pa_per_poss,
            "HOME_SELF_ThreePPercentage": home_three_p_percentage,
            "AWAY_SELF_ThreePPercentage": away_three_p_percentage,
            "HOME_OPP_ThreePPercentage": away_three_p_percentage,
            "AWAY_OPP_ThreePPercentage": home_three_p_percentage,
            "HOME_SELF_TrueShootingPercentage": home_true_shooting,
            "AWAY_SELF_TrueShootingPercentage": away_true_shooting,
            "HOME_OPP_TrueShootingPercentage": away_true_shooting,
            "AWAY_OPP_TrueShootingPercentage": home_true_shooting,
            "HOME_SELF_TotalReboundPercentage": home_total_reb_pct,
            "AWAY_SELF_TotalReboundPercentage": away_total_reb_pct,
            "HOME_OPP_TotalReboundPercentage": away_total_reb_pct,
            "AWAY_OPP_TotalReboundPercentage": home_total_reb_pct,
            "HOME_SELF_ReboundsPerPoss": home_reb_per_poss,
            "AWAY_SELF_ReboundsPerPoss": away_reb_per_poss,
            "HOME_OPP_ReboundsPerPoss": away_reb_per_poss,
            "AWAY_OPP_ReboundsPerPoss": home_reb_per_poss,
            "HOME_SELF_AssistPercentage": home_assist_pct,
            "AWAY_SELF_AssistPercentage": away_assist_pct,
            "HOME_OPP_AssistPercentage": away_assist_pct,
            "AWAY_OPP_AssistPercentage": home_assist_pct,
            "HOME_SELF_AssistToTurnover": home_assist_to,
            "AWAY_SELF_AssistToTurnover": away_assist_to,
            "HOME_OPP_AssistToTurnover": away_assist_to,
            "AWAY_OPP_AssistToTurnover": home_assist_to,
            "HOME_SELF_AssistsPerPoss": home_assist_per_poss,
            "AWAY_SELF_AssistsPerPoss": away_assist_per_poss,
            "HOME_OPP_AssistsPerPoss": away_assist_per_poss,
            "AWAY_OPP_AssistsPerPoss": home_assist_per_poss,
            "HOME_SELF_BlocksPerPoss": home_blocks_per_poss,
            "AWAY_SELF_BlocksPerPoss": away_blocks_per_poss,
            "HOME_OPP_BlocksPerPoss": away_blocks_per_poss,
            "AWAY_OPP_BlocksPerPoss": home_blocks_per_poss,
            "HOME_SELF_StealsPerPoss": home_steals_per_poss,
            "AWAY_SELF_StealsPerPoss": away_steals_per_poss,
            "HOME_OPP_StealsPerPoss": away_steals_per_poss,
            "AWAY_OPP_StealsPerPoss": home_steals_per_poss,
            "HOME_SELF_FtaPerFga": home_fta_per_fga,
            "AWAY_SELF_FtaPerFga": away_fta_per_fga,
            "HOME_OPP_FtaPerFga": away_fta_per_fga,
            "AWAY_OPP_FtaPerFga": home_fta_per_fga,
            "HOME_SELF_PfPerPoss": home_pf_per_poss,
            "AWAY_SELF_PfPerPoss": away_pf_per_poss,
            "HOME_OPP_PfPerPoss": away_pf_per_poss,
            "AWAY_OPP_PfPerPoss": home_pf_per_poss,
            "HOME_SELF_TurnoversPerPoss": home_turnovers_per_poss,
            "AWAY_SELF_TurnoversPerPoss": away_turnovers_per_poss,
            "HOME_OPP_TurnoversPerPoss": away_turnovers_per_poss,
            "AWAY_OPP_TurnoversPerPoss": home_turnovers_per_poss,
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
    season_counts = grouped.cumcount()
    metrics_to_process = [f"SELF_{m}" for m in SELF_METRICS] + [f"OPP_{m}" for m in OPP_METRICS]
    metric_results: Dict[str, pd.Series] = {}
    for metric in metrics_to_process:
        if metric not in team_df.columns:
            continue
        metric_name = metric.split("_", 1)[1] if "_" in metric else metric
        season_series = grouped[metric].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        if metric_name != "Elo":
            season_series = mask_by_games_played(season_series, season_counts, 1)
        metric_results[f"{metric}_season"] = season_series
        for window in windows:
            rolling_series = grouped[metric].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
            )
            if metric_name != "Elo":
                rolling_series = mask_by_games_played(
                    rolling_series, season_counts, window
                )
            metric_results[f"{metric}_r{window}"] = rolling_series
    if metric_results:
        team_df = pd.concat(
            [team_df, pd.DataFrame(metric_results, index=team_df.index)], axis=1
        )
    ensure_columns(team_df, list(REST_FEATURES) + list(TRAVEL_BASE_FEATURES))
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
        with np.errstate(divide="ignore", invalid="ignore"):
            density_values = np.maximum(counts_4 / 4.0, counts_6 / 6.0)
        density_values = np.clip(density_values.astype(float), 0.0, 1.0)
        team_df.loc[group_sorted.index, SCHEDULE_DENSITY_FEATURE] = density_values
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
        short_rest_component = pd.Series(np.nan, index=group_sorted.index, dtype=float)
        if days_since.notna().any():
            short_rest_component = (
                (REST_IDEAL_DAYS_BETWEEN_GAMES - days_since)
                / REST_IDEAL_DAYS_BETWEEN_GAMES
            ).clip(lower=0.0, upper=1.0)
        b2b_series = pd.Series(b2b, index=group_sorted.index, dtype=float)
        density_series = pd.Series(
            density_values, index=group_sorted.index, dtype=float
        )
        road_component = (
            pd.Series(away_streak, index=group_sorted.index, dtype=float)
            / REST_ROAD_STREAK_NORMALIZER
        ).clip(lower=0.0, upper=1.0)
        rest_components = pd.DataFrame(
            {
                "short_rest": short_rest_component,
                "b2b": b2b_series,
                "density": density_series,
                "road_streak": road_component,
            }
        )
        weight_series = pd.Series(REST_COMPONENT_WEIGHTS)
        weighted_sum = rest_components.mul(weight_series, axis=1).sum(
            axis=1, skipna=True
        )
        available_weights = rest_components.notna().mul(weight_series, axis=1).sum(axis=1)
        rest_score = weighted_sum.divide(available_weights)
        rest_score = rest_score.where(available_weights > 0)
        rest_score = rest_score.where(days_since.notna())
        rest_score = rest_score.clip(0.0, 1.0)
        team_df.loc[group_sorted.index, "rest_stress_score"] = rest_score
        prior_rest_component = short_rest_component
        future_component = pd.Series(np.nan, index=group_sorted.index, dtype=float)
        if days_until.notna().any():
            future_component = (
                (SCHEDULE_PRESSURE_FUTURE_WINDOW - days_until)
                / SCHEDULE_PRESSURE_FUTURE_WINDOW
            ).clip(lower=0.0, upper=1.0)
        home_series = pd.Series(home_streak, index=group_sorted.index, dtype=float)
        away_series = pd.Series(away_streak, index=group_sorted.index, dtype=float)
        location_balance = (
            away_series - SCHEDULE_PRESSURE_HOME_RELIEF * home_series
        ) / SCHEDULE_PRESSURE_STREAK_SCALE
        location_balance = location_balance.clip(-1.0, 1.0)
        location_balance = (location_balance + 1.0) / 2.0
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
                fatigue_raw = (
                    distance / TRAVEL_DISTANCE_NORMALIZER
                    + tz_change * TRAVEL_TIMEZONE_WEIGHT
                    + alt_change / TRAVEL_ALTITUDE_NORMALIZER
                )
                fatigue = 1.0 - math.exp(-TRAVEL_FATIGUE_RESPONSE * fatigue_raw)
                travel_miles[idx_pos] = distance
                timezone_changes[idx_pos] = tz_change
                altitude_changes[idx_pos] = alt_change
                fatigue_scores[idx_pos] = fatigue
            prev_info = info
        team_df.loc[group_sorted.index, "travel_miles_since_last_game"] = travel_miles
        team_df.loc[group_sorted.index, "travel_timezone_change"] = timezone_changes
        team_df.loc[group_sorted.index, "travel_altitude_change"] = altitude_changes
        team_df.loc[group_sorted.index, "travel_fatigue_score"] = fatigue_scores
        travel_series = pd.Series(fatigue_scores, index=group_sorted.index, dtype=float)
        travel_recent = (
            travel_series.fillna(0.0).shift(1).rolling(window=3, min_periods=1).mean()
        )
        travel_recent = travel_recent.fillna(travel_series.fillna(0.0))
        travel_component = (
            0.6 * travel_series.fillna(0.0) + 0.4 * travel_recent.fillna(0.0)
        )
        travel_component = travel_component.clip(0.0, 1.0)
        pressure_components = pd.DataFrame(
            {
                "rest_stress": rest_score,
                "density": density_series,
                "prior_rest": prior_rest_component,
                "future_congestion": future_component,
                "location_balance": location_balance,
                "travel_load": travel_component,
            }
        )
        pressure_weights = pd.Series(SCHEDULE_PRESSURE_WEIGHTS)
        pressure_weighted = pressure_components.mul(pressure_weights, axis=1).sum(
            axis=1, skipna=True
        )
        pressure_available = (
            pressure_components.notna().mul(pressure_weights, axis=1).sum(axis=1)
        )
        pressure_score = pressure_weighted.divide(pressure_available)
        pressure_score = pressure_score.where(pressure_available > 0)
        pressure_score = pressure_score.clip(0.0, 1.0)
        team_df.loc[group_sorted.index, SCHEDULE_PRESSURE_FEATURE] = pressure_score
    first_game_mask = season_counts == 0
    if first_game_mask.any():
        reset_cols = [
            "days_since_last_game",
            "is_back_to_back",
            SCHEDULE_DENSITY_FEATURE,
            "rest_stress_score",
            SCHEDULE_PRESSURE_FEATURE,
            "travel_miles_since_last_game",
            "travel_timezone_change",
            "travel_altitude_change",
            "travel_fatigue_score",
        ]
        available = [col for col in reset_cols if col in team_df.columns]
        if available:
            team_df.loc[first_game_mask, available] = np.nan
    head_scores = pd.Series(np.nan, index=team_df.index, dtype=float)
    matchup_groups = team_df.groupby(["TeamID", "OpponentID"], sort=False)
    for _, indices in matchup_groups.groups.items():
        subset = team_df.loc[indices].sort_values(
            by=["GameDate", "OriginalIndex"], kind="mergesort"
        )
        if subset.empty:
            continue
        wins = subset["TeamWin"].fillna(0.0).to_numpy(dtype=float, copy=False)
        margins = (
            coerce_numeric(subset["TeamScore"]).fillna(0.0)
            - coerce_numeric(subset["OpponentScore"]).fillna(0.0)
        ).to_numpy(dtype=float, copy=False)
        dates = subset["GameDate"].to_numpy(dtype="datetime64[D]")
        locations = subset["is_home"].astype(float).to_numpy(dtype=float, copy=False)
        streak_values = np.zeros(len(subset), dtype=float)
        running_streak = 0.0
        for idx_pos in range(len(subset)):
            streak_values[idx_pos] = running_streak
            result = wins[idx_pos] >= 0.5
            if result:
                running_streak = running_streak + 1 if running_streak >= 0 else 1.0
            else:
                running_streak = running_streak - 1 if running_streak <= 0 else -1.0
        for idx_pos in range(len(subset)):
            if idx_pos == 0:
                continue
            history_slice = slice(0, idx_pos)
            prior_wins = wins[history_slice]
            prior_margins = margins[history_slice]
            prior_dates = dates[history_slice]
            current_date = dates[idx_pos]
            day_deltas = (current_date - prior_dates).astype("timedelta64[D]")
            day_deltas = day_deltas.astype(float)
            recency_weights = np.exp(-day_deltas / HEAD_TO_HEAD_DECAY_DAYS)
            recency_weights = np.where(np.isfinite(recency_weights), recency_weights, 0.0)
            weight_sum = recency_weights.sum()
            if weight_sum <= 0:
                continue
            smoothed_wins = (
                np.dot(recency_weights, prior_wins) + HEAD_TO_HEAD_PRIOR_WEIGHT * 0.5
            ) / (weight_sum + HEAD_TO_HEAD_PRIOR_WEIGHT)
            margin_avg = np.dot(recency_weights, prior_margins) / weight_sum
            win_component = (smoothed_wins - 0.5) * HEAD_TO_HEAD_WIN_SCALER
            margin_component = margin_avg / HEAD_TO_HEAD_MARGIN_SCALE
            streak_component = streak_values[idx_pos - 1] / HEAD_TO_HEAD_STREAK_SCALE
            volume_component = 1.0 - math.exp(-weight_sum / HEAD_TO_HEAD_VOLUME_SCALE)
            volume_component -= 0.5
            prior_locations = locations[history_slice]
            home_mask = prior_locations >= 0.5
            away_mask = ~home_mask
            home_weight = recency_weights[home_mask].sum()
            away_weight = recency_weights[away_mask].sum()
            if home_weight > 0:
                home_success = (
                    np.dot(recency_weights[home_mask], prior_wins[home_mask])
                    + HEAD_TO_HEAD_PRIOR_WEIGHT * 0.5
                ) / (home_weight + HEAD_TO_HEAD_PRIOR_WEIGHT)
            else:
                home_success = 0.5
            if away_weight > 0:
                away_success = (
                    np.dot(recency_weights[away_mask], prior_wins[away_mask])
                    + HEAD_TO_HEAD_PRIOR_WEIGHT * 0.5
                ) / (away_weight + HEAD_TO_HEAD_PRIOR_WEIGHT)
            else:
                away_success = 0.5
            location_component = (home_success - away_success) * HEAD_TO_HEAD_LOCATION_SCALE
            location_component = float(np.clip(location_component, -1.0, 1.0))
            decisive_avg = np.dot(recency_weights, np.abs(prior_margins)) / weight_sum
            decisiveness_component = math.tanh(
                (decisive_avg - HEAD_TO_HEAD_DECISIVENESS_NEUTRAL)
                / HEAD_TO_HEAD_DECISIVENESS_SCALE
            )
            raw_index = (
                HEAD_TO_HEAD_COMPONENT_WEIGHTS["win"] * win_component
                + HEAD_TO_HEAD_COMPONENT_WEIGHTS["margin"] * margin_component
                + HEAD_TO_HEAD_COMPONENT_WEIGHTS["streak"] * streak_component
                + HEAD_TO_HEAD_COMPONENT_WEIGHTS["volume"] * volume_component
                + HEAD_TO_HEAD_COMPONENT_WEIGHTS["location"] * location_component
                + HEAD_TO_HEAD_COMPONENT_WEIGHTS["decisiveness"]
                * decisiveness_component
            )
            head_scores.loc[subset.index[idx_pos]] = np.tanh(raw_index)
    team_df["head_to_head_index"] = head_scores
    travel_grouped = team_df.groupby(["TeamID", "Season"], sort=False)
    travel_results: Dict[str, pd.Series] = {}
    for feature in TRAVEL_AGG_FEATURES:
        if feature not in team_df.columns:
            continue
        season_series = travel_grouped[feature].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        season_series = mask_by_games_played(season_series, season_counts, 1)
        travel_results[f"{feature}_season"] = season_series
        for window in windows:
            rolling_series = travel_grouped[feature].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
            )
            rolling_series = mask_by_games_played(rolling_series, season_counts, window)
            travel_results[f"{feature}_r{window}"] = rolling_series
    if travel_results:
        team_df = pd.concat(
            [team_df, pd.DataFrame(travel_results, index=team_df.index)], axis=1
        )
    return team_df


def assign_features(
    base_df: pd.DataFrame, team_df: pd.DataFrame, windows: Sequence[int]
) -> None:
    home_features = team_df[team_df["is_home"]].set_index("OriginalIndex")
    away_features = team_df[~team_df["is_home"]].set_index("OriginalIndex")
    required_columns: List[str] = []
    for metric in OUTPUT_SELF_METRICS:
        base = f"SELF_{metric}"
        required_columns.extend(
            [f"HOME_{base}_season", f"AWAY_{base}_season"]
        )
        for window in windows:
            required_columns.extend(
                [f"HOME_{base}_r{window}", f"AWAY_{base}_r{window}"]
            )
    for metric in OUTPUT_OPP_METRICS:
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
    injury_suffixes = [f"_r{window}" for window in INJURY_WINDOWS]
    for feature in INJURY_FEATURES:
        for suffix in injury_suffixes:
            required_columns.extend(
                [f"HOME_{feature}{suffix}", f"AWAY_{feature}{suffix}"]
            )
            diff_col = f"{feature}_diff{suffix}"
            required_columns.append(diff_col)
    poss_diff_suffixes = ["_season"] + [f"_r{window}" for window in windows]
    for metric in PER_POSSESSION_METRICS:
        for suffix in poss_diff_suffixes:
            diff_col = f"{metric}_diff{suffix}"
            required_columns.append(diff_col)
    pace_diff_suffixes = ["_season"] + [f"_r{window}" for window in windows]
    for suffix in pace_diff_suffixes:
        required_columns.append(f"Pace_diff{suffix}")
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

    for metric in OUTPUT_SELF_METRICS:
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
    for metric in OUTPUT_OPP_METRICS:
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
    per_poss_agg_suffixes = ["_season"] + [f"_r{window}" for window in windows]
    for metric in PER_POSSESSION_METRICS:
        for suffix in per_poss_agg_suffixes:
            key = f"SELF_{metric}{suffix}"
            if key in home_features.columns and key in away_features.columns:
                diff_col = f"{metric}_diff{suffix}"
                diff_values = coerce_numeric(home_features[key]) - coerce_numeric(
                    away_features[key]
                )
                base_df.loc[home_features.index, diff_col] = diff_values
    for suffix in pace_diff_suffixes:
        key = f"SELF_Pace{suffix}"
        if key in home_features.columns and key in away_features.columns:
            diff_col = f"Pace_diff{suffix}"
            diff_values = coerce_numeric(home_features[key]) - coerce_numeric(
                away_features[key]
            )
            base_df.loc[home_features.index, diff_col] = diff_values
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
    for feature in INJURY_FEATURES:
        for suffix in injury_suffixes:
            key = f"{feature}{suffix}"
            home_col = f"HOME_{key}"
            away_col = f"AWAY_{key}"
            if key in home_features.columns:
                base_df.loc[home_features.index, home_col] = home_features[key]
            if key in away_features.columns:
                base_df.loc[away_features.index, away_col] = away_features[key]
            diff_col = f"{feature}_diff{suffix}"
            if (
                home_col in base_df.columns
                and away_col in base_df.columns
            ):
                base_df[diff_col] = base_df[home_col] - base_df[away_col]
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
    season_counts = grouped.cumcount()
    season_series = grouped["TeamWin"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    season_series = mask_by_games_played(season_series, season_counts, 1)
    win_rate_results: Dict[str, pd.Series] = {"TeamWin_season": season_series}
    for window in windows:
        rolling_series = grouped["TeamWin"].transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
        )
        rolling_series = mask_by_games_played(rolling_series, season_counts, window)
        win_rate_results[f"TeamWin_r{window}"] = rolling_series
    team_df[list(win_rate_results)] = pd.DataFrame(
        win_rate_results, index=team_df.index
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
    sos_results: Dict[str, pd.Series] = {}
    for suffix in suffixes:
        source_col = f"TeamWin_{suffix}"
        if source_col not in lookup.columns:
            continue
        sos_values = lookup[source_col].reindex(opp_index)
        sos_results[f"strength_of_schedule_{suffix}"] = pd.Series(
            sos_values.values, index=team_df.index
        )
    if sos_results:
        team_df[list(sos_results)] = pd.DataFrame(sos_results)


def integrate_injury_features(
    team_df: pd.DataFrame, windows: Sequence[int], player_csv: Optional[str]
) -> None:
    injury_cols = [f"injury_proxy_r{w}" for w in INJURY_WINDOWS]
    ensure_columns(team_df, injury_cols)

    def _rolling_condition_share(
        series: pd.Series, window: int, limit: float, comparator
    ) -> pd.Series:
        def _evaluate(arr: np.ndarray) -> float:
            if np.isnan(arr).any():
                return np.nan
            return float(np.mean(comparator(arr, limit)))

        return series.rolling(window=window, min_periods=window).apply(
            _evaluate, raw=True
        )

    if not player_csv:
        logging.info(
            "Player statistics path not provided; skipping injury proxy feature generation"
        )
        return
    possible_columns = {
        "gameId",
        "GAME_ID",
        "GameID",
        "personId",
        "PERSON_ID",
        "playerId",
        "PLAYER_ID",
        "playerID",
        "player_id",
        "teamId",
        "TEAM_ID",
        "team_id",
        "playerTeamId",
        "playerteamId",
        "playerteamCity",
        "playerTeamCity",
        "playerteamName",
        "playerTeamName",
        "Minutes",
        "minutes",
        "numMinutes",
    }
    try:
        players = pd.read_csv(
            player_csv,
            usecols=lambda c: c in possible_columns,
            low_memory=False,
        )
    except FileNotFoundError:
        logging.warning(
            "Player statistics file %s not found; skipping injury proxy features", player_csv
        )
        return
    except ValueError:
        try:
            players = pd.read_csv(player_csv, low_memory=False)
        except FileNotFoundError:
            logging.warning(
                "Player statistics file %s not found; skipping injury proxy features",
                player_csv,
            )
            return
    if players.empty:
        logging.warning(
            "Player statistics file %s contains no rows; skipping injury proxy features",
            player_csv,
        )
        return

    rename_map = {
        "gameId": "GameID",
        "GAME_ID": "GameID",
        "personId": "PersonID",
        "PERSON_ID": "PersonID",
        "playerId": "PersonID",
        "PLAYER_ID": "PersonID",
        "playerID": "PersonID",
        "player_id": "PersonID",
        "teamId": "TeamID",
        "TEAM_ID": "TeamID",
        "team_id": "TeamID",
        "playerTeamId": "TeamID",
        "playerteamId": "TeamID",
        "playerteamCity": "PlayerTeamCity",
        "playerTeamCity": "PlayerTeamCity",
        "playerteamName": "PlayerTeamName",
        "playerTeamName": "PlayerTeamName",
        "numMinutes": "Minutes",
        "minutes": "Minutes",
    }
    for source, target in rename_map.items():
        if source in players.columns:
            players.rename(columns={source: target}, inplace=True)

    required_fields = ["GameID", "PersonID", "Minutes"]
    missing_fields = [field for field in required_fields if field not in players.columns]
    if missing_fields:
        logging.warning(
            "Player statistics file %s is missing required columns: %s",
            player_csv,
            ", ".join(sorted(missing_fields)),
        )
        return

    players["Minutes"] = players["Minutes"].apply(parse_minutes_value)
    players = players[players["Minutes"].notna()].copy()
    if players.empty:
        logging.warning(
            "Player statistics file %s has no usable minute data; skipping injury proxy features",
            player_csv,
        )
        return

    players = players[players["Minutes"] > 0].copy()
    if players.empty:
        logging.warning(
            "Player statistics file %s has no positive minute data; skipping injury proxy features",
            player_csv,
        )
        return

    players["GameID"] = players["GameID"].astype(str).str.strip()
    players = players[players["GameID"] != ""].copy()
    players["PersonID"] = players["PersonID"].astype(str).str.strip()
    players = players[players["PersonID"] != ""].copy()
    if players.empty:
        logging.warning(
            "Player statistics rows with usable player identifiers were not found; skipping injury proxy features"
        )
        return

    has_team_metadata = any(
        col in players.columns for col in ("PlayerTeamCity", "PlayerTeamName")
    )

    def _resolve_team_ids(subset: pd.DataFrame) -> pd.Series:
        if subset.empty:
            return pd.Series(dtype=object)
        cities = subset.get("PlayerTeamCity")
        names = subset.get("PlayerTeamName")
        if cities is None and names is None:
            return pd.Series(np.nan, index=subset.index, dtype=object)
        if cities is None:
            cities = pd.Series(np.nan, index=subset.index, dtype=object)
        if names is None:
            names = pd.Series(np.nan, index=subset.index, dtype=object)
        city_values = cities.to_numpy(dtype=object, copy=False)
        name_values = names.to_numpy(dtype=object, copy=False)
        resolved = [
            resolve_player_team_id(city, name)
            for city, name in zip(city_values, name_values)
        ]
        return pd.Series(resolved, index=subset.index, dtype=object)

    if "TeamID" in players.columns:
        players["TeamID"] = players["TeamID"].apply(_to_str_id)
        players["TeamID"] = players["TeamID"].astype(object)
    else:
        players["TeamID"] = pd.Series(np.nan, index=players.index, dtype=object)

    missing_mask = players["TeamID"].isna()
    if missing_mask.any():
        if not has_team_metadata:
            logging.warning(
                "Player statistics file %s is missing team identifiers and team metadata; skipping injury proxy features",
                player_csv,
            )
            return
        resolved_ids = _resolve_team_ids(players.loc[missing_mask])
        if resolved_ids.notna().any():
            players.loc[missing_mask, "TeamID"] = resolved_ids
        remaining = players["TeamID"].isna()
        if remaining.any():
            logging.warning(
                "Unable to map %d player rows to known team identifiers; they will be ignored",
                int(remaining.sum()),
            )
            players = players[~remaining].copy()

    players = players[players["TeamID"].notna()].copy()
    if players.empty:
        logging.warning(
            "Player statistics rows with usable team identifiers were not found; skipping injury proxy features"
        )
        return

    players["TeamID"] = players["TeamID"].astype(str)
    players = players[["GameID", "TeamID", "PersonID", "Minutes"]].copy()

    valid_game_ids = set(team_df["GameID"].astype(str))
    initial_player_rows = len(players)
    players = players[players["GameID"].isin(valid_game_ids)].copy()
    if players.empty:
        logging.info(
            "Player statistics rows do not overlap with filtered team games; skipping injury proxy features",
        )
        return
    trimmed_rows = initial_player_rows - len(players)
    if trimmed_rows > 0:
        logging.debug(
            "Dropped %d player rows outside the modeled schedule window",
            trimmed_rows,
        )

    records: List[Dict[str, object]] = []
    for (game_id, team_id), group in players.groupby(["GameID", "TeamID"], sort=False):
        person_totals = (
            group.groupby("PersonID", sort=False)["Minutes"].sum().astype(float)
        )
        person_ids = person_totals.index.astype(str).tolist()
        minutes_values = person_totals.to_numpy(dtype=float, copy=True)
        roster = frozenset(person_ids)
        total_minutes = np.nan
        dispersion = np.nan
        rotation_players_10 = np.nan
        heavy_minute_share = np.nan
        top3_share = np.nan
        top6_share = np.nan
        bench_minutes_share = np.nan
        if minutes_values.size:
            total_minutes = float(minutes_values.sum())
            dispersion = float(np.std(minutes_values))
            rotation_players_10 = float(np.sum(minutes_values >= 10))
            if total_minutes > 0:
                heavy_minute_share = float(
                    minutes_values[minutes_values >= 28].sum() / total_minutes
                )
                bench_minutes_share = float(
                    minutes_values[minutes_values < 10].sum() / total_minutes
                )
                sorted_minutes = np.sort(minutes_values)[::-1]
                if sorted_minutes.size:
                    top3_share = float(sorted_minutes[:3].sum() / total_minutes)
                    top6_share = float(sorted_minutes[:6].sum() / total_minutes)
        records.append(
            {
                "GameID": game_id,
                "TeamID": team_id,
                "roster_set": roster,
                "total_minutes": total_minutes,
                "minutes_dispersion": dispersion,
                "rotation_players_10": rotation_players_10,
                "heavy_minute_share": heavy_minute_share,
                "top3_share": top3_share,
                "top6_share": top6_share,
                "bench_minutes_share": bench_minutes_share,
            }
        )

    if not records:
        logging.warning(
            "Player statistics aggregates are empty; skipping injury proxy features"
        )
        return

    roster_stats = pd.DataFrame.from_records(records)


    roster_stats["GameID"] = roster_stats["GameID"].astype(str)
    roster_stats["TeamID"] = roster_stats["TeamID"].astype(str)

    ordering_cols = team_df[["GameID", "TeamID", "Season", "GameDate"]].copy()
    ordering_cols["GameID"] = ordering_cols["GameID"].astype(str)
    ordering_cols["TeamID"] = ordering_cols["TeamID"].astype(str)

    roster_stats = roster_stats.merge(
        ordering_cols, on=["GameID", "TeamID"], how="inner"
    )

    if roster_stats.empty:
        logging.warning(
            "Player statistics could not be aligned to team games; skipping injury proxy features"
        )
        return

    roster_stats.sort_values(
        ["TeamID", "Season", "GameDate", "GameID"], inplace=True
    )
    roster_stats.reset_index(drop=True, inplace=True)

    instability_values = np.full(len(roster_stats), np.nan)
    group_indices = roster_stats.groupby(["TeamID", "Season"], sort=False).indices
    for indices in group_indices.values():
        prev_roster: Optional[frozenset] = None
        for idx in indices:
            roster = roster_stats.at[idx, "roster_set"]
            if not isinstance(roster, frozenset) or not roster:
                instability_values[idx] = np.nan
                prev_roster = None
                continue
            if prev_roster is None or not prev_roster:
                instability_values[idx] = np.nan
            else:
                returning = len(roster & prev_roster)
                churn = (
                    1.0 - returning / len(roster)
                    if len(roster)
                    else np.nan
                )
                prev_size = len(prev_roster)
                curr_size = len(roster)
                if prev_size and curr_size:
                    unique_diff = abs(curr_size - prev_size) / max(curr_size, prev_size)
                else:
                    unique_diff = 0.0
                if not np.isfinite(churn):
                    instability_values[idx] = np.nan
                else:
                    instability = 0.7 * churn + 0.3 * unique_diff
                    instability_values[idx] = float(np.clip(instability, 0.0, 1.0))
            prev_roster = roster

    roster_stats["instability_raw"] = instability_values

    for window in INJURY_WINDOWS:
        roster_stats[f"injury_proxy_r{window}"] = np.nan

    for indices in group_indices.values():
        totals_series = roster_stats.loc[indices, "total_minutes"].astype(float)
        dispersion_series = roster_stats.loc[indices, "minutes_dispersion"].astype(float)
        instability_series = roster_stats.loc[indices, "instability_raw"].astype(float)
        roster_series = roster_stats.loc[indices, "roster_set"]
        rotation_series = roster_stats.loc[indices, "rotation_players_10"].astype(float)
        heavy_share_series = roster_stats.loc[indices, "heavy_minute_share"].astype(float)
        top3_share_series = roster_stats.loc[indices, "top3_share"].astype(float)
        top6_share_series = roster_stats.loc[indices, "top6_share"].astype(float)
        bench_share_series = roster_stats.loc[indices, "bench_minutes_share"].astype(float)
        for window in INJURY_WINDOWS:
            minutes_std = (
                totals_series.shift(1)
                .rolling(window=window, min_periods=window)
                .std()
            )
            dispersion_std = (
                dispersion_series.shift(1)
                .rolling(window=window, min_periods=window)
                .std()
            )
            unique_counts = rolling_unique_players(roster_series.shift(1), window)
            churn_avg = (
                instability_series.shift(1)
                .rolling(window=window, min_periods=window)
                .mean()
            )
            availability_gap = (
                1.0
                - totals_series.shift(1)
                .rolling(window=window, min_periods=window)
                .mean()
                / FULL_GAME_MINUTES
            )
            availability_gap = availability_gap.clip(lower=0.0, upper=1.0)
            prior_rotation = rotation_series.shift(1)
            rotation_volatility = (
                prior_rotation.rolling(window=window, min_periods=window).std()
            )
            rotation_shortage = _rolling_condition_share(
                prior_rotation, window, 7.5, np.less_equal
            ).clip(lower=0.0, upper=1.0)
            prior_heavy = heavy_share_series.shift(1)
            heavy_baseline = prior_heavy.rolling(window=window, min_periods=window).max()
            heavy_deficit = (heavy_baseline - prior_heavy).clip(lower=0.0)
            prior_top3 = top3_share_series.shift(1)
            top3_baseline = prior_top3.rolling(window=window, min_periods=window).mean()
            top3_deficit = (top3_baseline - prior_top3).clip(lower=0.0)
            prior_top6 = top6_share_series.shift(1)
            top6_volatility = (
                prior_top6.rolling(window=window, min_periods=window).std()
            )
            prior_bench = bench_share_series.shift(1)
            bench_spike = _rolling_condition_share(
                prior_bench, window, 0.35, np.greater_equal
            ).clip(lower=0.0, upper=1.0)
            component_df = pd.concat(
                [
                    min_max_scale_series(minutes_std).rename("total_minutes"),
                    min_max_scale_series(dispersion_std).rename(
                        "minutes_dispersion"
                    ),
                    min_max_scale_series(unique_counts).rename("unique_players"),
                    min_max_scale_series(churn_avg).rename("roster_churn"),
                    min_max_scale_series(availability_gap).rename(
                        "availability_gap"
                    ),
                    min_max_scale_series(rotation_volatility).rename(
                        "rotation_volatility"
                    ),
                    rotation_shortage.rename("rotation_shortage_rate"),
                    min_max_scale_series(heavy_deficit).rename(
                        "heavy_minutes_deficit"
                    ),
                    min_max_scale_series(top3_deficit).rename(
                        "top_end_minutes_deficit"
                    ),
                    min_max_scale_series(top6_volatility).rename(
                        "core_share_volatility"
                    ),
                    bench_spike.rename("bench_dependency_spike"),
                ],
                axis=1,
            )
            combined = component_df.mean(axis=1, skipna=True)
            combined = combined.where(component_df.notna().any(axis=1))
            roster_stats.loc[indices, f"injury_proxy_r{window}"] = combined.values

    lookup = roster_stats.set_index(["GameID", "TeamID"])[injury_cols]
    align_index = pd.MultiIndex.from_arrays(
        [team_df["GameID"].astype(str), team_df["TeamID"].astype(str)]
    )
    aligned = lookup.reindex(align_index)
    team_df.loc[:, injury_cols] = aligned.to_numpy()

    primary_window = INJURY_WINDOWS[0]
    primary_col = f"injury_proxy_r{primary_window}"
    populated = int(team_df[primary_col].notna().sum())
    logging.info(
        "Integrated injury proxy features from %s (populated %d team-games)",
        player_csv,
        populated,
    )


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
    rest_cols = [col for col in df.columns if "rest_stress_score" in col]
    pressure_cols = [col for col in df.columns if SCHEDULE_PRESSURE_FEATURE in col]
    for col in rest_cols:
        if ((df[col] < -1e-6) | (df[col] > 1 + 1e-6)).any():
            raise ValueError(f"Rest stress column {col} must be scaled to [0, 1]")
    for col in pressure_cols:
        if ((df[col] < -1e-6) | (df[col] > 1 + 1e-6)).any():
            raise ValueError(
                f"Schedule pressure column {col} must be scaled to [0, 1]"
            )
    injury_pattern = re.compile(r"^(HOME_|AWAY_)?injury_proxy_r\d+")
    injury_cols = [col for col in df.columns if injury_pattern.match(col)]
    for col in injury_cols:
        if ((df[col] < -1e-6) | (df[col] > 1 + 1e-6)).any():
            raise ValueError(
                f"Injury proxy column {col} must be scaled to [0, 1]"
            )
    fatigue_cols = [col for col in df.columns if "travel_fatigue_score" in col]
    for col in fatigue_cols:
        if ((df[col] < -1e-6) | (df[col] > 1 + 1e-6)).any():
            raise ValueError(
                f"Travel fatigue column {col} should be scaled to [0, 1]"
            )
    def _warn_low_spread(columns: Sequence[str], min_spread: float, label: str) -> None:
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            spread = float(series.max() - series.min())
            if spread < min_spread:
                logging.warning(
                    "%s column %s exhibits limited spread (%.4f)",
                    label,
                    col,
                    spread,
                )

    _warn_low_spread(rest_cols, REST_SCORE_MIN_SPREAD, "Rest stress")
    _warn_low_spread(pressure_cols, SCHEDULE_PRESSURE_MIN_SPREAD, "Schedule pressure")
    _warn_low_spread(fatigue_cols, FATIGUE_MIN_SPREAD, "Travel fatigue")
    _warn_low_spread(injury_cols, INJURY_PROXY_MIN_SPREAD, "Injury proxy")
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
        ("HOME_SELF", OUTPUT_SELF_METRICS),
        ("HOME_OPP", OUTPUT_OPP_METRICS),
        ("AWAY_SELF", OUTPUT_SELF_METRICS),
        ("AWAY_OPP", OUTPUT_OPP_METRICS),
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
    for prefix in ("HOME", "AWAY"):
        for feature in INJURY_FEATURES:
            for window in INJURY_WINDOWS:
                columns.append(f"{prefix}_{feature}_r{window}")
    for feature in INJURY_FEATURES:
        for window in INJURY_WINDOWS:
            columns.append(f"{feature}_diff_r{window}")
    poss_diff_suffixes = ["_season"] + [f"_r{window}" for window in windows]
    for metric in PER_POSSESSION_METRICS:
        for suffix in poss_diff_suffixes:
            columns.append(f"{metric}_diff{suffix}")
    pace_diff_suffixes = ["_season"] + [f"_r{window}" for window in windows]
    for suffix in pace_diff_suffixes:
        columns.append(f"Pace_diff{suffix}")
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
    allowlist = [col for col in NUMERIC_IMPUTE_ALLOWLIST if col in df.columns]
    for col in allowlist:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].isna().any():
            median = df[col].median()
            if pd.notna(median):
                df[col] = df[col].fillna(median)


def main() -> None:
    args = parse_args()
    configure_logging()
    requested_windows = sorted(set(args.windows)) if args.windows else []
    unsupported = [w for w in requested_windows if w not in SUPPORTED_WINDOWS]
    if unsupported:
        logging.warning(
            "Ignoring unsupported rolling window sizes %s; only %s are used",
            unsupported,
            SUPPORTED_WINDOWS,
        )
    if set(requested_windows) != set(SUPPORTED_WINDOWS):
        logging.info(
            "Using fixed rolling window sizes %s instead of requested %s",
            SUPPORTED_WINDOWS,
            requested_windows or list(SUPPORTED_WINDOWS),
        )
    windows = list(SUPPORTED_WINDOWS)
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
    integrate_injury_features(team_df, windows, args.player_csv)
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
    excluded = [col for col in EXCLUDED_FINAL_COLUMNS if col in df.columns]
    if excluded:
        df.drop(columns=excluded, inplace=True)
        logging.info("Dropped %d excluded columns", len(excluded))

    r20_columns = [col for col in df.columns if col.endswith("_r20")]
    if r20_columns:
        df.drop(columns=r20_columns, inplace=True)
        logging.info("Dropped %d rolling-20 columns", len(r20_columns))
    logging.info("Final dataset shape: %d rows x %d columns", df.shape[0], df.shape[1])
    df.to_csv(args.out_csv, index=False)
    logging.info(
        "Wrote %d rows x %d columns to %s", df.shape[0], df.shape[1], args.out_csv
    )
    logging.info("==== Dataset build complete ====")


if __name__ == "__main__":
    main()
