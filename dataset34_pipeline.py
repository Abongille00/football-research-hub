"""Dataset 3 (Wyscout) and Dataset 4 (Impect) integration pipeline.

Loader -> Validator -> Normalizer -> Local CSV Generator -> Engine Connector.
No primary-data replacement. Missing source data is reported as unavailable.
"""
from __future__ import annotations

import io, json, os, zipfile, hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

BASE = Path(__file__).resolve().parent
D3_DIR = BASE / "data" / "dataset3_wyscout"
D4_DIR = BASE / "data" / "dataset4_impect"
D3_OUT = BASE / "football_dataset3_wyscout_events.csv"
D3_MATCH_OUT = BASE / "football_dataset3_wyscout_matches.csv"
D3_PLAYER_OUT = BASE / "football_dataset3_wyscout_players.csv"
D4_OUT = BASE / "football_dataset4_impect_events.csv"
D4_KPI_OUT = BASE / "football_dataset4_impect_event_kpis.csv"
D4_PLAYER_OUT = BASE / "football_dataset4_impect_player_kpis.csv"
D4_MATCH_OUT = BASE / "football_dataset4_impect_matches.csv"

WYSCOUT_ZIP_URL = "https://ndownloader.figshare.com/files/14464685"
WYSCOUT_REPO = "https://github.com/koenvo/wyscout-soccer-match-event-dataset"
IMPECT_REPO = "https://github.com/ImpectAPI/open-data"

D3_EVENT_COLUMNS = [
    "source","source_match_id","date","season","competition",
    "home_team","away_team","event_id","period","event_sec",
    "team_id","team","player_id","player","event_type",
    "sub_event_type","outcome","x","y","end_x","end_y","tags","raw_file"
]
D3_MATCH_COLUMNS = ["source","source_match_id","date","season","competition","home_team","away_team","raw_file"]
D3_PLAYER_COLUMNS = ["source","player_id","player","raw_file"]
D4_EVENT_COLUMNS = ["source","source_match_id","date","competition","season","team","player_id","player","event_id","period","timestamp","event_type","action_type","phase","x","y","end_x","end_y","raw_file"]
D4_KPI_COLUMNS = ["source","source_match_id","event_id","kpi_id","kpi_name","value","raw_file"]
D4_PLAYER_COLUMNS = ["source","source_match_id","player_id","player","position","minutes","kpi_id","kpi_name","value","raw_file"]
D4_MATCH_COLUMNS = ["source","source_match_id","date","competition","season","home_team","away_team","raw_file"]


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy(); df.columns=[str(c).strip().lower().replace(" ","_") for c in df.columns]; return df


def _first(d: Dict[str,Any], *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _nested_name(x):
    if isinstance(x, dict): return _first(x,"name","shortName","label","playerName","teamName")
    return x


def _load_json(path: Path):
    with path.open("r",encoding="utf-8") as f: return json.load(f)


def _iter_json_files(root: Path):
    return sorted(root.rglob("*.json")) if root.exists() else []


def download_file(url: str, destination: Path, timeout: int=120):
    import requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    r=requests.get(url, timeout=timeout, stream=True, headers={"User-Agent":"FootballResearchEngine/1.0"})
    r.raise_for_status()
    with destination.open("wb") as f:
        for chunk in r.iter_content(1024*1024):
            if chunk: f.write(chunk)
    return destination


def _safe_json_from_bytes(data: bytes):
    return json.loads(data.decode("utf-8-sig"))


def extract_wyscout_zip(zip_path: Path, out_dir: Path=D3_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.lower().endswith(".json"):
                target=out_dir / Path(n).name
                with z.open(n) as src, target.open("wb") as dst: dst.write(src.read())
    return out_dir


def _wyscout_match_lookup(root: Path):
    lookup={}
    for p in _iter_json_files(root):
        if "match" not in p.name.lower() and "matches" not in p.name.lower(): continue
        try: obj=_load_json(p)
        except Exception: continue
        rows=obj.get("matches",obj) if isinstance(obj,dict) else obj
        if isinstance(rows,dict): rows=[rows]
        if not isinstance(rows,list): continue
        for m in rows:
            if not isinstance(m,dict): continue
            mid=_first(m,"wyId","matchId","id")
            teams=m.get("teams") or []
            home=away=None
            if isinstance(teams,list) and len(teams)>=2:
                home=_nested_name(teams[0].get("team")) if isinstance(teams[0],dict) else None
                away=_nested_name(teams[1].get("team")) if isinstance(teams[1],dict) else None
                # Wyscout often exposes side
                for t in teams:
                    side=_first(t,"side")
                    name=_nested_name(t.get("team")) if isinstance(t,dict) else None
                    if side=="home": home=name
                    if side=="away": away=name
            if mid is not None: lookup[str(mid)]={"date":_first(m,"date","dateutc"),"home_team":home,"away_team":away}
    return lookup


def normalize_wyscout(source_dir: Path=D3_DIR):
    files=_iter_json_files(source_dir)
    match_lookup=_wyscout_match_lookup(source_dir)
    events=[]; players={}; matches={}
    for p in files:
        if "match" in p.name.lower() and "event" not in p.name.lower(): continue
        try: obj=_load_json(p)
        except Exception: continue
        rows=obj.get("events",obj) if isinstance(obj,dict) else obj
        if isinstance(rows,dict): rows=[rows]
        if not isinstance(rows,list): continue
        for e in rows:
            if not isinstance(e,dict): continue
            mid=_first(e,"matchId","match_id")
            if mid is None: continue
            meta=match_lookup.get(str(mid),{})
            team=e.get("team")
            player=e.get("player")
            team_id=_nested_name(team) if isinstance(team,dict) else team
            player_id=player.get("wyId") if isinstance(player,dict) else player
            player_name=_nested_name(player) if isinstance(player,dict) else None
            tags=e.get("tags")
            events.append({
                "source":"Wyscout","source_match_id":mid,"date":meta.get("date"),"season":None,"competition":None,
                "home_team":meta.get("home_team"),"away_team":meta.get("away_team"),"event_id":_first(e,"eventId","id"),
                "period":_first(e,"matchPeriod","period"),"event_sec":_first(e,"eventSec","event_sec"),
                "team_id":team_id,"team":team_id,"player_id":player_id,"player":player_name,
                "event_type":_first(e,"eventName","eventType","type"),"sub_event_type":_first(e,"subEventName","subEventType"),
                "outcome":None,"x":None,"y":None,"end_x":None,"end_y":None,"tags":json.dumps(tags,ensure_ascii=False) if tags is not None else None,"raw_file":p.name
            })
            if player_id is not None: players[str(player_id)]={"source":"Wyscout","player_id":player_id,"player":player_name,"raw_file":p.name}
            matches[str(mid)]={"source":"Wyscout","source_match_id":mid,"date":meta.get("date"),"season":None,"competition":None,"home_team":meta.get("home_team"),"away_team":meta.get("away_team"),"raw_file":p.name}
    return pd.DataFrame(events,columns=D3_EVENT_COLUMNS), pd.DataFrame(matches.values(),columns=D3_MATCH_COLUMNS), pd.DataFrame(players.values(),columns=D3_PLAYER_COLUMNS)


def _find_impect_root(root: Path):
    return root / "data" if (root/"data").exists() else root


def normalize_impect(source_dir: Path=D4_DIR):
    root=_find_impect_root(source_dir); events=[]; kpis=[]; players=[]; matches=[]
    event_files=list((root/"events").rglob("*.json")) if (root/"events").exists() else []
    for p in event_files:
        try: obj=_load_json(p)
        except Exception: continue
        rows=obj.get("events",obj) if isinstance(obj,dict) else obj
        if isinstance(rows,dict): rows=[rows]
        if not isinstance(rows,list): continue
        mid=p.stem.split("_")[-1]
        for e in rows:
            if not isinstance(e,dict): continue
            events.append({"source":"Impect","source_match_id":mid,"date":_first(e,"date"),"competition":None,"season":None,
                           "team":_nested_name(e.get("team")),"player_id":_first(e,"playerId","player_id"),"player":_nested_name(e.get("player")),
                           "event_id":_first(e,"id","eventId"),"period":_first(e,"period","phase"),"timestamp":_first(e,"timestamp","time"),
                           "event_type":_first(e,"eventType","type"),"action_type":_first(e,"actionType"),"phase":_first(e,"phase"),
                           "x":_first(e,"x","startX"),"y":_first(e,"y","startY"),"end_x":_first(e,"endX"),"end_y":_first(e,"endY"),"raw_file":p.name})
    kpi_files=list((root/"events_kpis").rglob("*.json")) if (root/"events_kpis").exists() else []
    for p in kpi_files:
        try: obj=_load_json(p)
        except Exception: continue
        rows=obj.get("events",obj) if isinstance(obj,dict) else obj
        if isinstance(rows,dict): rows=[rows]
        if not isinstance(rows,list): continue
        mid=p.stem.split("_")[-1]
        for e in rows:
            if not isinstance(e,dict): continue
            eid=_first(e,"eventId","event_id","id")
            for key,val in e.items():
                if key in {"eventId","event_id","id"}: continue
                if isinstance(val,(int,float,str)) and not isinstance(val,bool):
                    kpis.append({"source":"Impect","source_match_id":mid,"event_id":eid,"kpi_id":key,"kpi_name":key,"value":val,"raw_file":p.name})
    pkpi_files=list((root/"player_kpis").rglob("*.json")) if (root/"player_kpis").exists() else []
    for p in pkpi_files:
        try: obj=_load_json(p)
        except Exception: continue
        rows=obj.get("players",obj) if isinstance(obj,dict) else obj
        if isinstance(rows,dict): rows=[rows]
        if not isinstance(rows,list): continue
        mid=p.stem.split("_")[-1]
        for r in rows:
            if not isinstance(r,dict): continue
            pid=_first(r,"playerId","player_id","id")
            name=_nested_name(r.get("player")) or _first(r,"playerName","name")
            for key,val in r.items():
                if key in {"playerId","player_id","id","player","playerName","name","position","minutes"}: continue
                if isinstance(val,(int,float,str)) and not isinstance(val,bool):
                    players.append({"source":"Impect","source_match_id":mid,"player_id":pid,"player":name,"position":_first(r,"position"),"minutes":_first(r,"minutes","playDuration"),"kpi_id":key,"kpi_name":key,"value":val,"raw_file":p.name})
    return (pd.DataFrame(events,columns=D4_EVENT_COLUMNS),pd.DataFrame(kpis,columns=D4_KPI_COLUMNS),pd.DataFrame(players,columns=D4_PLAYER_COLUMNS),pd.DataFrame(matches,columns=D4_MATCH_COLUMNS))


def validate(df: pd.DataFrame, required: List[str], name: str):
    checks=[]
    missing=[c for c in required if c not in df.columns]
    checks.append({"Dataset":name,"Check":"Required columns","Status":"PASS" if not missing else "FAIL","Detail":"OK" if not missing else ", ".join(missing)})
    checks.append({"Dataset":name,"Check":"Rows","Status":"PASS" if len(df)>0 else "UNAVAILABLE","Detail":str(len(df))})
    if len(df)>0 and "source" in df.columns:
        checks.append({"Dataset":name,"Check":"Source provenance","Status":"PASS" if df["source"].notna().all() else "FAIL","Detail":"source populated"})
    return pd.DataFrame(checks)


def write_outputs(d3e,d3m,d3p,d4e,d4k,d4p,d4m,base:Path=BASE):
    outputs=[(d3e,D3_OUT),(d3m,D3_MATCH_OUT),(d3p,D3_PLAYER_OUT),(d4e,D4_OUT),(d4k,D4_KPI_OUT),(d4p,D4_PLAYER_OUT),(d4m,D4_MATCH_OUT)]
    for df,path in outputs:
        path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)
    return [str(p) for _,p in outputs]


def build_all(auto_download=False):
    """Build normalized CSVs from local raw source directories; optionally fetch raw Wyscout zip and Impect repo archive."""
    D3_DIR.mkdir(parents=True,exist_ok=True); D4_DIR.mkdir(parents=True,exist_ok=True)
    if auto_download:
        try:
            z=D3_DIR/"wyscout_events.zip"
            if not z.exists(): download_file(WYSCOUT_ZIP_URL,z)
            if not any(D3_DIR.rglob("*.json")): extract_wyscout_zip(z,D3_DIR)
        except Exception as e: d3_error=str(e)
        else: d3_error=None
        try:
            import requests
            tar=D4_DIR/"impect_open_data.zip"
            if not tar.exists(): download_file("https://github.com/ImpectAPI/open-data/archive/refs/heads/main.zip",tar)
            extract_dir=D4_DIR/"open-data-main"
            if not extract_dir.exists():
                with zipfile.ZipFile(tar) as z: z.extractall(D4_DIR)
        except Exception as e: d4_error=str(e)
        else: d4_error=None
    d3e,d3m,d3p=normalize_wyscout(D3_DIR)
    d4e,d4k,d4p,d4m=normalize_impect(D4_DIR)
    paths=write_outputs(d3e,d3m,d3p,d4e,d4k,d4p,d4m)
    checks=pd.concat([validate(d3e,D3_EVENT_COLUMNS,"Dataset 3 Wyscout events"),validate(d4e,D4_EVENT_COLUMNS,"Dataset 4 Impect events")],ignore_index=True)
    return {"paths":paths,"checks":checks,"d3_rows":len(d3e),"d4_rows":len(d4e),"d3_error":locals().get("d3_error"),"d4_error":locals().get("d4_error")}


def load_engine_layers(base:Path=BASE):
    """Engine connector: returns normalized D3/D4 evidence and status, never fabricates missing data."""
    files={"wyscout_events":D3_OUT,"wyscout_matches":D3_MATCH_OUT,"wyscout_players":D3_PLAYER_OUT,
           "impect_events":D4_OUT,"impect_event_kpis":D4_KPI_OUT,"impect_player_kpis":D4_PLAYER_OUT,"impect_matches":D4_MATCH_OUT}
    data={}; status={}
    for key,path in files.items():
        if path.exists():
            try: data[key]=pd.read_csv(path); status[key]="LOADED" if not data[key].empty else "EMPTY"
            except Exception: data[key]=pd.DataFrame(); status[key]="ERROR"
        else: data[key]=pd.DataFrame(); status[key]="MISSING"
    return data,status


def fixture_context(data:Dict[str,pd.DataFrame],home:str,away:str):
    out={}
    for key,df in data.items():
        if df.empty: continue
        cols=set(df.columns)
        if {"home_team","away_team"}.issubset(cols):
            x=df[((df.home_team==home)&(df.away_team==away))|((df.home_team==away)&(df.away_team==home))]
            out[key]=x
        elif "team" in cols:
            out[key]=df[df.team.isin([home,away])]
        else: out[key]=df
    return out
