"""Dataset 3/4 event -> feature -> fixture -> tactical/game-state evidence layer.

This module is deliberately descriptive: it extracts observed event evidence, reconstructs
basic game state where the source contains goal events, and returns evidence-strength and
availability metadata. It never invents missing events or converts historical rates into
certainty.
"""
from __future__ import annotations
import math
from typing import Dict, Any
import numpy as np
import pandas as pd


def _clean_name(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return ""
    s=str(x).strip().lower()
    for token in [" football club"," fc"," afc"," cf"]: s=s.replace(token,"")
    return " ".join(s.split())


def _match_team(name, teams):
    key=_clean_name(name)
    lookup={_clean_name(t):t for t in teams if str(t).strip()}
    if key in lookup: return lookup[key]
    return None


def _contains(series, words):
    if series is None: return pd.Series(dtype=bool)
    s=series.fillna("").astype(str).str.lower()
    return s.apply(lambda x:any(w in x for w in words))


def _num(s): return pd.to_numeric(s,errors="coerce")


def reconstruct_game_state(events: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Add running score and score-state labels when goal events can be identified."""
    if events is None or events.empty: return pd.DataFrame()
    x=events.copy()
    x["_event_order"]=_num(x.get("event_sec", pd.Series(index=x.index,dtype=float)))
    if x["_event_order"].isna().all() and "timestamp" in x.columns:
        x["_event_order"]=_num(x["timestamp"])
    x=x.sort_values([c for c in ["period","_event_order","event_id"] if c in x.columns],kind="stable")
    et=(x.get("event_type",pd.Series("",index=x.index)).fillna("").astype(str).str.lower()+" "+x.get("sub_event_type",pd.Series("",index=x.index)).fillna("").astype(str).str.lower())
    goal=et.str.contains("goal",regex=False) & ~et.str.contains("attempt|keeper|save",regex=False)
    hg=ag=0; states=[]
    for idx,row in x.iterrows():
        team=_clean_name(row.get("team"));
        states.append((hg,ag,"Home leading" if hg>ag else "Away leading" if ag>hg else "Level"))
        if goal.loc[idx]:
            if team==_clean_name(home): hg+=1
            elif team==_clean_name(away): ag+=1
    x[["home_score_before","away_score_before","score_state"]]=states
    return x.drop(columns=["_event_order"],errors="ignore")


def _team_events(events, team):
    if events is None or events.empty or "team" not in events.columns: return pd.DataFrame()
    key=_clean_name(team)
    return events[events["team"].map(_clean_name).eq(key)].copy()


def event_feature_vector(events: pd.DataFrame, team: str) -> Dict[str,Any]:
    """Extract normalized event features from a team's observed events."""
    e=_team_events(events,team)
    if e.empty: return {"available":False,"team":team,"events":0}
    et=(e.get("event_type",pd.Series("",index=e.index)).fillna("").astype(str).str.lower()+" "+e.get("sub_event_type",pd.Series("",index=e.index)).fillna("").astype(str).str.lower())
    n=len(e); f={"available":True,"team":team,"events":n}
    categories={
        "passes": ["pass"], "shots": ["shot"], "goals": ["goal"],
        "duels": ["duel"], "recoveries": ["recovery"], "interceptions":["interception"],
        "fouls": ["foul"], "crosses":["cross"], "corners":["corner"],
        "free_kicks":["free kick","freekick"], "offsides":["offside"],
        "clearances":["clearance"], "tackles":["tackle"],
    }
    for k,words in categories.items(): f[k]=int(_contains(et,words).sum())
    f["shots_per_100_events"]=100*f["shots"]/n
    f["goal_rate_per_shot"]=f["goals"]/f["shots"] if f["shots"] else np.nan
    if "period" in e.columns:
        first=_num(e["period"]).eq(1)
        # Some sources use strings such as 1H/2H.
        p=e["period"].fillna("").astype(str).str.lower()
        first=first|p.isin(["1h","first half","firsthalf"])
        f["first_half_events"]=int(first.sum()); f["second_half_events"]=int(n-first.sum())
    else: f["first_half_events"]=np.nan; f["second_half_events"]=np.nan
    # Coordinate-derived territorial features, only when coordinates actually exist.
    x=_num(e.get("x",pd.Series(index=e.index,dtype=float)))
    f["coordinate_coverage"]=float(x.notna().mean())
    shotmask=_contains(et,["shot"])
    if shotmask.any() and x.notna().any():
        sx=x[shotmask].dropna(); f["shots_final_third_proxy_rate"]=float((sx>=66).mean()) if len(sx) else np.nan
    else: f["shots_final_third_proxy_rate"]=np.nan
    return f


def game_state_features(events: pd.DataFrame, team: str) -> Dict[str,Any]:
    if events is None or events.empty: return {"available":False,"team":team}
    x=reconstruct_game_state(events, events.get("home_team",pd.Series([None])).iloc[0] if "home_team" in events.columns else "", events.get("away_team",pd.Series([None])).iloc[0] if "away_team" in events.columns else "")
    if x.empty or "score_state" not in x.columns: return {"available":False,"team":team}
    t=_clean_name(team); et=x.get("event_type",pd.Series("",index=x.index)).fillna("").astype(str).str.lower()
    own=x.get("team",pd.Series("",index=x.index)).map(_clean_name).eq(t)
    goal=et.str.contains("goal",regex=False)
    # Event shares while team is level / leading / trailing.
    out={"available":True,"team":team,"events":len(x)}
    for state in ["Level","Home leading","Away leading"]:
        mask=x["score_state"].eq(state)
        out[f"events_{state.lower().replace(' ','_')}"]=int((mask&own).sum())
        out[f"shots_{state.lower().replace(' ','_')}"]=int((mask&own&et.str.contains("shot",regex=False)).sum())
        out[f"goals_{state.lower().replace(' ','_')}"]=int((mask&own&goal).sum())
    return out


def tactical_profile(events: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    rows=[]
    for team,side in [(home,"Home"),(away,"Away")]:
        f=event_feature_vector(events,team)
        if not f.get("available"): rows.append({"Team":team,"Side":side,"Status":"UNAVAILABLE"}); continue
        rows.append({"Team":team,"Side":side,"Status":"OBSERVED","Events":f["events"],"Passes":f["passes"],"Shots":f["shots"],"Goals":f["goals"],"Duels":f["duels"],"Recoveries":f["recoveries"],"Interceptions":f["interceptions"],"Fouls":f["fouls"],"Crosses":f["crosses"],"Corners":f["corners"],"Tackles":f["tackles"],"Shots/100 Events":f["shots_per_100_events"],"Goal/Shot":f["goal_rate_per_shot"],"Coordinate Coverage":f["coordinate_coverage"]})
    return pd.DataFrame(rows)


def fixture_match_layer(data: Dict[str,pd.DataFrame], home: str, away: str) -> Dict[str,Any]:
    """Match exact historical fixture first; otherwise create team-event context."""
    out={"exact_matches":pd.DataFrame(),"wyscout_events":pd.DataFrame(),"impect_events":pd.DataFrame(),"match_status":"UNAVAILABLE"}
    for key in ["wyscout_matches","impect_matches"]:
        df=data.get(key,pd.DataFrame())
        if df.empty or not {"home_team","away_team"}.issubset(df.columns): continue
        x=df[((df.home_team.map(_clean_name)==_clean_name(home))&(df.away_team.map(_clean_name)==_clean_name(away)))|((df.home_team.map(_clean_name)==_clean_name(away))&(df.away_team.map(_clean_name)==_clean_name(home)))]
        if not x.empty: out["exact_matches"]=pd.concat([out["exact_matches"],x],ignore_index=True)
    for key in ["wyscout_events","impect_events"]:
        df=data.get(key,pd.DataFrame())
        if df.empty: continue
        if {"home_team","away_team"}.issubset(df.columns):
            x=df[((df.home_team.map(_clean_name)==_clean_name(home))&(df.away_team.map(_clean_name)==_clean_name(away)))|((df.home_team.map(_clean_name)==_clean_name(away))&(df.away_team.map(_clean_name)==_clean_name(home)))]
        elif "team" in df.columns:
            x=df[df.team.map(_clean_name).isin([_clean_name(home),_clean_name(away)])]
        else: x=pd.DataFrame()
        out[key]=x
    if not out["exact_matches"].empty: out["match_status"]="EXACT_HISTORICAL_MEETING"
    elif not out["wyscout_events"].empty or not out["impect_events"].empty: out["match_status"]="TEAM_EVENT_CONTEXT"
    return out


def tactical_game_state_evidence(data: Dict[str,pd.DataFrame], home: str, away: str) -> Dict[str,Any]:
    layer=fixture_match_layer(data,home,away)
    frames=[x for x in [layer.get("wyscout_events"),layer.get("impect_events")] if x is not None and not x.empty]
    events=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    # Avoid duplicate source rows being treated as independent if both layers point to same source match.
    profiles=tactical_profile(events,home,away) if not events.empty else pd.DataFrame()
    states={"home":game_state_features(events,home),"away":game_state_features(events,away)} if not events.empty else {}
    availability={
        "Wyscout events":int(len(layer.get("wyscout_events",pd.DataFrame()))),
        "Impect events":int(len(layer.get("impect_events",pd.DataFrame()))),
        "Exact historical meetings":int(len(layer.get("exact_matches",pd.DataFrame()))),
    }
    return {"layer":layer,"events":events,"profiles":profiles,"game_states":states,"availability":availability}


def market_event_relevance(events: pd.DataFrame, team: str, market: str, direction: str, line: float) -> Dict[str,Any]:
    """Convert event features into market-relevant supporting/contradicting evidence."""
    f=event_feature_vector(events,team)
    if not f.get("available"): return {"status":"UNAVAILABLE","support":None,"reason":"No normalized events for target team."}
    market_key={"Shots":"shots","Shots on Target":"shots","Corners":"corners","Goals":"goals"}.get(market)
    if market_key is None: return {"status":"UNAVAILABLE","support":None,"reason":"Market not mapped to event layer."}
    observed=f.get(market_key)
    # Event layer does not contain SOT reliably in the current normalized schema; do not fake it.
    if market=="Shots on Target": return {"status":"LIMITED","support":None,"reason":"Current normalized D3/D4 schema does not guarantee an SOT outcome field."}
    if observed is None: return {"status":"LIMITED","support":None,"reason":"Feature unavailable."}
    # This is a contextual signal, not a probability: compare event intensity to the requested line.
    support=(observed>line) if direction=="Over" else (observed<line)
    return {"status":"OBSERVED","support":bool(support),"observed_event_count":observed,"reason":"Event-layer count is contextual and not a direct match-level forecast."}


def integrate_candidates(candidates: pd.DataFrame, evidence: Dict[str,Any], home: str, away: str) -> pd.DataFrame:
    """Attach D3/D4 evidence to candidate rows and apply a conservative evidence gate.

    Event evidence is additive only. It can downgrade/hold a candidate but cannot manufacture
    a candidate that did not survive the core historical/market engine.
    """
    if candidates is None or candidates.empty: return pd.DataFrame()
    x=candidates.copy()
    profiles=evidence.get("profiles",pd.DataFrame())
    rows=[]
    for _,r in x.iterrows():
        # Existing candidate represents the fixture market; evaluate both team contexts.
        team=r.get("Team") or r.get("Target Team") or ""
        targets=[team] if team else [home,away]
        rel=[]
        for t in targets:
            z=market_event_relevance(evidence.get("events",pd.DataFrame()),t,r.get("Market"),r.get("Direction"),float(r.get("Line",0)))
            if z.get("status")=="OBSERVED": rel.append(z.get("support"))
        observed_support=float(np.mean(rel)) if rel else np.nan
        r["Event Layer Status"]="OBSERVED" if rel else "UNAVAILABLE/LIMITED"
        r["Event Context Support"]=observed_support
        # Do not change the original research score when event evidence is unavailable.
        base=float(r.get("Research Support",0) or 0)
        if np.isfinite(observed_support):
            # Small bounded adjustment: event evidence is corroborative, not dominant.
            r["Integrated Research Support"]=round(base + 5*(observed_support-0.5),1)
        else: r["Integrated Research Support"]=round(base,1)
        r["D3/D4 Evidence Role"]="Corroborative only"
        rows.append(r)
    out=pd.DataFrame(rows)
    if out.empty:return out
    sortcol="Integrated Research Support"
    return out.sort_values(sortcol,ascending=False).reset_index(drop=True)
