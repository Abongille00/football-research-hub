"""Phases A-H data governance and maintenance layer.
No fabricated values: missing evidence stays missing and is reported.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re
import pandas as pd

START_DATE=pd.Timestamp('2024-07-01')
END_DATE=pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
SEASONS={'2024/25':('2024-07-01','2025-06-30'),'2025/26':('2025-07-01','2026-06-30'),'2026/27':('2026-07-01','2027-06-30')}
DATASET1_REQUIRED=['date','competition','season','home_team','away_team','home_goals','away_goals','home_shots','away_shots','home_sot','away_sot','home_corners','away_corners']
DATASET2_ID=['date','home_team','away_team','player','team']
PHASES={
 'A':'Source registry, date/season boundary, competition/team/fixture discovery, URL resolution and canonical identity',
 'B':'Dataset 1 match-level acquisition, normalization, provenance and reconciliation',
 'C':'Dataset 2 player identity, lineups, minutes, production and player provenance',
 'D':'StatsBomb/Understat/Wyscout/Impect event and tactical evidence connectors',
 'E':'Historical windows, home/away, competition, opponent and player-form features',
 'F':'Completeness, duplicate, range, consistency, disagreement and audit validation',
 'G':'Automatic refresh, corrected fixtures, versioning, stale-data and acquisition-failure tracking',
 'H':'Evidence-to-engine integration with timestamps, missing-data handling and provenance',
}

def now(): return datetime.now(timezone.utc).isoformat()
def season_for_date(v):
 d=pd.to_datetime(v,errors='coerce')
 if pd.isna(d): return ''
 y=d.year if d.month>=7 else d.year-1
 return f'{y}/{str(y+1)[-2:]}'
def in_scope(v):
 d=pd.to_datetime(v,errors='coerce'); return bool(pd.notna(d) and START_DATE<=d<END_DATE)
def registry():
 return pd.DataFrame([{'source':'Flashscore','class':'public match source','fields':'match stats, event/stat pages where exposed','provenance':'URL + retrieval timestamp'}, {'source':'LiveScore','class':'public match source','fields':'match stats where exposed','provenance':'URL + retrieval timestamp'}, {'source':'StatsBomb','class':'event data','fields':'events/xG where legitimately available','provenance':'dataset/file/API metadata'}, {'source':'Understat','class':'xG/event data','fields':'xG/xGA/shots where legitimately available','provenance':'page/data metadata'}, {'source':'Wyscout','class':'event data','fields':'event/tactical fields where licensed/available','provenance':'source metadata'}, {'source':'Impect','class':'tactical data','fields':'tactical metrics where licensed/available','provenance':'source metadata'}])

def validate_dataset1(df):
 issues=[]
 if df is None: return {'rows':0,'missing_required':DATASET1_REQUIRED,'duplicate_keys':0,'out_of_scope':0,'invalid_relationships':0,'issues':['dataset missing']}
 d=df.copy(); missing=[c for c in DATASET1_REQUIRED if c not in d.columns]
 if missing: issues.append('missing required columns')
 for c in [x for x in DATASET1_REQUIRED if x not in ('date','competition','season','home_team','away_team') and x in d]: d[c]=pd.to_numeric(d[c],errors='coerce')
 key=[c for c in ['date','competition','home_team','away_team'] if c in d]
 dup=int(d.duplicated(key,keep=False).sum()) if len(key)==4 else 0
 out=int((~pd.to_datetime(d['date'],errors='coerce').map(in_scope)).sum()) if 'date' in d else len(d)
 rel=0
 if {'home_sot','home_shots'}<=set(d): rel+=int((d.home_sot>d.home_shots).fillna(False).sum())
 if {'away_sot','away_shots'}<=set(d): rel+=int((d.away_sot>d.away_shots).fillna(False).sum())
 if dup:issues.append('duplicate fixture keys')
 if out:issues.append('out-of-scope dates')
 if rel:issues.append('SOT exceeds shots')
 return {'rows':len(d),'missing_required':missing,'duplicate_keys':dup,'out_of_scope':out,'invalid_relationships':rel,'issues':issues}

def validate_dataset2(df):
 if df is None:return {'rows':0,'missing_identity':DATASET2_ID,'duplicates':0,'out_of_scope':0,'issues':['dataset missing']}
 d=df.copy(); missing=[c for c in DATASET2_ID if c not in d]
 key=[c for c in DATASET2_ID if c in d]
 dup=int(d.duplicated(key,keep=False).sum()) if len(key)==len(DATASET2_ID) else 0
 out=int((~pd.to_datetime(d['date'],errors='coerce').map(in_scope)).sum()) if 'date' in d else 0
 issues=[]
 if missing:issues.append('missing player identity columns')
 if dup:issues.append('duplicate player-fixture records')
 if out:issues.append('out-of-scope dates')
 return {'rows':len(d),'missing_identity':missing,'duplicates':dup,'out_of_scope':out,'issues':issues}

def build_historical_features(df):
 if df is None or df.empty:return pd.DataFrame()
 d=df.copy();d['date']=pd.to_datetime(d.get('date'),errors='coerce');d=d[d.date.map(in_scope)].copy()
 for c in ['home_goals','away_goals','home_shots','away_shots','home_sot','away_sot','home_corners','away_corners']:
  if c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.sort_values('date')

def source_coverage(df):
 if df is None or df.empty:return pd.DataFrame(columns=['source','rows'])
 if 'sources' in df:
  vals=[]
  for x in df.sources.fillna(''):
   vals.extend([v.strip() for v in str(x).split(',') if v.strip()])
  return pd.Series(vals).value_counts().rename_axis('source').reset_index(name='rows')
 return pd.DataFrame()

def phase_health(dataset1=None,dataset2=None,source_status=None):
 v1=validate_dataset1(dataset1);v2=validate_dataset2(dataset2)
 return {'generated_at_utc':now(),'scope':f'2024-07-01 through {END_DATE - pd.Timedelta(days=1):%Y-%m-%d}','phases':PHASES,'dataset1':v1,'dataset2':v2,'source_status_rows':0 if source_status is None else len(source_status),'coverage':source_coverage(dataset1).to_dict('records')}


def scoped_seasons():
    """Return the requested seasons, with 2026/27 truncated at today's date."""
    today=pd.Timestamp.now().normalize()
    out={}
    for season,(a,b) in SEASONS.items():
        start=pd.Timestamp(a); end=min(pd.Timestamp(b),today)
        if start<=end: out[season]=(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'))
    return out

def assign_scope_columns(df):
    if df is None or df.empty:return df
    x=df.copy()
    x['date']=pd.to_datetime(x.get('date'),errors='coerce')
    x['season']=x['date'].map(season_for_date)
    x['in_scope']=x['date'].map(in_scope)
    return x

def field_inventory(df, source_col='source'):
    """Measure populated fields without counting placeholder/blank values."""
    if df is None or df.empty:return pd.DataFrame(columns=['source','field','rows','populated','missing_pct'])
    x=df.copy(); sources=x[source_col].fillna('Unknown').astype(str) if source_col in x else pd.Series(['Canonical']*len(x),index=x.index)
    rows=[]
    ignore={'source','sources','source_url','source_urls','source_match_id','source_match_ids','source_fetched_at','canonical_fixture_id','match_id'}
    for src in sorted(sources.unique()):
        z=x.loc[sources==src]
        for c in x.columns:
            if c in ignore:continue
            populated=int(z[c].notna().sum()) if c in z else 0
            if c in z and z[c].dtype==object: populated=int(z[c].astype(str).str.strip().replace({'':'nan','None':'nan'}).ne('nan').sum())
            rows.append({'source':src,'field':c,'rows':len(z),'populated':populated,'missing_pct':round(100*(1-populated/max(len(z),1)),2)})
    return pd.DataFrame(rows)

def source_summary(records):
    """Compact source-by-source acquisition measurement."""
    if not records:return pd.DataFrame(columns=['source','discovered','acquired','normalized','reachable_unparsed','errors','fields_found'])
    x=pd.DataFrame(records)
    for c in ['source','status','fields_found']:
        if c not in x:x[c]=''
    rows=[]
    for src,g in x.groupby('source',dropna=False):
        rows.append({'source':src,'discovered':len(g),'acquired':int(g.status.isin(['FETCHED','CACHED','NORMALIZED','REACHABLE_UNPARSED']).sum()),'normalized':int(g.status.eq('NORMALIZED').sum()),'reachable_unparsed':int(g.status.eq('REACHABLE_UNPARSED').sum()),'errors':int(g.status.isin(['ERROR','HTTP_ERROR']).sum()),'fields_found':int(pd.to_numeric(g.fields_found,errors='coerce').fillna(0).sum())})
    return pd.DataFrame(rows)

def data_inventory(dataset1=None,dataset2=None,acquisition_records=None):
    return {'generated_at_utc':now(),'scope':scoped_seasons(),'dataset1':validate_dataset1(dataset1),'dataset2':validate_dataset2(dataset2),'source_summary':source_summary(acquisition_records or []).to_dict('records'),'dataset1_fields':field_inventory(dataset1).to_dict('records') if dataset1 is not None else []}
