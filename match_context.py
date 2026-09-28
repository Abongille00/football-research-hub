"""Context-only public-source layer for Sports Mole and Forebet.
Prediction/forecast/betting content is hard-excluded before evidence reaches the engine.
"""
from __future__ import annotations
import re, html
from datetime import datetime, timezone
from urllib.parse import quote_plus
from pathlib import Path
import pandas as pd
try:
    import requests
except Exception:
    requests=None

SOURCES={"Sports Mole":"sportsmole.co.uk","Forebet":"forebet.com"}
PREDICTION_PATTERNS=[
 r'\bprediction\b',r'\bpredictions\b',r'\bpredicted\b',r'\bpredicts\b',r'\bforecast\b',r'\bforecasts\b',
 r'\btip\b',r'\btips\b',r'\bbetting tip\b',r'\bbest bet\b',r'\bwin probability\b',r'\bprobability\b',
 r'\bcorrect score\b',r'\bto win\b',r'\bwill win\b',r'\bexpected score\b'
]
CONTEXT_HEADINGS=re.compile(r'(?im)^(?:team news|team news and predicted lineups|injuries|injury|suspensions?|lineups?|starting lineups?|formation|form|head to head|h2h|manager comments?|team news|weather|match preview)\s*:?\s*$')

def now(): return datetime.now(timezone.utc).isoformat()
def _strip(text):
    text=re.sub(r'(?is)<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>',' ',text)
    text=re.sub(r'<[^>]+>','\n',text); return re.sub(r'\s+',' ',html.unescape(text)).strip()
def _contains_prediction(s): return any(re.search(p,s,re.I) for p in PREDICTION_PATTERNS)
def _context_sentences(text):
    chunks=re.split(r'(?<=[.!?])\s+',text)
    return [c.strip() for c in chunks if len(c.strip())>=30 and not _contains_prediction(c)]
def _search(query,domain,limit=3):
    if requests is None:return []
    try:
        u='https://html.duckduckgo.com/html/?q='+quote_plus(f'site:{domain} {query}')
        r=requests.get(u,timeout=15,headers={'User-Agent':'Mozilla/5.0 (compatible; FootballResearchEngine/1.0)'})
        if not r.ok:return []
        urls=[]
        for u in re.findall(r'href="(https?://[^" ]+)"',r.text):
            if domain in u and u not in urls:urls.append(u)
            if len(urls)>=limit:break
        return urls
    except Exception:return []
def acquire_context(home,away,match_date='',competition='',cache_dir='source_cache/context',limit=3):
    out=[]; Path(cache_dir).mkdir(parents=True,exist_ok=True)
    q=f'"{home}" "{away}"'; q += f' "{match_date}"' if match_date else ''; q += f' "{competition}"' if competition else ''
    for source,domain in SOURCES.items():
        urls=_search(q,domain,limit)
        for url in urls:
            try:
                r=requests.get(url,timeout=15,headers={'User-Agent':'Mozilla/5.0 (compatible; FootballResearchEngine/1.0)'})
                raw=_strip(r.text or '') if r.ok else ''
                sentences=_context_sentences(raw)
                key=re.sub(r'[^A-Za-z0-9]+','_',source+'_'+url)[-100:]
                Path(cache_dir,key+'.txt').write_text('\n'.join(sentences),encoding='utf-8')
                out.append({'source':source,'url':url,'retrieved_utc':now(),'status':'ACQUIRED','verification_status':'UNVERIFIED','prediction_content_excluded':True,'context_items':len(sentences),'context_text':' '.join(sentences[:80])})
            except Exception as e:
                out.append({'source':source,'url':url,'retrieved_utc':now(),'status':'ERROR','verification_status':'UNVERIFIED','prediction_content_excluded':True,'context_items':0,'error':str(e)})
    return pd.DataFrame(out)
