"""Module 10: Route haul analysis."""
import re,logging,numpy as np,pandas as pd,matplotlib.pyplot as plt
from config import OUTPUTS_DIR,SENT_COLOR
from pipeline.utils import banner,save_fig
log=logging.getLogger(__name__)
REGION_MAP={k:v for v,keys in {"Europe":["London","Gatwick","Heathrow","Stansted","Manchester","Birmingham","Edinburgh","Dublin","Paris","Lyon","Nice","Frankfurt","Munich","Berlin","Hamburg","Amsterdam","Madrid","Barcelona","Lisbon","Rome","Milan","Venice","Zurich","Geneva","Vienna","Brussels","Copenhagen","Stockholm","Oslo","Helsinki","Athens","Thessaloniki","Heraklion","Prague","Budapest","Warsaw","Bucharest","Sofia","Belgrade","Zagreb","Tallinn","Riga","Vilnius","Moscow","Istanbul","Malta","Larnaca","Reykjavik","Gothenburg","Bergen","Marseille","Seville","Valencia","Porto","Naples","Turin"],"N.America":["New York","Newark","JFK","Los Angeles","San Francisco","Seattle","Chicago","Boston","Miami","Orlando","Atlanta","Dallas","Houston","Denver","Las Vegas","Toronto","Vancouver","Montreal","Calgary","Mexico City","Cancun","Minneapolis","Detroit","Phoenix"],"S.America":["Lima","Bogota","Buenos Aires","Santiago","Sao Paulo","Rio","Caracas","Quito","Montevideo","La Paz","Medellin"],"Asia":["Bangkok","Singapore","Kuala Lumpur","Jakarta","Manila","Tokyo","Osaka","Seoul","Beijing","Shanghai","Hong Kong","Taipei","Delhi","Mumbai","Bangalore","Chennai","Hyderabad","Kolkata","Colombo","Dhaka","Karachi","Denpasar","Bali","Ho Chi Minh","Hanoi","Phnom Penh","Yangon","Male","Chengdu","Guangzhou","Shenzhen","Kathmandu"],"Middle East":["Dubai","Abu Dhabi","Sharjah","Doha","Kuwait","Bahrain","Riyadh","Jeddah","Muscat","Amman","Beirut","Tel Aviv","Cairo","Tehran"],"Africa":["Johannesburg","Cape Town","Nairobi","Lagos","Accra","Addis Ababa","Dar es Salaam","Casablanca","Tunis","Algiers","Kampala","Kigali","Lusaka","Harare","Durban"],"Oceania":["Sydney","Melbourne","Brisbane","Perth","Adelaide","Auckland","Wellington","Christchurch","Queenstown","Cairns","Gold Coast","Darwin"],"Caribbean":["Havana","Kingston","San Juan","Nassau","Bridgetown"],"C.America":["Guatemala","Panama","San Jose","Managua","Belize"]}.items() for k in keys}
HAUL_MAP={("Europe","Europe"):"Short-haul",("Europe","N.America"):"Long-haul",("Europe","S.America"):"Ultra-long",("Europe","Asia"):"Long-haul",("Europe","Middle East"):"Medium-haul",("Europe","Africa"):"Medium-haul",("Europe","Oceania"):"Ultra-long",("Europe","Caribbean"):"Long-haul",("N.America","N.America"):"Short-haul",("N.America","S.America"):"Medium-haul",("N.America","Asia"):"Ultra-long",("N.America","Oceania"):"Ultra-long",("Asia","Asia"):"Short-haul",("Asia","Oceania"):"Medium-haul",("Asia","Middle East"):"Medium-haul",("Middle East","Middle East"):"Short-haul",("Middle East","Africa"):"Medium-haul",("Africa","Africa"):"Short-haul",("Oceania","Oceania"):"Short-haul",("S.America","S.America"):"Short-haul"}
def _city_region(cs):
    cs=cs.strip().title()
    if cs in REGION_MAP: return REGION_MAP[cs]
    for k,v in REGION_MAP.items():
        if k.lower() in cs.lower() or cs.lower() in k.lower(): return v
    return "Unknown"
def _parse_route(r):
    if pd.isna(r): return None,None,False,None
    r=str(r).strip(); has_conn=bool(re.search(r"\bvia\b",r,re.I))
    r_clean=re.sub(r"\bvia\b.+$","",r,flags=re.I).strip()
    parts=[p.strip() for p in re.split(r"\s+to\s+",r_clean,flags=re.I) if p.strip()]
    if len(parts)<2: return None,None,has_conn,None
    o_r,d_r=_city_region(parts[0]),_city_region(parts[-1])
    haul=HAUL_MAP.get((o_r,d_r)) or HAUL_MAP.get((d_r,o_r)) or "Unknown"
    return parts[0],parts[-1],has_conn,haul
def run(df):
    banner("MODULE 10 — ROUTE ANALYSIS"); df=df.copy()
    parsed=df["Route"].apply(_parse_route)
    df["route_origin"],df["route_dest"],df["has_connection"],df["haul_type"]=zip(*parsed)
    df["has_connection"]=df["has_connection"].astype(bool)
    haul_order=["Short-haul","Medium-haul","Long-haul","Ultra-long"]
    haul_colors={"Short-haul":"#2196F3","Medium-haul":"#FF9800","Long-haul":"#F44336","Ultra-long":"#9C27B0"}
    df_haul=df[df["haul_type"].isin(haul_order)].copy()
    haul_stats=df_haul.groupby("haul_type").agg(n=("Overall_Rating","count"),mean_rating=("Overall_Rating","mean")).reindex(haul_order).dropna()
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    bars=axes[0].bar(haul_stats.index,haul_stats["mean_rating"],color=[haul_colors[h] for h in haul_stats.index],edgecolor="k")
    axes[0].set_ylim(0,8); axes[0].set_title("Mean Rating by Haul Type",fontweight="bold")
    for b,v,n in zip(bars,haul_stats["mean_rating"],haul_stats["n"]):
        axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f"{v:.2f}\n(n={n:,})",ha="center",fontsize=9,fontweight="bold")
    bot=np.zeros(len(haul_stats))
    for cat in ["positive","neutral","negative"]:
        vals=df_haul.groupby("haul_type")["sentiment"].apply(lambda x,c=cat:(x==c).mean()*100).reindex(haul_order).fillna(0)
        axes[1].bar(haul_order,vals,bottom=bot,label=cat,color=SENT_COLOR[cat],edgecolor="white"); bot+=vals.values
    axes[1].set_title("Sentiment by Haul Type",fontweight="bold"); axes[1].legend(fontsize=9)
    save_fig(f"{OUTPUTS_DIR}/route_haul_analysis.png")
    return df
