"""
scraper/skytrax_scraper.py — SkyTrax review downloader using real Chrome browser.
Incremental: only downloads reviews newer than the newest date in existing_csv.
"""
import logging, random, re, time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup
from config import SCRAPER_BASE_URL, SCRAPER_AZ_PAGE, SCRAPER_PAGE_SIZE, SCRAPER_DELAY, SCRAPE_COLS

log = logging.getLogger(__name__)
_DATE_FMTS = ["%d %B %Y", "%B %Y"]

def _parse_review_date(raw):
    clean = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw).strip()
    for fmt in _DATE_FMTS:
        try: return datetime.strptime(clean, fmt)
        except ValueError: pass
    return None

def _make_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--lang=en-US")
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
        {"source":"Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"})
    return driver

def _fetch(driver, url, delay=SCRAPER_DELAY, retries=3):
    from selenium.common.exceptions import WebDriverException
    for attempt in range(1, retries+1):
        try:
            time.sleep(random.uniform(*delay))
            driver.get(url)
            time.sleep(random.uniform(1.5, 3.0))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.3);")
            time.sleep(random.uniform(0.5, 1.2))
            html = driver.page_source
            if html: return BeautifulSoup(html, "lxml")
        except WebDriverException as e:
            log.warning(f"WebDriver error ({attempt}/{retries}): {e}")
            time.sleep(5*attempt)
    return None

def get_airline_list(driver):
    soup = _fetch(driver, SCRAPER_AZ_PAGE, delay=(2.0,4.0))
    if not soup: raise ConnectionError("Cannot fetch airline list.")
    airlines, seen = [], set()
    for a in soup.find_all("a", href=re.compile(r"^/airline-reviews/[^/]+/?$")):
        slug = a["href"].rstrip("/").split("/")[-1]
        name = a.get_text(strip=True)
        if name and slug not in seen:
            seen.add(slug)
            airlines.append({"name":name,"slug":slug,"url":f"{SCRAPER_BASE_URL}{a['href']}"})
    log.info(f"Found {len(airlines)} airlines on A-Z page.")
    return airlines

def _count_filled_stars(cell):
    for tag in cell.find_all(True):
        for attr in ("data-rating","data-score"):
            v = tag.get(attr)
            if v:
                try: return str(int(float(v)))
                except: pass
    for tag in cell.find_all(True):
        m = re.search(r"(\d+)\s+out\s+of", tag.get("aria-label",""))
        if m: return m.group(1)
    filled = cell.find_all("span", class_=re.compile(r"fill|filled|active|on", re.I))
    if filled: return str(min(len(filled),5))
    all_stars = cell.find_all("span", class_=re.compile(r"star", re.I))
    filled2 = [s for s in all_stars if re.search(r"fill|active|on|solid"," ".join(s.get("class",[])),re.I)]
    if filled2: return str(min(len(filled2),5))
    m2 = re.search(r"^\s*(\d)\s*$", cell.get_text(strip=True))
    if m2: return m2.group(1)
    return None

def _parse_block(block, airline_name):
    r = {c:None for c in SCRAPE_COLS}
    r["Airline Name"] = airline_name
    h = block.find("h2") or block.find("h3")
    if h: r["Review_Title"] = f'"{h.get_text(strip=True)}"'
    rt = (block.find("div", class_=re.compile(r"rating-10|overall",re.I)) or
          block.find("span", itemprop="ratingValue"))
    if rt:
        rv = rt.find("span", itemprop="ratingValue")
        m  = re.search(r"\b(\d{1,2})\b", (rv or rt).get_text(strip=True))
        if m: r["Overall_Rating"] = m.group(1)
    meta = (block.find(class_=re.compile(r"userStatus|reviewer|author",re.I)) or
            block.find("h3", class_=re.compile(r"sub.*header",re.I)))
    if meta:
        raw = meta.get_text(" ",strip=True)
        dm  = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{4})",raw)
        if dm: r["Review Date"] = dm.group(1)
    cd = (block.find("div",class_="text_content") or block.find("div",itemprop="reviewBody"))
    if cd:
        txt = cd.get_text()
        r["Verified"] = "Not Verified" not in txt
        r["Review"]   = re.sub(r"\s*(?:✅\s*Verified Review\s*\|?|Not Verified\s*\|?)","",txt,flags=re.I).strip()
    table = (block.find("table",class_=re.compile(r"review-ratings",re.I)) or
             block.find("table",class_=re.compile(r"rating",re.I)) or block.find("table"))
    if table:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells)<2: continue
            label = cells[0].get_text(strip=True); matched=False
            for key in ["Seat Comfort","Cabin Staff Service","Food & Beverages",
                        "Ground Service","Inflight Entertainment","Wifi & Connectivity","Value For Money"]:
                if key.lower() in label.lower():
                    sv = _count_filled_stars(cells[1])
                    if sv: r[key]=sv
                    matched=True; break
            if not matched:
                if "Aircraft" in label: r["Aircraft"]=cells[1].get_text(strip=True) or None
                elif re.search(r"traveller|type of traveller",label,re.I): r["Type Of Traveller"]=cells[1].get_text(strip=True) or None
                elif re.search(r"seat type|cabin flown",label,re.I): r["Seat Type"]=cells[1].get_text(strip=True) or None
                elif "Route" in label: r["Route"]=cells[1].get_text(strip=True) or None
                elif re.search(r"date flown|flown",label,re.I): r["Date Flown"]=cells[1].get_text(strip=True) or None
                elif re.search(r"recommended",label,re.I):
                    rec=cells[1].get_text(strip=True).lower()
                    r["Recommended"]="yes" if "yes" in rec else "no" if "no" in rec else None
    return r

def _scrape_airline(driver, airline, max_pages, delay, cutoff_date):
    name,slug = airline["name"],airline["slug"]
    base = f"{SCRAPER_BASE_URL}/airline-reviews/{slug}"
    reviews=[]; page=1; reached_cutoff=False
    while page<=max_pages:
        url  = f"{base}/page/{page}/?sortby=post_date%3ADesc&pagesize={SCRAPER_PAGE_SIZE}"
        soup = _fetch(driver,url,delay)
        if not soup: break
        blocks = (soup.find_all("article",itemprop="review") or soup.find_all("div",itemprop="review"))
        if not blocks: log.info(f"  [{name}] page {page}: no blocks — done."); break
        page_reviews=[]; page_hit_cutoff=False
        for b in blocks:
            try:
                rv=_parse_block(b,name)
                if not rv.get("Review"): continue
                if cutoff_date and rv.get("Review Date"):
                    rv_dt=_parse_review_date(rv["Review Date"])
                    if rv_dt and rv_dt<=cutoff_date: page_hit_cutoff=True; continue
                page_reviews.append(rv)
            except Exception as e: log.debug(f"Block parse error: {e}")
        reviews.extend(page_reviews)
        log.info(f"  [{name}] page {page}: +{len(page_reviews)} new"+(" [cutoff]" if page_hit_cutoff else ""))
        if page_hit_cutoff: reached_cutoff=True; break
        if len(blocks)<SCRAPER_PAGE_SIZE: break
        page+=1
    return reviews, reached_cutoff

def _dedup_key(row):
    return (str(row.get("Airline Name","")).strip(),
            str(row.get("Review Date","")).strip(),
            str(row.get("Review_Title","")).strip())

def _merge_with_existing(new_rows, existing_csv):
    new_df = pd.DataFrame(new_rows, columns=SCRAPE_COLS)
    if not Path(existing_csv).exists(): return new_df
    ex = pd.read_csv(existing_csv, low_memory=False)
    for col in SCRAPE_COLS:
        if col not in ex.columns: ex[col]=None
    ex = ex[SCRAPE_COLS]
    combined = pd.concat([new_df,ex],ignore_index=True)
    key_series = combined.apply(_dedup_key,axis=1)
    return combined[~key_series.duplicated(keep="first")].copy()

def scrape(out_csv, airlines=None, max_pages=999, delay=SCRAPER_DELAY,
           existing_csv=None, cutoff_date=None):
    driver=None
    try:
        if existing_csv and Path(existing_csv).exists() and cutoff_date is None:
            try:
                ex=pd.read_csv(existing_csv,usecols=["Review Date"],low_memory=False)
                parsed=[_parse_review_date(str(d)) for d in ex["Review Date"].dropna()]
                valid=[d for d in parsed if d]
                if valid:
                    cutoff_date=max(valid)
                    log.info(f"Incremental mode — newest: {cutoff_date.strftime('%d %B %Y')}")
            except Exception as e: log.warning(f"Could not determine cutoff: {e}")
        log.info("Launching Chrome browser …")
        driver=_make_driver(); log.info("Chrome ready.")
        all_airlines=get_airline_list(driver)
        if airlines:
            wanted={s.strip().lower() for s in airlines}
            to_scrape=[a for a in all_airlines if a["slug"].lower() in wanted]
            found={a["slug"].lower() for a in to_scrape}
            for slug in wanted-found:
                to_scrape.append({"name":slug.replace("-"," ").title(),"slug":slug,
                                   "url":f"{SCRAPER_BASE_URL}/airline-reviews/{slug}"})
        else: to_scrape=all_airlines
        mode_str=(f"incremental from {cutoff_date.strftime('%d %B %Y')}" if cutoff_date else "full scrape")
        log.info(f"Scraping {len(to_scrape)} airline(s) [{mode_str}] …")
        all_new=[]; caught_up=0
        for i,airline in enumerate(to_scrape,1):
            log.info(f"[{i}/{len(to_scrape)}] {airline['name']}")
            try:
                rows,reached=_scrape_airline(driver,airline,max_pages,delay,cutoff_date)
                all_new.extend(rows)
                if reached: caught_up+=1; log.info(f"  → caught up ✓")
            except Exception as e: log.warning(f"  Skipped {airline['name']}: {e}")
        if not all_new:
            if existing_csv and Path(existing_csv).exists():
                log.info("No new reviews — dataset already up to date.")
                return Path(existing_csv)
            log.error("Scraper returned 0 reviews."); return None
        out_csv=Path(out_csv); out_csv.parent.mkdir(parents=True,exist_ok=True)
        merged=_merge_with_existing(all_new,existing_csv or Path("__nonexistent__"))
        merged.to_csv(out_csv,index=False)
        log.info(f"\n  New this run: {len(all_new):,} | Total: {len(merged):,} | Saved → {out_csv}")
        return out_csv
    except Exception as e: log.error(f"Scraper failed: {e}"); return None
    finally:
        if driver:
            try: driver.quit(); log.info("Chrome closed.")
            except: pass
