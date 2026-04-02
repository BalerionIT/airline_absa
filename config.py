"""
config.py — Single source of truth for all constants, paths, and lexicons.
All modules import from here.

NOTE ON NLP MODELS
------------------
This pipeline uses NO external NLP model downloads.
- BIO tagging:  custom lexicon-based rule tagger (no spaCy, no NLTK)
- Sentiment:    TF-IDF + Logistic Regression trained at runtime
- AOPE:         lexicon window search (no transformers)
Everything runs offline after `pip install -r requirements.txt`.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

FALLBACK_CSV = DATA_DIR / "Airline_review.csv"
SCRAPED_CSV  = OUTPUTS_DIR / "skytrax_fresh.csv"

# ── Column schema ─────────────────────────────────────────────────────────────
SUB_COLS = [
    "Seat Comfort", "Cabin Staff Service", "Food & Beverages",
    "Ground Service", "Inflight Entertainment", "Wifi & Connectivity",
    "Value For Money",
]

SCRAPE_COLS = [
    "Airline Name", "Overall_Rating", "Review_Title", "Review Date",
    "Verified", "Review", "Aircraft", "Type Of Traveller", "Seat Type",
    "Route", "Date Flown",
    "Seat Comfort", "Cabin Staff Service", "Food & Beverages",
    "Ground Service", "Inflight Entertainment", "Wifi & Connectivity",
    "Value For Money", "Recommended",
]

# ── Aspect schema ─────────────────────────────────────────────────────────────
ASPECT_TERMS = {
    "Punctuality": [
        "delay","delayed","delays","on time","ontime","late","cancelled",
        "cancellation","cancel","schedule","departure","arrival","punctual",
        "tarmac","connection","missed connection","behind schedule","overdue",
        "hours late","flight time",
    ],
    "Cabin Comfort": [
        "seat","seats","legroom","leg room","comfortable","uncomfortable",
        "cramped","spacious","recline","reclined","cushion","headrest","cabin",
        "overhead bin","storage","narrow","pitch","armrest","window seat",
        "aisle seat","blanket","pillow",
    ],
    "Cabin Crew Service": [
        "crew","staff","attendant","flight attendant","cabin crew","steward",
        "stewardess","service","helpful","rude","friendly","unfriendly","polite",
        "impolite","attitude","professional","unprofessional","attentive",
        "smile","inattentive","ignored",
    ],
    "Food & Beverages": [
        "food","meal","drink","beverage","snack","water","coffee","tea","wine",
        "beer","juice","menu","catering","sandwich","breakfast","lunch","dinner",
        "refreshment","buy on board","complimentary","free food","no food",
        "biscuit","bread",
    ],
    "Value For Money": [
        "price","value","cheap","expensive","cost","worth","affordable",
        "overpriced","budget","fare","ticket","fee","charge","bargain","money",
        "pay","paid","euro","dollar","pound","reasonable","unreasonable",
        "rip off",
    ],
}
ASPECTS = list(ASPECT_TERMS.keys())

ASP2SUB = {
    "Punctuality":        None,
    "Cabin Comfort":      "Seat Comfort",
    "Cabin Crew Service": "Cabin Staff Service",
    "Food & Beverages":   "Food & Beverages",
    "Value For Money":    "Value For Money",
}

# ── Sentiment lexicons ────────────────────────────────────────────────────────
POS_LEX = {
    "good","great","excellent","wonderful","fantastic","amazing","nice",
    "comfortable","smooth","friendly","helpful","professional","polite","fast",
    "efficient","clean","tasty","punctual","reliable","pleasant","outstanding",
    "superb","perfect","love","loved","enjoy","enjoyed","satisfied","pleased",
    "brilliant","exceptional","on time","worth","affordable","bargain","best",
    "attentive","warm","prompt","courteous","generous","impressive","recommend",
    "recommended","spacious","delicious",
}
NEG_LEX = {
    "bad","terrible","awful","horrible","worst","poor","disappointing","rude",
    "unfriendly","unprofessional","impolite","unhelpful","dirty","cramped",
    "uncomfortable","delayed","late","cancelled","expensive","overpriced","lost",
    "missing","broken","damaged","disgusting","unacceptable","shame","avoid",
    "disorganised","chaotic","ignored","useless","boring","noisy","smelly",
    "failed","failure","problem","disgrace","pathetic","incompetent","waste",
    "rip-off",
}
NEG_WORDS = {
    "not","no","never","neither","nor","without","lack","hardly","barely",
    "cannot","can't","won't","didn't","don't","doesn't","wasn't","weren't",
    "couldn't","wouldn't","nothing",
}

OPINION_LEXICON = {
    "excellent":3,"outstanding":3,"exceptional":3,"superb":3,"fantastic":3,
    "brilliant":3,"perfect":3,"wonderful":3,"amazing":3,"extraordinary":3,
    "good":2,"great":2,"nice":2,"comfortable":2,"pleasant":2,"friendly":2,
    "helpful":2,"professional":2,"polite":2,"attentive":2,"clean":2,
    "spacious":2,"smooth":2,"efficient":2,"punctual":2,"tasty":2,"delicious":2,"warm":2,
    "ok":1,"okay":1,"fine":1,"decent":1,"adequate":1,"acceptable":1,"reasonable":1,
    "poor":-1,"basic":-1,"average":-1,"mediocre":-1,"slow":-1,"cramped":-1,
    "tight":-1,"small":-1,"narrow":-1,"limited":-1,"disappointing":-1,
    "bad":-2,"terrible":-2,"awful":-2,"horrible":-2,"rude":-2,
    "unfriendly":-2,"unprofessional":-2,"dirty":-2,"uncomfortable":-2,
    "expensive":-2,"overpriced":-2,"delayed":-2,"cancelled":-2,"cold":-2,"stale":-2,
    "disgrace":-3,"appalling":-3,"catastrophic":-3,"unacceptable":-3,
    "outrageous":-3,"shocking":-3,"pathetic":-3,"disastrous":-3,
}

# ── Visual constants ──────────────────────────────────────────────────────────
SENT_COLOR = {"positive":"#2ca02c","neutral":"#ff7f0e","negative":"#d62728"}
CLASS_COLORS = {
    "Economy Class":"#4C72B0","Business Class":"#C44E52",
    "Premium Economy":"#55A868","First Class":"#DD8452",
}
CLASSES    = list(CLASS_COLORS.keys())
TRAVELLERS = ["Solo Leisure","Couple Leisure","Family Leisure","Business"]
TRAV_COLORS = dict(zip(TRAVELLERS,["#E84855","#3A86FF","#28B463","#F39C12"]))

# ── Scraper settings ──────────────────────────────────────────────────────────
SCRAPER_BASE_URL  = "https://www.airlinequality.com"
SCRAPER_AZ_PAGE   = f"{SCRAPER_BASE_URL}/review-pages/a-z-airline-reviews/"
SCRAPER_PAGE_SIZE = 100
SCRAPER_DELAY     = (2.0, 4.0)
