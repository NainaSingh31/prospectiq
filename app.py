from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests as req
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json, re, time, os, sqlite3

from fuzzywuzzy import fuzz
from groq import Groq

app = Flask(__name__, static_folder='static')
CORS(app)

# ========= CONFIG =========
API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_key_here")
client = Groq(api_key=API_KEY)
DB_FILE = "results.db"

# ========= DATABASE =========
def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS results (
                url TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                enriched_at TEXT NOT NULL
            )
        """)
        con.commit()

def save_result(entry):
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "INSERT OR REPLACE INTO results (url, data, enriched_at) VALUES (?, ?, ?)",
            (entry["url"], json.dumps(entry), entry.get("enriched_at", ""))
        )
        con.commit()

def load_results():
    with sqlite3.connect(DB_FILE) as con:
        rows = con.execute(
            "SELECT data FROM results ORDER BY enriched_at DESC"
        ).fetchall()
    return [json.loads(r[0]) for r in rows]

# ========= SCRAPING =========
RELEVANT_KEYWORDS = [
    "about", "contact", "service", "solution",
    "team", "product", "company", "who-we-are", "what-we-do"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def get_html(url, timeout=12):
    try:
        r = req.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception:
        try:
            r = req.get(url, headers=HEADERS, timeout=timeout, verify=False, allow_redirects=True)
            return r.text
        except Exception:
            return None

def get_sitemap_links(base_url):
    for path in ["/sitemap.xml", "/sitemap_index.xml"]:
        try:
            r = req.get(base_url.rstrip("/") + path, headers=HEADERS, timeout=8)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "lxml-xml")
                links = [loc.text for loc in soup.find_all("loc")]
                if links:
                    return links
        except Exception:
            pass
    return []

def extract_links(base_url, html):
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"].strip())
        if urlparse(full).netloc == base_domain:
            links.add(full.split("#")[0].split("?")[0])
    return list(links)

def score_link(url):
    path = urlparse(url).path.lower()
    return max(fuzz.partial_ratio(path, kw) for kw in RELEVANT_KEYWORDS)

def select_best_pages(base_url, homepage_html, max_pages=4):
    all_links = get_sitemap_links(base_url)
    if not all_links:
        all_links = extract_links(base_url, homepage_html)
    scored = sorted(all_links, key=score_link, reverse=True)
    best = [base_url] + [
        l for l in scored if l != base_url and score_link(l) > 45
    ][:max_pages]
    return best

def clean_html(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg",
                     "nav", "footer", "header", "aside", "form",
                     "button", "input", "select", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all(True):
        cls = " ".join(tag.get("class", []))
        id_ = tag.get("id", "")
        if any(kw in cls.lower() or kw in id_.lower()
               for kw in ["cookie", "banner", "popup", "modal", "overlay", "gdpr", "consent"]):
            tag.decompose()
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if len(l.strip()) > 20]
    return "\n".join(lines)

def extract_contacts_regex(text):
    emails = list(set(re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)))
    emails = [e for e in emails if not any(
        x in e.lower() for x in ["example", "test", "domain", "sentry", "wix", "wordpress"]
    )]
    phones = list(set([
        p.strip() for p in re.findall(r"(\+?[\d][\d\s\-().]{6,}[\d])", text)
        if 7 <= len(re.sub(r"\D", "", p)) <= 15
    ]))
    return emails[:5], phones[:3]

def scrape_company(base_url):
    base_url = base_url.rstrip("/")
    if not base_url.startswith("http"):
        base_url = "https://" + base_url

    homepage_html = get_html(base_url)
    if not homepage_html:
        return None, [], []

    pages = select_best_pages(base_url, homepage_html)
    all_text = ""
    for page_url in pages:
        time.sleep(0.4)
        html = get_html(page_url)
        if html:
            cleaned = clean_html(html)
            if len(cleaned.split()) > 20:
                all_text += f"\n\n--- PAGE: {page_url} ---\n" + cleaned

    if len(all_text.split()) < 100:
        for fallback in ["/about", "/about-us", "/contact", "/services"]:
            html = get_html(base_url + fallback)
            if html:
                cleaned = clean_html(html)
                all_text += f"\n\n--- PAGE: {base_url+fallback} ---\n" + cleaned

    words = all_text.split()
    if len(words) > 6000:
        all_text = " ".join(words[:6000])

    emails, phones = extract_contacts_regex(all_text)
    return all_text if all_text.strip() else None, emails, phones

# ========= LLM =========
def call_llm(prompt):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a B2B research analyst. Extract ONLY information explicitly present "
                            "in the provided website text. NEVER invent or hallucinate any data. "
                            "If a field is not found, return empty string or N/A. "
                            "Respond ONLY with valid JSON. No markdown, no extra text."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=900,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            # Guard: must be a dict
            if not isinstance(parsed, dict):
                raise ValueError("LLM returned non-dict JSON")
            return parsed
        except json.JSONDecodeError:
            time.sleep(1)
        except Exception as e:
            print(f"LLM attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def build_prompt(url, text, emails, phones):
    email_hint = ", ".join(emails) if emails else "not found"
    phone_hint = ", ".join(phones) if phones else "not found"
    return f"""
Analyze this website content for: {url}

Emails extracted by regex (use AS-IS, do not alter): {email_hint}
Phones extracted by regex (use AS-IS, do not alter): {phone_hint}

=== WEBSITE CONTENT ===
{text[:4500]}
=== END ===

Return ONLY this JSON (no extra text, no markdown):
{{
  "website_name": "Short brand name visible on the site",
  "company_name": "Full official company name",
  "address": "Physical address if explicitly stated, else N/A",
  "mobile_number": "Use the phone(s) from regex above only. If none, return N/A",
  "mail": ["Use emails from regex above only. If none, return empty list []"],
  "core_service": "Primary product or service in 1 sentence",
  "target_customer": "Who they sell to — industry, size, persona",
  "probable_pain_point": "The core business problem their customers face",
  "outreach_opener": "2-sentence personalized cold outreach referencing something specific from their site"
}}

STRICT RULES:
- mail and mobile_number: ONLY use regex values above. Do NOT invent contact info.
- address: only if explicitly on site.
- outreach_opener: must reference a real specific detail from their website.
"""

def enrich_company(url: str) -> dict:
    EMPTY = {
        "website_name": "N/A", "company_name": "N/A", "address": "N/A",
        "mobile_number": "N/A", "mail": [], "core_service": "N/A",
        "target_customer": "N/A", "probable_pain_point": "N/A", "outreach_opener": "N/A"
    }

    text, emails, phones = scrape_company(url)
    if not text or len(text.split()) < 30:
        return {**EMPTY, "website_name": url}

    prompt = build_prompt(url, text, emails, phones)
    result = call_llm(prompt)

    # FIX: guard against None or non-dict LLM response (was causing Zoho-style crash)
    if result is None or not isinstance(result, dict):
        return {**EMPTY, "website_name": url}

    result["mail"] = emails if emails else result.get("mail", [])
    result["mobile_number"] = phones[0] if phones else result.get("mobile_number", "N/A")

    for key in EMPTY:
        if key not in result:
            result[key] = EMPTY[key]

    return result

# ========= ROUTES =========
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/enrich", methods=["POST"])
def enrich():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    url = data.get("url", "").strip()
    website_label = data.get("website_name", "").strip()

    if not url:
        return jsonify({"error": "URL is required"}), 400

    # Basic URL validation
    try:
        parsed = urlparse(url if url.startswith("http") else "https://" + url)
        if not parsed.netloc:
            return jsonify({"error": "Invalid URL format"}), 400
    except Exception:
        return jsonify({"error": "Invalid URL format"}), 400

    try:
        result = enrich_company(url)
        if website_label:
            result["website_name"] = website_label
        result["url"] = url
        result["enriched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_result(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/results", methods=["GET"])
def results():
    return jsonify(load_results())

# ========= STARTUP =========
init_db()

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5000)
