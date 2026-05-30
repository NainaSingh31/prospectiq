# ProspectIQ — AI Company Intelligence

Hackathon submission for Relu Consultancy AI & Automation Developer challenge.

## Local Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # Windows: set GROQ_API_KEY=your_key_here
python app.py
# Open http://localhost:5000
```

## Deploy to Render (Free)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set environment variable: GROQ_API_KEY = your key
5. Build command: `pip install -r requirements.txt`
6. Start command: `gunicorn app:app`
7. Click Deploy → get your public URL

## API Endpoints

- `POST /enrich` — body: `{"url": "https://...", "website_name": "optional label"}`
- `GET /results` — returns all enriched companies

## Tech Stack

- Backend: Flask + Gunicorn
- Scraping: requests + BeautifulSoup + fuzzy matching
- AI: Groq LLaMA 3.3 70B
- Frontend: Vanilla HTML/CSS/JS
