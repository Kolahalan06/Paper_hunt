import streamlit as st
import feedparser
import requests
import csv
import io
import re
from datetime import datetime, timedelta
import urllib.parse

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove symbols
    return text
# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

try:
    from transformers import pipeline
    SUMMARIZER_AVAILABLE = True
except Exception:
    SUMMARIZER_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperHunt",
    page_icon="🔬",
    layout="wide",
)



# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'Space Mono', monospace; }

    .paper-card {
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem 1.2rem 0.6rem;
        margin-bottom: 0.6rem;
        background: #0e0e0e;
        transition: border-color 0.2s;
    }
    .paper-card:hover { border-color: #00e5ff; }
    .paper-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #00e5ff;
        text-decoration: none;
    }
    .paper-title:hover { text-decoration: underline; }
    .score-badge {
        display: inline-block;
        background: #00e5ff15;
        border: 1px solid #00e5ff44;
        color: #00e5ff;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.75rem;
        margin-left: 8px;
    }
    .source-badge {
        display: inline-block;
        border-radius: 3px;
        padding: 1px 7px;
        font-size: 0.72rem;
        margin-left: 6px;
        font-weight: bold;
    }
    .source-arxiv  { background:#1a1a2e; color:#8888ff; border:1px solid #444; }
    .source-ss     { background:#1a2e1a; color:#88ff88; border:1px solid #444; }
    .source-pubmed { background:#2e1a1a; color:#ff8888; border:1px solid #444; }

    .novelty-box {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        background: #0a0a0a;
        margin-bottom: 1.2rem;
    }
    .novelty-label { font-size: 1.5rem; font-weight: bold; }
    .novel-high    { color: #00e676; }
    .novel-medium  { color: #ffea00; }
    .novel-low     { color: #ff5252; }

    .export-bar {
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if not EMBEDDING_AVAILABLE:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    if not SUMMARIZER_AVAILABLE:
        return None
    for mid in ["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn"]:
        try:
            return pipeline("summarization", model=mid)
        except Exception:
            continue
    return None


# ── arXiv fetcher ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arxiv(tech: str, domain: str, max_results: int, days_back: int):
    q = f"all:{tech}" if domain == "All" else f"all:{tech} AND all:{domain.lower().replace(' ', '_')}"
    encoded = urllib.parse.quote(q)
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={encoded}&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    feed = feedparser.parse(url)
    date_limit = datetime.now() - timedelta(days=days_back)
    papers = []
    for entry in feed.entries:
        pub = None
        for field in ("published", "updated"):
            try:
                pub = datetime.strptime(getattr(entry, field, ""), "%Y-%m-%dT%H:%M:%SZ")
                break
            except Exception:
                pass
        if pub is None or pub < date_limit:
            continue
        authors = ", ".join(a.name for a in getattr(entry, "authors", [])) or "Unknown"
        # extract arXiv ID for BibTeX key
        arxiv_id = re.search(r"\d{4}\.\d+", entry.link)
        arxiv_id = arxiv_id.group() if arxiv_id else entry.link.split("/")[-1]
        papers.append({
            "source":    "arXiv",
            "title":     entry.title.strip(),
            "link":      entry.link,
            "abstract":  getattr(entry, "summary", ""),
            "published": pub,
            "authors":   authors,
            "citations": None,
            "score":     None,
            "doi":       None,
            "pmid":      None,
            "arxiv_id":  arxiv_id,
            "journal":   "arXiv preprint",
        })
    return papers


# ── Semantic Scholar fetcher ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_semantic_scholar(tech: str, domain: str, max_results: int, days_back: int):
    query  = tech if domain == "All" else f"{tech} {domain}"
    url    = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query,
        "limit":  min(max_results, 100),
        "fields": "title,abstract,authors,year,citationCount,url,externalIds,venue",
    }
    try:
        resp = requests.get(url, params=params, timeout=10,
                            headers={"User-Agent": "PaperHunt/1.0"})
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    year_limit = datetime.now().year - (days_back // 365) - 1
    papers = []
    for item in data.get("data", []):
        if not item.get("abstract"):
            continue
        year = item.get("year") or 0
        if year < year_limit:
            continue
        authors = ", ".join(a.get("name", "") for a in item.get("authors", [])) or "Unknown"
        ext  = item.get("externalIds") or {}
        link = (
            f"https://arxiv.org/abs/{ext['ArXiv']}"
            if ext.get("ArXiv")
            else item.get("url")
            or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}"
        )
        try:
            pub = datetime(int(year), 1, 1)
        except Exception:
            pub = datetime(2000, 1, 1)

        arxiv_id = ext.get("ArXiv")
        papers.append({
            "source":    "Semantic Scholar",
            "title":     (item.get("title") or "").strip(),
            "link":      link,
            "abstract":  item.get("abstract", ""),
            "published": pub,
            "authors":   authors,
            "citations": item.get("citationCount"),
            "score":     None,
            "doi":       ext.get("DOI"),
            "pmid":      ext.get("PubMed"),
            "arxiv_id":  arxiv_id,
            "journal":   item.get("venue") or "Semantic Scholar",
        })
    return papers


# ── PubMed fetcher ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pubmed(tech: str, domain: str, max_results: int, days_back: int):
    query = tech if domain == "All" else f"{tech} {domain}"
    # Step 1 — search for IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "date",
        "datetype":"pdat",
        "reldate": days_back,
    }
    try:
        r = requests.get(search_url, params=search_params, timeout=10)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

    if not ids:
        return []

    # Step 2 — fetch details for those IDs
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db":      "pubmed",
        "id":      ",".join(ids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    try:
        r = requests.get(fetch_url, params=fetch_params, timeout=15)
        r.raise_for_status()
        xml = r.text
    except Exception:
        return []

    # Parse XML with regex (no lxml needed)
    articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
    papers = []
    for article in articles:
        # Title
        title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
        title   = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""
        if not title:
            continue

        # Abstract
        abstract_m = re.search(r"<AbstractText.*?>(.*?)</AbstractText>", article, re.DOTALL)
        abstract   = re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip() if abstract_m else ""

        # Authors
        author_blocks = re.findall(r"<Author[^>]*>(.*?)</Author>", article, re.DOTALL)
        author_names  = []
        for block in author_blocks:
            last  = re.search(r"<LastName>(.*?)</LastName>",  block)
            first = re.search(r"<ForeName>(.*?)</ForeName>", block)
            if last:
                name = last.group(1)
                if first:
                    name += f" {first.group(1)}"
                author_names.append(name)
        authors = ", ".join(author_names) or "Unknown"

        # PMID
        pmid_m = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
        pmid   = pmid_m.group(1) if pmid_m else ""

        # DOI
        doi_m = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
        doi   = doi_m.group(1).strip() if doi_m else None

        # Journal
        journal_m = re.search(r"<Title>(.*?)</Title>", article)
        journal   = journal_m.group(1).strip() if journal_m else "PubMed"

        # Date
        year_m  = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>",  article, re.DOTALL)
        month_m = re.search(r"<PubDate>.*?<Month>(\w+)</Month>", article, re.DOTALL)
        year    = int(year_m.group(1)) if year_m else 2000
        month_str = month_m.group(1) if month_m else "Jan"
        try:
            pub = datetime.strptime(f"{year} {month_str}", "%Y %b")
        except Exception:
            try:
                pub = datetime(year, int(month_str), 1)
            except Exception:
                pub = datetime(year, 1, 1)

        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        papers.append({
            "source":    "PubMed",
            "title":     title,
            "link":      link,
            "abstract":  abstract,
            "published": pub,
            "authors":   authors,
            "citations": None,
            "score":     None,
            "doi":       doi,
            "pmid":      pmid,
            "arxiv_id":  None,
            "journal":   journal,
        })
    return papers

STOPWORDS = set([
    "the", "and", "is", "in", "to", "of", "for", "on", "with",
    "a", "an", "by", "from", "at", "as", "this", "that",
    "we", "our", "their", "be", "are", "was", "were"
])

def extract_keywords(text):
    words = clean_text(text).split()
    return set([w for w in words if w not in STOPWORDS and len(w) > 3])

def get_match_keywords(user_text, paper_text, top_k=5):
    user_words = extract_keywords(user_text)
    paper_words = extract_keywords(paper_text)

    common = user_words.intersection(paper_words)

    # prioritize longer / more meaningful words
    common = sorted(common, key=lambda x: -len(x))

    return list(common)[:top_k]

def generate_explanation(match_words):
    if not match_words:
        return "This paper is related but uses different terminology."

    return (
        "This paper shares key concepts like "
        + ", ".join(match_words[:3])
        + ", indicating a similar approach or problem domain."
    )

# ── Export helpers ────────────────────────────────────────────────────────────
def _bibtex_key(paper: dict, idx: int) -> str:
    """Generate a citekey like Smith2023 or arxiv2024_01"""
    first_author = paper["authors"].split(",")[0].split()[-1] if paper["authors"] != "Unknown" else "Unknown"
    first_author = re.sub(r"[^A-Za-z]", "", first_author)
    year = paper["published"].year
    return f"{first_author}{year}_{idx}"


def papers_to_bibtex(papers: list) -> str:
    lines = []
    for idx, p in enumerate(papers, 1):
        key     = _bibtex_key(p, idx)
        authors = p["authors"].replace(", ", " and ")
        year    = p["published"].year
        title   = p["title"].replace("{", "").replace("}", "")
        journal = p.get("journal") or p["source"]
        doi_line     = f"  doi       = {{{p['doi']}}},\n"     if p.get("doi")      else ""
        pmid_line    = f"  note      = {{PMID: {p['pmid']}}},\n" if p.get("pmid")   else ""
        arxiv_line   = f"  eprint    = {{{p['arxiv_id']}}},\n"  if p.get("arxiv_id") else ""
        abstract_line = f"  abstract  = {{{p['abstract'][:500].replace('{','').replace('}','')}}},\n" if p.get("abstract") else ""

        entry = (
            f"@article{{{key},\n"
            f"  title     = {{{title}}},\n"
            f"  author    = {{{authors}}},\n"
            f"  year      = {{{year}}},\n"
            f"  journal   = {{{journal}}},\n"
            f"  url       = {{{p['link']}}},\n"
            f"{doi_line}{pmid_line}{arxiv_line}{abstract_line}"
            f"}}\n"
        )
        lines.append(entry)
    return "\n".join(lines)


def papers_to_csv(papers: list) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "title", "authors", "year", "source", "journal",
        "citations", "score", "doi", "pmid", "arxiv_id", "link", "abstract"
    ])
    writer.writeheader()
    for p in papers:
        writer.writerow({
            "title":     p["title"],
            "authors":   p["authors"],
            "year":      p["published"].year,
            "source":    p["source"],
            "journal":   p.get("journal", ""),
            "citations": p.get("citations", ""),
            "score":     f"{p['score']:.4f}" if p.get("score") is not None else "",
            "doi":       p.get("doi", ""),
            "pmid":      p.get("pmid", ""),
            "arxiv_id":  p.get("arxiv_id", ""),
            "link":      p["link"],
            "abstract":  p["abstract"][:300],
        })
    return output.getvalue()


def render_export_bar(papers: list, label: str = "results"):
    """Render the BibTeX + CSV download buttons."""
    st.markdown('<div class="export-bar">', unsafe_allow_html=True)
    st.markdown(f"**📦 Export {len(papers)} {label}**")
    col_b, col_c, _ = st.columns([1, 1, 4])
    with col_b:
        st.download_button(
            label="⬇️ BibTeX (.bib)",
            data=papers_to_bibtex(papers),
            file_name="paperhunt_results.bib",
            mime="text/plain",
            use_container_width=True,
        )
    with col_c:
        st.download_button(
            label="⬇️ CSV (.csv)",
            data=papers_to_csv(papers),
            file_name="paperhunt_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
DOMAINS = [
    "All", "Healthcare", "Defense", "Finance", "Education", "Robotics",
    "Energy", "Transportation", "Agriculture", "Space", "Climate Science",
    "Cybersecurity", "Quantum Computing", "Blockchain", "Social Sciences",
    "Astrophysics",
]

st.sidebar.title("🔬 PaperHunt")
st.sidebar.markdown("*Multi-source semantic paper explorer*")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["🔍 Paper Search", "🧪 Novelty Checker"], horizontal=True)
st.sidebar.markdown("---")

search_term     = st.sidebar.text_input("AI Technique / Topic", placeholder="e.g. deep learning, RAG, ViT")
selected_domain = st.sidebar.selectbox("Application Domain", DOMAINS)
max_results     = st.sidebar.number_input("Papers to fetch per source", 1, 100, 25)
if max_results < 10:
    st.sidebar.warning("⚠️ Try at least 10 for better results.")
display_count   = st.sidebar.number_input("Top results to display", 1, int(max_results) * 3, 10)
days_back       = st.sidebar.slider("Published within (days)", 1, 3650, 730)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sources**")
use_arxiv  = st.sidebar.checkbox("arXiv",             value=True)
use_ss     = st.sidebar.checkbox("Semantic Scholar",   value=True)
use_pubmed = st.sidebar.checkbox("PubMed",             value=False)

st.sidebar.markdown("**Options**")
use_semantic   = st.sidebar.checkbox("Semantic ranking", value=True,  disabled=not EMBEDDING_AVAILABLE)
use_summarizer = st.sidebar.checkbox("AI summaries",     value=False, disabled=not SUMMARIZER_AVAILABLE)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model status**")
st.sidebar.write(f"{'✅' if EMBEDDING_AVAILABLE else '❌'} Embeddings (MiniLM)")
st.sidebar.write(f"{'✅' if SUMMARIZER_AVAILABLE else '❌'} Summarizer (DistilBART)")

search_btn = st.sidebar.button("🔍 Search / Analyse", use_container_width=True, type="primary")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔬 PaperHunt")
st.markdown("Search **arXiv · Semantic Scholar · PubMed** — ranked by semantic similarity.")

if not search_btn:
    st.info("👈 Fill in the sidebar and hit **Search / Analyse**.")
    st.stop()

if not search_term.strip():
    st.warning("Please enter a search topic.")
    st.stop()

tech = search_term.strip()

# ── Fetch from selected sources ───────────────────────────────────────────────
all_papers = []

if use_arxiv:
    with st.spinner("Fetching from arXiv…"):
        ap = fetch_arxiv(tech, selected_domain, int(max_results), days_back)
    st.caption(f"arXiv → {len(ap)} papers")
    all_papers.extend(ap)

if use_ss:
    with st.spinner("Fetching from Semantic Scholar…"):
        sp = fetch_semantic_scholar(tech, selected_domain, int(max_results), days_back)
    st.caption(f"Semantic Scholar → {len(sp)} papers")
    all_papers.extend(sp)

if use_pubmed:
    with st.spinner("Fetching from PubMed…"):
        pp = fetch_pubmed(tech, selected_domain, int(max_results), days_back)
    st.caption(f"PubMed → {len(pp)} papers")
    all_papers.extend(pp)

if not all_papers:
    st.error("No papers found from any source.")
    with st.expander("💡 Tips"):
        st.markdown("""
- Use full words: `machine learning` not `ml`
- Try domain = **All**
- Increase fetch count and days range
- Enable more sources in the sidebar
        """)
    st.stop()

# ── Deduplicate by title ──────────────────────────────────────────────────────
seen, deduped = set(), []
for p in all_papers:
    key = re.sub(r'\s+', ' ', p["title"].lower().strip())
    if key not in seen:
        seen.add(key)
        deduped.append(p)
all_papers = deduped
st.caption(f"**{len(all_papers)} unique papers** after deduplication.")

# ── Load embedding model once ─────────────────────────────────────────────────
emb_model = None
if EMBEDDING_AVAILABLE and (use_semantic or mode == "🧪 Novelty Checker"):
    with st.spinner("Loading embedding model…"):
        emb_model = load_embedding_model()


# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — PAPER SEARCH
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔍 Paper Search":

    if use_semantic and emb_model:
        with st.spinner("Computing semantic similarity…"):
            docs   = [p["title"] + ". " + re.sub(r'\s+', ' ', p["abstract"]) for p in all_papers]
            p_embs = emb_model.encode(docs, convert_to_tensor=True, show_progress_bar=False)
            sem_q  = tech if selected_domain == "All" else f"{tech} applied to {selected_domain}"
            q_emb  = emb_model.encode(sem_q, convert_to_tensor=True, show_progress_bar=False)
            sims   = util.cos_sim(q_emb, p_embs)[0].cpu().numpy()
            for i, s in enumerate(sims):
                all_papers[i]["score"] = float(s)
        all_papers = sorted(all_papers, key=lambda x: x["score"] or 0, reverse=True)

    summarizer = None
    if use_summarizer and SUMMARIZER_AVAILABLE:
        with st.spinner("Loading summarizer…"):
            summarizer = load_summarizer_model()

    st.markdown("---")

    # ── Export bar ────────────────────────────────────────────────────────────
    display_papers = all_papers[:int(display_count)]
    render_export_bar(display_papers, label="displayed papers")

    st.markdown(f"### Top {len(display_papers)} results")

    for i, paper in enumerate(display_papers):
        score_html = (
            f'<span class="score-badge">sim {paper["score"]:.3f}</span>'
            if paper["score"] is not None else ""
        )
        src_map = {
            "arXiv":            "source-arxiv",
            "Semantic Scholar": "source-ss",
            "PubMed":           "source-pubmed",
        }
        src_cls   = src_map.get(paper["source"], "source-ss")
        cite_html = (
            f'<span class="score-badge">📚 {paper["citations"]} citations</span>'
            if paper["citations"] is not None else ""
        )
        doi_html = (
            f'<span class="score-badge">DOI</span>'
            if paper.get("doi") else ""
        )

        st.markdown(
            f'<div class="paper-card">'
            f'<a class="paper-title" href="{paper["link"]}" target="_blank">{i+1}. {paper["title"]}</a>'
            f'<span class="source-badge {src_cls}">{paper["source"]}</span>'
            f'{score_html}{cite_html}{doi_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        authors_short = paper["authors"][:120] + ("…" if len(paper["authors"]) > 120 else "")
        st.caption(f"📅 {paper['published'].strftime('%Y-%m-%d')}  |  ✍️ {authors_short}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if "arxiv.org/abs" in paper["link"]:
                pdf_url = paper["link"].replace("/abs/", "/pdf/")
                st.markdown(f"[📄 PDF]({pdf_url})")
            elif paper["link"]:
                st.markdown(f"[🔗 Open]({paper['link']})")
            if paper.get("doi"):
                st.markdown(f"[🔗 DOI](https://doi.org/{paper['doi']})")
            github_links = re.findall(
                r"(https?://github\.com[^\s\)\]\.,<>]+)",
                paper["abstract"], flags=re.IGNORECASE,
            )
            for repo in github_links[:2]:
                st.markdown(f"[💻 GitHub]({repo})")

        with col2:
            if summarizer:
                try:
                    text    = " ".join(paper["abstract"].split()[:512])
                    summary = summarizer(text, max_length=80, min_length=20, do_sample=False)
                    st.markdown(f"**AI Summary:** {summary[0]['summary_text']}")
                except Exception:
                    with st.expander("Abstract"):
                        st.write(paper["abstract"])
            else:
                with st.expander("Abstract"):
                    st.write(paper["abstract"])

    st.success(f"✅ Showing {len(display_papers)} of {len(all_papers)} papers.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — NOVELTY CHECKER
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("---")
    st.markdown("### 🧪 Novelty Checker")

    user_idea = st.text_area(
        "Your research idea / abstract",
        height=160,
        placeholder="Describe your idea..."
    )

    if not user_idea.strip():
        st.info("👆 Paste your research idea above to run the analysis.")
        st.stop()

    if not emb_model:
        st.error("❌ Embedding model not available.")
        st.stop()

    with st.spinner("Analysing novelty…"):
        docs     = [p["title"] + ". " + re.sub(r'\s+', ' ', p["abstract"]) for p in all_papers]
        p_embs   = emb_model.encode(docs, convert_to_tensor=True, show_progress_bar=False)
        idea_emb = emb_model.encode(user_idea.strip(), convert_to_tensor=True, show_progress_bar=False)
        sims     = util.cos_sim(idea_emb, p_embs)[0].cpu().numpy()
        for i, s in enumerate(sims):
            all_papers[i]["score"] = float(s)

    ranked   = sorted(all_papers, key=lambda x: x["score"], reverse=True)
    top5     = ranked[:5]

    # ── Export top 5 matches ──────────────────────────────────────────────────
    render_export_bar(top5, label="closest matching papers")

    # ── Top 5 most similar ────────────────────────────────────────────────────
    st.markdown("#### 📄 Most similar existing papers")
    st.caption("Papers your idea overlaps with the most — read these before writing your proposal.")

    src_map = {"arXiv": "source-arxiv", "Semantic Scholar": "source-ss", "PubMed": "source-pubmed"}

    for i, paper in enumerate(top5):
        overlap_pct = round(paper["score"] * 100, 1)
        src_cls     = src_map.get(paper["source"], "source-ss")

    # 👉 CARD
    st.markdown(
        f'<div class="paper-card">'
        f'<a class="paper-title" href="{paper["link"]}" target="_blank">{i+1}. {paper["title"]}</a>'
        f'<span class="source-badge {src_cls}">{paper["source"]}</span>'
        f'<span class="score-badge">overlap {overlap_pct}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 👉 AUTHORS
    authors_short = paper["authors"][:100] + ("…" if len(paper["authors"]) > 100 else "")
    st.caption(f"📅 {paper['published'].strftime('%Y')}  |  ✍️ {authors_short}")

    # =========================
    # 🔥 EXPLAINABILITY (FIXED POSITION)
    # =========================
    paper_text = (paper.get("title", "") + " " + paper.get("abstract", "")).strip()
    match_words = get_match_keywords(user_idea, paper_text)

    if match_words:
        match_words = [w[:15] for w in match_words]

        st.markdown(f"🧠 **Matches:** {', '.join(match_words)}")
        explanation = generate_explanation(match_words)
        st.markdown(f"💡 *{explanation}*")
    # =========================

    # 👉 DOI
    if paper.get("doi"):
        st.markdown(f"[🔗 DOI](https://doi.org/{paper['doi']})")

    # 👉 ABSTRACT
    with st.expander("Abstract"):
        st.write(paper["abstract"])


        # =========================
        # 🔥 KEYWORD MATCHING
        # =========================
        paper_text = (paper.get("title", "") + " " + paper.get("abstract", "")).strip()
        match_words = get_match_keywords(user_idea, paper_text)
        if match_words:
            match_words = [w[:15] for w in match_words]  # clean UI
            st.caption(f"🧠 Matches: {', '.join(match_words)}")

        if paper.get("doi"):
            st.markdown(f"[🔗 DOI](https://doi.org/{paper['doi']})")

        with st.expander("Abstract"):
            st.write(paper["abstract"])

    # ── Research gaps ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🌱 Related but less explored directions")
    st.caption("Lower overlap with your idea — potential research gaps worth exploring.")

    gap_papers = ranked[-5:][::-1]
    for paper in gap_papers:
        overlap_pct = round(paper["score"] * 100, 1)
        src_cls     = src_map.get(paper["source"], "source-ss")
        st.markdown(
            f'<div class="paper-card">'
            f'<a class="paper-title" href="{paper["link"]}" target="_blank">{paper["title"]}</a>'
            f'<span class="source-badge {src_cls}">{paper["source"]}</span>'
            f'<span class="score-badge">overlap {overlap_pct}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"📅 {paper['published'].strftime('%Y')}  |  ✍️ {paper['authors'][:100]}")
