import streamlit as st
import feedparser
from datetime import datetime, timedelta
import urllib.parse
import re

try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

try:
    from transformers import pipeline
    SUMMARIZER_AVAILABLE = True
except Exception:
    SUMMARIZER_AVAILABLE = False
st.set_page_config(
    page_title="PaperHunt – arXiv Semantic Search",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace;
    }
    .paper-card {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        background: #0e0e0e;
        transition: border-color 0.2s;
    }
    .paper-card:hover { border-color: #00e5ff; }
    .paper-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.15rem;
        color: #00e5ff;
        text-decoration: none;
    }
    .paper-title:hover { text-decoration: underline; }
    .score-badge {
        display: inline-block;
        background: #00e5ff22;
        border: 1px solid #00e5ff55;
        color: #00e5ff;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-left: 8px;
    }
    .meta-line { color: #888; font-size: 0.8rem; margin: 4px 0 8px; }
    .tag {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #444;
        border-radius: 3px;
        padding: 1px 7px;
        font-size: 0.75rem;
        color: #aaa;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if not EMBEDDING_AVAILABLE:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    if not SUMMARIZER_AVAILABLE:
        return None
    for model_id in ["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn"]:
        try:
            return pipeline("summarization", model=model_id)
        except Exception:
            continue
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_papers(search_query_str: str, max_results: int, days_back: int):
    encoded = urllib.parse.quote(search_query_str)
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
        for fmt in ('%Y-%m-%dT%H:%M:%SZ',):
            for field in ('published', 'updated'):
                try:
                    pub = datetime.strptime(getattr(entry, field, ""), fmt)
                    break
                except Exception:
                    pass
            if pub:
                break
        if pub is None or pub < date_limit:
            continue

        authors = ", ".join(
            a.name for a in getattr(entry, "authors", [])
        ) if getattr(entry, "authors", None) else "Unknown"

        papers.append({
            "id": getattr(entry, "id", ""),
            "title": entry.title.strip(),
            "link": entry.link,
            "abstract": getattr(entry, "summary", ""),
            "published": pub,
            "authors": authors,
            "score": None,
        })
    return papers

DOMAINS = [
    "All", "Healthcare", "Defense", "Finance", "Education", "Robotics",
    "Energy", "Transportation", "Agriculture", "Space", "Climate Science",
    "Cybersecurity", "Quantum Computing", "Blockchain", "Social Sciences",
    "Astrophysics",
]

st.sidebar.title("🔬 PaperHunt")
st.sidebar.markdown("*Semantic arXiv explorer*")
st.sidebar.markdown("---")

search_term     = st.sidebar.text_input("AI Technique", placeholder="e.g. deep learning, ViT, RAG").upper()
selected_domain = st.sidebar.selectbox("Application Domain", DOMAINS)
max_results     = st.sidebar.number_input("Papers to fetch from arXiv", 1, 100, 30)
if max_results < 10:
    st.sidebar.warning("⚠️ Try at least 10 for better results.")
display_count   = st.sidebar.number_input("Top results to display", 1, int(max_results), min(10, int(max_results)))
days_back       = st.sidebar.slider("Published within (days)", 1, 3650, 365)
use_semantic    = st.sidebar.checkbox("Semantic ranking", value=True, disabled=not EMBEDDING_AVAILABLE)
use_summarizer  = st.sidebar.checkbox("AI summaries", value=False, disabled=not SUMMARIZER_AVAILABLE)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model status**")
st.sidebar.write(f"{'✅' if EMBEDDING_AVAILABLE else '❌'} Embeddings (MiniLM)")
st.sidebar.write(f"{'✅' if SUMMARIZER_AVAILABLE else '❌'} Summarizer (DistilBART)")

search_btn = st.sidebar.button("🔍 Search", use_container_width=True, type="primary")
st.markdown("## 🔬 PaperHunt")
st.markdown("Find arXiv papers by AI technique × application domain, ranked by semantic similarity.")

if not search_btn:
    st.info("👈 Enter a technique and domain in the sidebar, then hit **Search**.")
    st.stop()

if not search_term.strip():
    st.warning("Please enter a search technique (e.g. 'machine learning').")
    st.stop()

tech = search_term.strip()
# Use loose keyword search (no quotes) so arXiv matches naturally.
# Quoted exact-match kills results for short terms like "ml" or "cv".
if selected_domain == "All":
    search_query_str = f"all:{tech}"
else:
    search_query_str = f"all:{tech} AND all:{selected_domain.lower().replace(' ', '_')}"

with st.expander("🛠 Debug — arXiv query being sent", expanded=False):
    st.code(search_query_str)

with st.spinner(f"Fetching papers from arXiv for **{tech}**…"):
    papers = fetch_papers(search_query_str, int(max_results), days_back)

if not papers:
    st.error("No papers found. Try increasing *days back* or *max results*, or broaden your search terms.")
    with st.expander("💡 Tips to fix this"):
        st.markdown("""
- Use full words: `machine learning` not `ml`, `computer vision` not `cv`
- Try selecting **All** as domain first
- Increase **Papers to fetch** to at least 20
- Increase **Published within** to 1000+ days
        """)
    st.stop()

st.caption(f"Fetched **{len(papers)}** papers matching the query.")

# ── Semantic ranking 
if use_semantic and EMBEDDING_AVAILABLE:
    with st.spinner("Computing semantic similarity…"):
        model = load_embedding_model()
        docs = [p["title"] + ". " + re.sub(r'\s+', ' ', p["abstract"]) for p in papers]
        paper_embs = model.encode(docs, convert_to_tensor=True, show_progress_bar=False)

        # Use the human-readable query (tech + domain) as the anchor
        semantic_query = tech if selected_domain == "All" else f"{tech} applied to {selected_domain}"
        query_emb = model.encode(semantic_query, convert_to_tensor=True, show_progress_bar=False)

        scores = util.cos_sim(query_emb, paper_embs)[0].cpu().numpy()
        for i, score in enumerate(scores):
            papers[i]["score"] = float(score)

    papers = sorted(papers, key=lambda x: x["score"], reverse=True)

# ── Summarizer 
summarizer = None
if use_summarizer and SUMMARIZER_AVAILABLE:
    with st.spinner("Loading summarizer model (one-time, may take a minute)…"):
        summarizer = load_summarizer_model()
    if summarizer is None:
        st.warning("Summarizer could not be loaded — showing abstracts instead.")

# ── Results
st.markdown("---")
st.markdown(f"### Top {min(display_count, len(papers))} results")

for i, paper in enumerate(papers[:display_count]):
    score_html = (
        f'<span class="score-badge">score {paper["score"]:.3f}</span>'
        if paper["score"] is not None else ""
    )

    # GitHub repos 
    github_links = re.findall(
        r"(https?://github\.com[^\s\)\]\.,<>]+)",
        paper["abstract"],
        flags=re.IGNORECASE,
    )

    with st.container():
        st.markdown(
            f'<div class="paper-card">'
            f'<a class="paper-title" href="{paper["link"]}" target="_blank">{i+1}. {paper["title"]}</a>'
            f'{score_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        authors_short = paper['authors'][:120] + ('…' if len(paper['authors']) > 120 else '')
        st.caption(f"📅 {paper['published'].strftime('%Y-%m-%d')}  |  ✍️ {authors_short}")

        col1, col2 = st.columns([1, 5])
        with col1:
            pdf_url = paper["link"].replace("/abs/", "/pdf/")
            st.markdown(f"[📄 PDF]({pdf_url})", unsafe_allow_html=True)
            if github_links:
                for repo in github_links[:2]:
                    st.markdown(f"[💻 GitHub]({repo})")

        with col2:
            if summarizer:
                try:
                    text = " ".join(paper["abstract"].split()[:512])
                    summary = summarizer(text, max_length=80, min_length=20, do_sample=False)
                    st.markdown(f"**AI Summary:** {summary[0]['summary_text']}")
                except Exception:
                    with st.expander("Abstract"):
                        st.write(paper["abstract"])
            else:
                with st.expander("Abstract"):
                    st.write(paper["abstract"])

st.success(f"✅ Showing {min(display_count, len(papers))} of {len(papers)} fetched papers.")
