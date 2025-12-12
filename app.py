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
st.set_page_config(page_title="Semantic arXiv Search", layout="wide")

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if not EMBEDDING_AVAILABLE:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")
credits
@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    if not SUMMARIZER_AVAILABLE:
        return None
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        try:
            return pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception:
            return None
st.sidebar.header("Search Filters")
domains = [
    "All", "Healthcare", "Defense", "Finance", "Education", "Robotics", "Energy",
    "Transportation", "Agriculture", "Space", "Climate Science", "Cybersecurity",
    "Quantum Computing", "Blockchain", "Social Sciences", "Astrophysics"
]
search_term = st.sidebar.text_input("Enter technique (e.g., machine learning, deep learning, CV)", "")
selected_domain = st.sidebar.selectbox("Select Application Domain", domains)
max_results = st.sidebar.number_input("Max papers to fetch (from arXiv)", min_value=1, max_value=50, value=25)
display_count = st.sidebar.number_input("How many top semantically relevant to display", min_value=1, max_value=25, value=10)
use_semantic = st.sidebar.checkbox("Use semantic (embedding) ranking", value=True)
use_summarizer = st.sidebar.checkbox("Show AI summary of abstracts", value=False)
days_back = st.sidebar.slider("Published within (days)", 1, 3650, 365)

st.sidebar.markdown("---")
st.sidebar.markdown("Model availability:")
st.sidebar.write(f"Embeddings loaded: {EMBEDDING_AVAILABLE}")
st.sidebar.write(f"Summarizer available: {SUMMARIZER_AVAILABLE}")
if not search_term:
    st.info("Enter a search term (technique) like 'machine learning' and pick a domain, then press Search.")
    st.stop()
tech = search_term.strip()
if selected_domain == "All":
    search_query_str = f'all:"{tech}"'
else:
    search_query_str = f'all:"{tech}" AND all:"{selected_domain}"'
encoded_query = urllib.parse.quote(search_query_str)
date_limit = datetime.now() - timedelta(days=days_back)
base_url = "https://export.arxiv.org/api/query?"
api_query = (
    f"search_query={encoded_query}&start=0&max_results={max_results}"
    "&sortBy=submittedDate&sortOrder=descending"
)
feed = feedparser.parse(base_url + api_query)
feed = feedparser.parse(base_url + api_query)
entries = []
for entry in feed.entries:
    try:
        published = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        try:
            published = datetime.strptime(entry.updated, '%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            continue
    if published < date_limit:
        continue
    abstract = getattr(entry, "summary", "")
    authors = ", ".join([a.name for a in getattr(entry, "authors", [])]) if getattr(entry, "authors", None) else ""
    entries.append({
        "id": entry.get("id", entry.get("id", "")),
        "title": entry.title,
        "link": entry.link,
        "abstract": abstract,
        "published": published,
        "authors": authors
    })
if not entries:
    st.warning("No papers found for that combination. Try increasing max_results or days range, or change terms.")
    st.stop()
embedding_model = None
paper_embeddings = None
query_embedding = None
if use_semantic and EMBEDDING_AVAILABLE:
    with st.spinner("Loading embedding model..."):
        embedding_model = load_embedding_model()
    docs = [(e["title"] + ". " + re.sub(r'\s+', ' ', e["abstract"])) for e in entries]
    with st.spinner("Computing embeddings for fetched papers..."):
        paper_embeddings = embedding_model.encode(docs, convert_to_tensor=True, show_progress_bar=False)
        query_embedding = embedding_model.encode(final_query, convert_to_tensor=True, show_progress_bar=False)
    similarities = util.cos_sim(query_embedding, paper_embeddings)[0].cpu().numpy()
    for idx, score in enumerate(similarities):
        entries[idx]["score"] = float(score)
    entries = sorted(entries, key=lambda x: x.get("score", 0), reverse=True)
else:
    for e in entries:
        e["score"] = None
summarizer = None
if use_summarizer and SUMMARIZER_AVAILABLE:
    with st.spinner("Loading summarizer (this can take time)..."):
        summarizer = load_summarizer_model()
        if summarizer is None:
            st.warning("Summarizer could not be loaded. Summaries will be skipped.")
            use_summarizer = False
shown = 0
for i, e in enumerate(entries):
    if shown >= display_count:
        break
    title_md = f"### {shown+1}. [{e['title'].strip()}]({e['link']})"
    st.markdown(title_md)
    pdf_link = e["link"].replace("abs", "pdf")
    st.markdown(f"[Open PDF]({pdf_link})")
    st.write(f"**Authors:** {e['authors']}")
    st.caption(f"Published: {e['published'].strftime('%Y-%m-%d')}")
    if e.get("score") is not None:
        st.write(f"**Semantic score:** {e['score']:.4f}")
    if use_summarizer and summarizer:
        try:
            text_to_sum = e["abstract"]
            if len(text_to_sum.split()) > 512:
                text_to_sum = " ".join(text_to_sum.split()[:512])
            summary = summarizer(text_to_sum, max_length=80, min_length=20, do_sample=False)
            summary_text = summary[0]["summary_text"]
            st.markdown(f"**AI Summary:** {summary_text}")
        except Exception:
            st.markdown(f"**Abstract:** {e['abstract'][:1000]}...")
    else:
        with st.expander("Abstract"):
            st.write(e["abstract"])

    if "github.com" in e["abstract"].lower():
        matches = re.findall(r"(https?://github\.com[^\s\)\]\.,<>]+)", e["abstract"], flags=re.IGNORECASE)
        if matches:
            for repo in matches[:3]:
                st.markdown(f"[💻 GitHub Repo]({repo})")

    st.markdown("---")
    shown += 1
st.success(f"Displayed {shown} papers (from {len(entries)} fetched).")