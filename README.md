# PaperHunt — Semantic Ai Research paper discovery tool

PaperHunt is an Ai research exploration tool that helps users find the most relevant academic papers across domains such as Healthcare, Defense, Cybersecurity, Finance, Education, Astrophysics, and more.  Users can search by *technique* (Ml, Dl, Cv, Nlp) and *application domain*, and PaperHunt fetches matching papers directly from arXiv

who can use tihs:
Masters and phd students
This helps users quickly discover whether research similar to their own idea has already been published

Key Features
- Search by AI technique + application domain  
- Semantic ranking using Sentence-Transformers (MiniLM)  
- Optional AI-generated summaries using Hugging Face transformers  
- Direct PDF links + GitHub repository detection  
- Filters for publication date, result count, relevance, and domain  

Tech Stack
- Python
- Streamlit UI
- arXiv API (atom feed) 
- Sentence transformers (semantic embeddings)
- Hugging face ransformer 
- Feedparser
  
 Run Locally
 
pip install -r requirements.txt
streamlit run app.py
