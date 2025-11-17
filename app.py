import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import json
try:
    import orjson 
except Exception:
    orjson = None
import os
import requests
from difflib import get_close_matches
import functools
from typing import Dict, List, Optional, Tuple
import time
import concurrent.futures
from threading import Lock


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit UI Enhancement - Must be first Streamlit command
st.set_page_config(page_title="Pustakwaale", page_icon="📚", layout="wide")

@functools.lru_cache(maxsize=1)
def load_inventory() -> List[Dict]:
    """Load inventory with caching to avoid repeated file reads.

    Uses orjson when available for faster parsing.
    """
    if orjson is not None:
        with open("books.json", "rb") as f:
            return orjson.loads(f.read())
    with open("books.json", "r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def build_inventory_lookup() -> Dict[Tuple[str, str], str]:
    """Build fast lookup dict with caching."""
    inventory = load_inventory()
    return {
        (book["title"].lower().strip(), book["author"].lower().strip()): book["available"]
        for book in inventory
    }

@functools.lru_cache(maxsize=1)
def build_title_to_authors() -> Dict[str, List[Tuple[str, str]]]:
    """Map normalized title -> list of (author, availability) for O(1) access by title."""
    inventory = load_inventory()
    title_to_authors: Dict[str, List[Tuple[str, str]]] = {}
    for book in inventory:
        t = book["title"].lower().strip()
        a = book["author"].lower().strip()
        title_to_authors.setdefault(t, []).append((a, book["available"]))
    return title_to_authors


def sanitize_author_text(author_text: str) -> str:
    """Strip explanations/extra words from author text."""
    text = author_text.strip()
    if not text:
        return ""

    text_lower = text.lower()
    if text_lower.startswith("by "):
        text = text[3:].strip()

    # Remove availability or reason keywords
    for marker in ["available"]:
        idx = text.lower().find(marker)
        if idx != -1:
            text = text[:idx].strip()

    # Remove trailing context after separators
    for sep in [" – ", " — ", " - ", " | ", " • ", ": "]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()

    if "(" in text:
        text = text.split("(", 1)[0].strip()

    return text.rstrip(".").strip()


@functools.lru_cache(maxsize=512)
def get_default_author_for_title(book_title: str) -> Optional[str]:
    """Get canonical author for a given title directly from inventory."""
    normalized = book_title.lower().strip()
    for book in load_inventory():
        if book["title"].lower().strip() == normalized:
            return book["author"]
    return None


# Initialize data structures with memory optimization
if 'inventory_lookup' not in st.session_state:
    st.session_state.inventory_lookup = build_inventory_lookup()
if 'title_to_authors' not in st.session_state:
    st.session_state.title_to_authors = build_title_to_authors()

# Initialize Gemini model with lazy loading
if 'model' not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.0-flash")
    generationConfig = {
                "temperature": 0.3   
            }
# Use session state for better memory management
inventory_lookup = st.session_state.inventory_lookup
title_to_authors = st.session_state.title_to_authors
model = st.session_state.model


# Display


# Memory optimization: Clear unused variables periodically
def cleanup_memory():
    """Clean up unused variables and clear caches if memory usage is high."""
    import gc
    gc.collect()
    
    # Clear old cache entries if cache is getting too large
    if hasattr(get_book_cover, 'cache_info'):
        cache_info = get_book_cover.cache_info()
        if cache_info.currsize > 800:  # Clear if more than 800 entries
            get_book_cover.cache_clear()








# Cache for book covers to avoid repeated API calls
@st.cache_resource(show_spinner=False)
def get_requests_session() -> requests.Session:
    """Shared requests session with retries and keep-alive for faster HTTP."""
    session = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    except Exception:
        pass
    session.headers.update({"Connection": "keep-alive"})
    return session

@functools.lru_cache(maxsize=20000)
def get_book_cover(title: str, author: str) -> Optional[str]:
    """Fetch book cover with caching and proper error handling."""
    query = f"{title} {author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1&zoom=2"
    
    try:
        session = get_requests_session()
        response = session.get(url, timeout=4)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        if "items" not in data or not data["items"]:
            return None
            
        image = data["items"][0]["volumeInfo"].get("imageLinks", {}).get("thumbnail")
        if image:
            # Clean up the image URL
            image = image.replace("http://", "https://").replace("&edge=curl", "")
        return image
    except (requests.RequestException, KeyError, IndexError) as e:
        # Log error in production (for now, silently return None)
        return None

@functools.lru_cache(maxsize=1000)
def find_in_inventory(title: str, author: str, cutoff: float = 0.8) -> str:
    """
    Optimized inventory search with caching and improved fuzzy matching.
    cutoff: similarity threshold (0 to 1)
    """
    title = title.lower().strip()
    author = author.lower().strip()

    # First check exact match
    if (title, author) in inventory_lookup:
        return inventory_lookup[(title, author)]

    # If not found, try fuzzy matching on titles using precomputed title map
    if 'all_titles' not in st.session_state:
        st.session_state.all_titles = list(title_to_authors.keys())

    close_titles = get_close_matches(title, st.session_state.all_titles, n=1, cutoff=cutoff)

    if close_titles:
        best_title = close_titles[0]
        # Prefer same author if present; else return first available mapping
        for a_norm, availability in title_to_authors.get(best_title, []):
            if a_norm == author:
                return availability
        if title_to_authors.get(best_title):
            return title_to_authors[best_title][0][1]

    return "Not in Inventory"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def recommend_books(last_book: Optional[str] = None, fav_author: Optional[str] = None, 
                   genre: str = "Any", language: str = "Any", age_group: str = "Any") -> str:
    """Generate book recommendations with caching."""
    prompt = f"""
            You are an expert book curator with deep knowledge of literature, genres, and reader psychology.

            The user’s profile and context are as follows:
            - Last book read: "{last_book}"
            - Favorite author: "{fav_author if fav_author else 'Not specified'}"
            - Preferred language: "{language}"
            - Preferred genre: "{genre}"
            - Target age group: "{age_group}"

            You have access to an inventory of books. Always prioritize books available in this inventory.

            ---

            ### Your Objectives:

            1. **Personalized relevance:**
            - Suggest 5 books that the user is *most likely to enjoy next*, based on the tone, themes, writing style, and 
            genre of the last book they read.
            - If the last book is fiction, maintain a similar emotional or narrative depth.
            - If it’s non-fiction, find books with comparable subject matter or intellectual appeal.

            2. **Inventory priority:**
            - At least 3 of the 5 main recommendations must exist in the inventory.
            - If any are unavailable, note them internally (but don’t display replacements from outside the inventory unless necessary).

            3. **Favorite author handling:**
            - If a favorite author is specified, include 2–3 of their notable books from the inventory (if available).
            - Prefer a mix of the author’s most iconic works and lesser-known but well-reviewed titles.

            4. **Author similarity:**
            - Suggest 2–3 *similar authors* whose writing style or themes align with the favorite author or last read book.
            - For each similar author, list 2–3 of their popular works (preferably in the inventory).

            5. **Overall tone:**
            - The final list should feel tailored, literary, and natural — not generic or random.ations or categories.)'

            6.Sequence of the books:
            - Always show availabel books first and then the not available books.
                   
                   You have to return the recommendations in the following format:

                   <book_title> - <author> -<Availability>
                   <book_title> - <author> -<Availability>
                   <book_title> - <author> -<Availability>
                Dont include symbols in the name of the book or author.
                   *Dont include any other text or formatting.*
                    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return ""



def fetch_single_book_data(book_title: str, author: str) -> Dict:
    """Fetch data for a single book."""
    lookup_author = author
    if not author or author.lower() == "unknown author":
        fallback_author = get_default_author_for_title(book_title)
        if fallback_author:
            lookup_author = fallback_author
            author = fallback_author

    availability = find_in_inventory(book_title, lookup_author or author)
    cover = get_book_cover(book_title, author)
    return {
        "title": book_title,
        "author": author,
        "availability": availability,
        "cover": cover
    }

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def check_inventory_with_images(suggestions: str) -> List[Dict]:
    """Process recommendations and fetch book data with parallel processing."""
    book_tasks = []
    
    # Parse all books first
    for line in suggestions.split("\n"):
        if "-" in line:
            parts = line.split("-", 1)
            book_title = parts[0].strip("1234567890. ").strip()
            author_hint = sanitize_author_text(parts[1])
            if not author_hint:
                default_author = get_default_author_for_title(book_title)
                author_hint = default_author if default_author else ""
            author = author_hint or "Unknown Author"
            
            # Skip if title or author is empty
            if not book_title or not author:
                continue
                
            book_tasks.append((book_title, author))
    
    # Process books in parallel for faster execution
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_book = {
            executor.submit(fetch_single_book_data, title, author): (title, author)
            for title, author in book_tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_book):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Log error but continue processing other books
                title, author = future_to_book[future]
                results.append({
                    "title": title,
                    "author": author,
                    "availability": "Error",
                    "cover": None
                })
    
    return results

@st.cache_data(ttl=86400, max_entries=2000)  # Cache for 24 hours, up to 2000 entries
def get_book_cover_cached(title: str, author: str) -> Optional[str]:
    """Fetch book cover with persistent Streamlit caching."""
    # Check session state cache first (fastest)
    cache_key = f"{title.lower().strip()}|{author.lower().strip()}"
    if cache_key in st.session_state.cover_cache:
        return st.session_state.cover_cache[cache_key]
    
    query = f"{title} {author}"
    google_url = "https://www.googleapis.com/books/v1/volumes"
    
    session = get_requests_session()

    # Try Google Books API first
    try:
        response = session.get(
            google_url,
            params={
                "q": query,
                "maxResults": 1,
                "printType": "books",
                "fields": "items(volumeInfo/imageLinks)",
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("items"):
            volume_info = data["items"][0].get("volumeInfo", {})
            image_links = volume_info.get("imageLinks", {})
            image = (
                image_links.get("extraLarge")
                or image_links.get("large")
                or image_links.get("medium")
                or image_links.get("small")
                or image_links.get("thumbnail")
                or image_links.get("smallThumbnail")
            )
            if image:
                result = (
                    image.replace("http://", "https://")
                    .replace("&edge=curl", "")
                    .replace("&zoom=1", "")
                )
                # Store in session state cache
                st.session_state.cover_cache[cache_key] = result
                return result
    except (requests.RequestException, ValueError) as e:
        # Log but don't retry immediately
        pass

    # Fallback to Open Library covers only if Google Books fails
    try:
        response = session.get(
            "https://openlibrary.org/search.json",
            params={"title": title, "author": author, "limit": 1},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        docs = data.get("docs", [])
        if docs:
            doc = docs[0]
            if cover_id := doc.get("cover_i"):
                result = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
                st.session_state.cover_cache[cache_key] = result
                return result
            if "isbn" in doc and doc["isbn"]:
                result = f"https://covers.openlibrary.org/b/isbn/{doc['isbn'][0]}-L.jpg"
                st.session_state.cover_cache[cache_key] = result
                return result
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass

    # Cache None to avoid repeated failed lookups
    st.session_state.cover_cache[cache_key] = None
    return None

# Custom CSS for header styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

/* Set entire page background to white */
.stApp {
    background-color: #ffffff;
    color: #000000;
    font-family: 'Roboto', sans-serif;
}

body {
    font-family: 'Roboto', sans-serif;
}

/* Main container styling */
.main-container {
    padding: 20px;
    background-color: #ffffff;
    color: #000000;
}

/* Logo styling */
.logo-container {
    text-align: left;
    margin-bottom: 30px;
    background-color: #ffffff;
    color: #000000;
}

/* Left column styling - white background */
.left-column {
    padding: 20px;
    background-color: #ffffff;
    border-radius: 15px;
    min-height: 600px;
    color: #000000;
}

.left-column-content {
    background-color: #ffffff;
    color: #000000;
}

/* Right column styling */
.right-column {
    padding: 20px;
    background-color: #ffffff;
    color: #000000;
}

/* GIF container */
.gif-container {
    margin-top: 30px;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    background-color: #ffffff;
}

/* Ensure all sections have white background and black text */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #000000;
}

[data-testid="stHeader"] {
    background-color: #ffffff;
    color: #000000;
}

.block-container {
    background-color: #ffffff;
    color: #000000;
}

/* Change all text colors to black */
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #000000 !important;
}

/* Streamlit specific text elements */
.stMarkdown, .stText, .stTextInput > label, .stSelectbox > label {
    color: #000000 !important;
}

/* Input fields - white background with black text and green border */
input, textarea, select {
    color: #000000 !important;
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 2px solid #22c55e !important;
    border-radius: 4px !important;
}

/* Streamlit input fields */
[data-baseweb="input"] input,
[data-baseweb="input"] textarea,
[data-baseweb="select"] select,
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stTextArea > div > div > textarea {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #22c55e !important;
    border-radius: 4px !important;
}

/* BaseWeb components (used by Streamlit) */
input[type="text"],
input[type="email"],
input[type="number"],
textarea,
select {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 2px solid #22c55e !important;
    border-radius: 4px !important;
    color: #000000 !important;
}

/* Streamlit selectbox dropdown */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
}

/* Dropdown options */
[role="listbox"],
[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Dropdown option text */
[role="option"] > div,
[role="option"] > span {
    color: #000000 !important;
}

/* Select dropdown text */
select option {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Input containers */
.stTextInput > div,
.stSelectbox > div,
.stTextArea > div {
    background-color: #ffffff !important;
}

/* BaseWeb input wrapper */
[data-baseweb="base-input"],
[data-baseweb="select"] {
    background-color: #ffffff !important;
}

/* BaseWeb input inner elements */
[data-baseweb="base-input"] input,
[data-baseweb="base-input"] textarea {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #22c55e !important;
    border-radius: 4px !important;
}

/* Streamlit widget containers */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
}

/* Narrower input containers for cleaner layout */
.stTextInput > div,
.stSelectbox > div,
.stTextArea > div {
    max-width: 420px;
    width: 100%;
}

/* Remove border from internal searchable input inside selectbox */
.stSelectbox [data-baseweb="select"] input {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Ensure all input-like elements are white with black text and green border */
.stTextInput input,
.stSelectbox select,
.stTextArea textarea {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #22c55e !important;
    border-radius: 4px !important;
}

/* BaseWeb input focus state - keep green border */
[data-baseweb="input"] input:focus,
[data-baseweb="input"] textarea:focus,
[data-baseweb="select"] select:focus,
.stTextInput input:focus,
.stSelectbox select:focus,
.stTextArea textarea:focus {
    border: 2px solid #22c55e !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.1) !important;
}

/* Enhanced dropdown arrow styling */
[data-baseweb="select"] svg {
    color: #0f172a !important;
    width: 1.15rem !important;
    height: 1.15rem !important;
    transform: translateY(-1px);
}

/* BaseWeb select border */
[data-baseweb="select"] {
    border: 2px solid #22c55e !important;
    border-radius: 4px !important;
}

/* Streamlit selectbox selected value text */
[data-baseweb="select"] [aria-selected="true"],
[data-baseweb="select"] > div > div {
    color: #000000 !important;
}

/* Selectbox value display */
.stSelectbox [data-baseweb="select"] > div {
    color: #000000 !important;
}

/* All input placeholder text - lighter gray */
input::placeholder,
textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}

/* Button backgrounds - keep white or transparent */
.stButton > button {
    background-color: #22c55e !important;
    color: #ffffff !important;
}

.stButton > button:hover {
    background-color: #16a34a !important;
    color: #ffffff !important;
}

/* Headings */
h1, h2, h3 {
    color: #000000 !important;
}

/* Paragraph and general text */
p, div, span {
    color: #000000 !important;
}

/* Streamlit widget labels and text - black for labels, white for buttons */
[data-baseweb="input"] label,
[data-baseweb="select"] label,
.stSelectbox label,
.stTextInput label {
    color: #000000 !important;
}

/* All markdown text */
.stMarkdown p,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown div {
    color: #000000 !important;
}

/* Info and warning messages */
.stInfo,
.stWarning,
.stSuccess {
    color: #000000 !important;
}

.stInfo p,
.stWarning p,
.stSuccess p,
.stInfo div,
.stWarning div,
.stSuccess div {
    color: #000000 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] {
    color: #000000 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


# Sleek & Modern Custom CSS for Book Cards
st.markdown("""
<style>





.book-card {
    background: #0f172a;
    border-radius: 20px;
    padding: 24px 22px 28px;
    margin: 16px 8px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 430px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.55);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.book-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 28px 50px rgba(15, 23, 42, 0.75);
}

.book-cover {
    width: 100%;
    max-width: 210px;
    height: 280px;
    object-fit: cover;
    border-radius: 14px;
    margin: 12px 0 20px;
    background-color: #050505;
    padding: 10px;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
}

.book-title {
    font-size: 18px;
    font-weight: 700;
    margin: 4px 0 6px;
    color: #f8fafc;
    text-overflow: ellipsis;
    overflow: hidden;
    max-width: 95%;
    white-space: nowrap;
}

.book-author {
    font-size: 15px;
    color: #cbd5f5;
    margin-bottom: 18px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.4px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
    text-transform: uppercase;
}

.badge-available {
    background: linear-gradient(135deg, #33d17a, #70ffb3);
    color: #08220f;
}
.badge-notavailable {
    background: linear-gradient(135deg, #ff6b6b, #f43f5e);
    color: white;
}
.badge-warning {
    background: linear-gradient(135deg, #fed049, #fcb045);
    color: #3d2c00;
}
</style>
""", unsafe_allow_html=True)

# Sleek & Modern Custom CSS for Book Cards
st.markdown("""
<style>





.book-cover {
    width: 100%;
    max-width: 180px;
    height: 260px;
    object-fit: contain;
}

.book-title {
    font-size: 17px;
    font-weight: 700;
    margin: 6px 0 2px 0;
    color: #000000;
    text-overflow: ellipsis;
    overflow: hidden;
    max-width: 90%;
    white-space: nowrap;
}

.book-author {
    font-size: 14px;
    color: #000000;
    margin-bottom: 12px;
}

.book-availability {
    font-size: 13px;
    font-weight: 500;
    margin-top: 6px;
}

.badge-available {
    background: linear-gradient(135deg, #33d17a, #70ffb3);
    color: #08220f;
}
.badge-notavailable {
    background: linear-gradient(135deg, #ff6b6b, #f43f5e);
    color: white;
}


</style>
""", unsafe_allow_html=True)


left_col, right_col = st.columns([1, 1.5], gap="large")

# Left Column: Logo and GIF
with left_col:
    st.markdown('<div class="left-column-content">', unsafe_allow_html=True)
    
    # Logo in top-left corner
    logo_path = "logo.png"  # Update this path to your logo file
    gif_path = "project.gif"  # Update this path to your GIF file
    
    # Check if logo exists, otherwise show placeholder
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        # Placeholder logo - you can replace this with your actual logo
        st.markdown("""
        <div style="text-align: left; margin-bottom: 30px;">
            <h1 style="font-size: 32px; font-weight: 800; color: #000000; margin: 0;">
                📚 Pustakwale
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project GIF
    if os.path.exists(gif_path):
        st.markdown('<div class="gif-container">', unsafe_allow_html=True)
        st.image(gif_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder - user should add their GIF
        st.info("📁 Please add your project GIF as 'project.gif' in the project folder")
    
    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown("## 📚 Smart Genie")
    st.markdown("### Discover & Get Personalized Recommendations")
    st.markdown("<br>", unsafe_allow_html=True)
    
    last_book = st.text_input("Enter the last book you read:", key="last_book")
    fav_author = st.text_input("👤 (Optional) Your favorite author", key="fav_author")
    
    st.markdown("### 🎯 Filter Your Preferences")
    selected_language = st.selectbox(
        "📘 Preferred Language",
        ["Any", "English", "Hindi", "Marathi"],
        key="language"
    )
    
    selected_genre = st.selectbox(
        "🎭 Preferred Genre",
        ["Any", "Fiction", "Non-Fiction", "Science", "Biography", "Children", "Fantasy", "Romance", "Mystery"],
        key="genre"
    )
    
    selected_age_group = st.selectbox(
        "👶 Target Age Group",
        ["Any", "Kids (5–12)", "Teens (13–19)", "Adults (20+)", "All Ages"],
        key="age_group"
    )
    
    # Keep the latest fetched books in session so UI can render outside button callback
    books: List[Dict] = st.session_state.get("current_books", [])

    if st.button("Get Recommendations"):
        # Create a cache key for this request
        cache_key = f"{last_book or 'none'}_{fav_author or 'none'}_{selected_genre}_{selected_language}_{selected_age_group}"
        
        # Check if we have cached results
        if 'recommendations_cache' not in st.session_state:
            st.session_state.recommendations_cache = {}
        
        if cache_key in st.session_state.recommendations_cache:
            books = st.session_state.recommendations_cache[cache_key]
            st.info("Showing cached recommendations. Click 'Clear Cache' to refresh.")
        else:
            with st.spinner("Fetching recommendations..."):
                # Pass filters to the recommendation function
                suggestions = recommend_books(last_book or None, fav_author, selected_genre, selected_language, selected_age_group)
                books = check_inventory_with_images(suggestions)
                # Cache the results
                st.session_state.recommendations_cache[cache_key] = books
        st.session_state.current_books = books

        # Cache management buttons
    st.markdown("<br>", unsafe_allow_html=True)
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        if st.button("🗑️ Clear Cache", use_container_width=True, key="clear_cache"):
            st.session_state.recommendations_cache = {}
            if 'current_books' in st.session_state:
                del st.session_state.current_books
            st.session_state.inventory_lookup = build_inventory_lookup()
            st.success("Cache cleared!")
            st.rerun()
    
    with cache_col2:
        if st.button("🔄 Clear All Caches", use_container_width=True, key="clear_all"):
            if 'current_books' in st.session_state:
                del st.session_state.current_books
            # Clear cover caches
            try:
                get_book_cover_cached.clear()
            except AttributeError:
                # If clear() method doesn't exist, the cache will expire naturally
                pass
            if 'cover_cache' in st.session_state:
                st.session_state.cover_cache = {}
            load_inventory.cache_clear()
            build_inventory_lookup.cache_clear()
            st.session_state.recommendations_cache = {}
            st.session_state.inventory_lookup = build_inventory_lookup()
            cleanup_memory()  # Clean up memory
            st.success("All caches cleared and memory optimized!")
            st.rerun()

if books:
        st.subheader("Recommended Books:")
        cols = st.columns(4)  # 4 books per row

        for i, book in enumerate(books):
            # Skip entries without a proper title or author to avoid blank cards
            title_val = (book.get("title") or "").strip()
            author_val = (book.get("author") or "").strip()
            cover_val = book.get("cover")
            if not title_val or not author_val:
                continue

            with cols[i % 4]:
                

                # Book cover (fallback placeholder if no cover URL)
                cover_url = cover_val or "https://via.placeholder.com/128x200?text=No+Image"
                st.markdown(f'<img src="{cover_url}" class="book-cover" />', unsafe_allow_html=True)

                # Title & Author
                st.markdown(f'<div class="book-title">{book.get("title", "Unknown Title")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="book-author">by {book.get("author", "Unknown Author")}</div>', unsafe_allow_html=True)

                # Availability badge
                availability = book.get("availability", "").lower()
                if availability == "yes":
                    st.markdown('<span class="badge badge-available">Available ✅</span>', unsafe_allow_html=True)
                elif availability == "no":
                    st.markdown('<span class="badge badge-notavailable">Not Available ❌</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-warning">Not in Inventory⚠️</span>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("No matching recommendations found. Try different filters!")
    
    
    





    

# Add cache management buttons


# Add memory usage display (optional, requires psutil)
if st.sidebar.checkbox("Show Memory Info"):
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        st.sidebar.metric("Memory Usage", f"{memory_mb:.1f} MB")
    except ImportError:
        st.sidebar.warning("Install psutil to see memory usage: pip install psutil")
    
    # Show cache statistics
    if hasattr(get_book_cover, 'cache_info'):
        cache_info = get_book_cover.cache_info()
        st.sidebar.write(f"Cover Cache: {cache_info.currsize} entries")
    
    if hasattr(find_in_inventory, 'cache_info'):
        cache_info = find_in_inventory.cache_info()
        st.sidebar.write(f"Inventory Cache: {cache_info.currsize} entries")
