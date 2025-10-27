import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import json
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


@functools.lru_cache(maxsize=1)
def load_inventory() -> List[Dict]:
    """Load inventory with caching to avoid repeated file reads."""
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


# Initialize data structures with memory optimization
if 'inventory_lookup' not in st.session_state:
    st.session_state.inventory_lookup = build_inventory_lookup()

# Initialize Gemini model with lazy loading
if 'model' not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-pro")

# Use session state for better memory management
inventory_lookup = st.session_state.inventory_lookup
model = st.session_state.model


# Display
st.markdown('<div class="main-title">📚 Pustakwale</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your smart book buddy for discovery & recommendations</div>', unsafe_allow_html=True)
st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

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
# Streamlit UI Enhancement
st.set_page_config(page_title="Pustakwaale", page_icon="📚", layout="wide")


last_book = st.text_input("Enter the last book you read:")
fav_author = st.text_input("👤 (Optional) Your favorite author")
st.markdown("### 🎯 Filter Your Preferences")
col1, col2, col3 = st.columns(3)

with col1:
    selected_language = st.selectbox(
        "📘 Preferred Language",
        ["Any", "English", "Hindi", "Marathi"]
    )

with col2:
    selected_genre = st.selectbox(
        "🎭 Preferred Genre",
        ["Any", "Fiction", "Non-Fiction", "Science", "Biography", "Children", "Fantasy", "Romance", "Mystery"]
    )

with col3:
    selected_age_group = st.selectbox(
        "👶 Target Age Group",
        ["Any", "Kids (5–12)", "Teens (13–19)", "Adults (20+)", "All Ages"]
    )

# Cache for book covers to avoid repeated API calls
@functools.lru_cache(maxsize=1000)
def get_book_cover(title: str, author: str) -> Optional[str]:
    """Fetch book cover with caching and proper error handling."""
    query = f"{title} {author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1&zoom=2"
    
    try:
        response = requests.get(url, timeout=5)  # Add timeout
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

    # If not found, try fuzzy matching on titles
    # Use a more efficient approach by pre-computing titles
    if 'all_titles' not in st.session_state:
        st.session_state.all_titles = [t for (t, a) in inventory_lookup.keys()]
    
    close_titles = get_close_matches(title, st.session_state.all_titles, n=1, cutoff=cutoff)

    if close_titles:
        best_title = close_titles[0]
        # Find matching author with best title - more efficient search
        for (t, a) in inventory_lookup.keys():
            if t == best_title:
                return inventory_lookup[(t, a)]

    return "Not in Inventory"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def recommend_books(last_book: Optional[str] = None, fav_author: Optional[str] = None, 
                   genre: str = "Any", language: str = "Any", age_group: str = "Any") -> str:
    """Generate book recommendations with caching."""
    prompt = f"""
        The user last read the book: "{last_book}".
        Their favorite author is: "{fav_author if fav_author else 'Not specified'}". if author is specified include all the books by that author from the inventory.

        User preferences:
        - Preferred language: "{language}"
        - Preferred genre: "{genre}"
        - Target age group: "{age_group}"

        Your task:
        1. Recommend 5 books (fiction or non-fiction) that best fit the user's preferences above.
        2. Give priority to books that are available in the inventory.
        3. If a favorite author is provided, include 2–3 notable works by that author (if available).
        4. Additionally, recommend 2–3 similar authors and list 2–3 of their popular books.
        5. Avoid repetition between sections.

        Formatting:
        Provide the final answer as a clean numbered list in this format:
        1. Title - Author
        2. Title - Author
        3. Title - Author
        (No explanations or categories.)
        """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return ""



def fetch_single_book_data(book_title: str, author: str) -> Dict:
    """Fetch data for a single book."""
    availability = find_in_inventory(book_title, author)
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
            author = parts[1].strip()
            
            # Skip if title or author is empty
            if not book_title or not author:
                continue
                
            book_tasks.append((book_title, author))
    
    # Process books in parallel for faster execution
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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



# Custom CSS for header styling
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    color: Bright Yellow;
    margin-bottom: 8px;
    text-shadow: 1px 2px 4px rgba(0,0,0,0.1);
}

.subtitle {
    font-size: 18px;
    font-weight: 400;
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}

.header-bar {
    height: 6px;
    width: 80px;
    margin: 0 auto 20px auto;
    border-radius: 4px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
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
    border-radius: 10px;
    margin: 14px 0;
    background-color: black;
    padding: 8px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
}

.book-title {
    font-size: 17px;
    font-weight: 700;
    margin: 6px 0 2px 0;
    color: #333;
    text-overflow: ellipsis;
    overflow: hidden;
    max-width: 90%;
    white-space: nowrap;
}

.book-author {
    font-size: 14px;
    color: #777;
    margin-bottom: 12px;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

.badge-available {
    background: linear-gradient(135deg, #28a745, #4cd964);
    color: white;
}
.badge-notavailable {
    background: linear-gradient(135deg, #ff4b2b, #ff416c);
    color: white;
}
.badge-warning {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #333;
}
</style>
""", unsafe_allow_html=True)





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

    if books:
        st.subheader("Recommended Books:")
        cols = st.columns(4)  # 4 books per row

        for i, book in enumerate(books):
            with cols[i % 4]:
                st.markdown('<div class="book-card">', unsafe_allow_html=True)

                # Book cover
                cover_url = book.get("cover", "https://via.placeholder.com/128x200?text=No+Image")
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
                    st.markdown('<span class="badge badge-warning">Not in Inventory ⚠️</span>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No matching recommendations found. Try different filters!")

# Add cache management buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Clear Cache"):
        st.session_state.recommendations_cache = {}
        st.session_state.inventory_lookup = build_inventory_lookup()
        st.success("Cache cleared!")
        

        
with col3:
    if st.button("Clear All Caches"):
        get_book_cover.cache_clear()
        find_in_inventory.cache_clear()
        load_inventory.cache_clear()
        build_inventory_lookup.cache_clear()
        st.session_state.recommendations_cache = {}
        st.session_state.inventory_lookup = build_inventory_lookup()
        cleanup_memory()  # Clean up memory
        st.success("All caches cleared and memory optimized!")

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
