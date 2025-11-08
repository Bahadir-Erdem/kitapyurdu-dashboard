from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from typing import List

from config import SAMPLED_BOOK_DATA_FILE_PATH

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Kitapyurdu Books EDA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

CACHE_TTL = 3600

NUMERIC_COLUMNS = [
    "Purchase Count", "Comment Count", "Kitapyurdu Price",
    "Going to Read Count", "Reading Count", "Read Count",
    "List Price", "Page Count", "estimated_revenue",
    "publish_year", "en", "boy"
]

TURKISH_STOP_WORDS = {
    "acaba", "altmış", "altı", "ama", "bana", "bazı", "belki", "ben", "benden",
    "beni", "benim", "beş", "bin", "bir", "biri", "birkaç", "birkez", "birşey",
    "birşeyi", "biz", "bizden", "bizi", "bizim", "bu", "buna", "bunda", "bundan",
    "bunu", "bunun", "da", "daha", "dahi", "de", "defa", "diye", "doksan", "dokuz",
    "dört", "elli", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç", "iki", "ile",
    "ise", "için", "katrilyon", "kez", "ki", "kim", "kimden", "kime", "kimi", "kırk",
    "milyar", "milyon", "mu", "mü", "mı", "nasıl", "ne", "neden", "nerde", "nerede",
    "nereye", "niye", "niçin", "on", "ona", "ondan", "onlar", "onlardan", "onları",
    "onların", "onu", "otuz", "sanki", "sekiz", "seksen", "sen", "senden", "seni",
    "senin", "siz", "sizden", "sizi", "sizin", "trilyon", "tüm", "ve", "veya", "ya",
    "yani", "yedi", "yetmiş", "yirmi", "yüz", "çok", "çünkü", "üç", "şey", "şeyden",
    "şeyi", "şeyler", "şu", "şuna", "şunda", "şundan", "şunu"
}

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=CACHE_TTL)
def load_data(file_path: Path) -> pd.DataFrame:
    """Load the complete dataset."""
    with st.spinner("📂 Loading dataset..."):
        # Ensure file_path is a string or Path object
        file_path_str = str(file_path)
        df = pd.read_parquet(file_path_str, engine="fastparquet")
        
        # --- FIX for Caching Error ---
        # Convert list columns to tuples for caching compatibility
        # Tuples are hashable, lists are not.
        if "categories" in df.columns:
            # Convert the main 'categories' column to tuples
            df["categories"] = df["categories"].apply(lambda x: tuple(x) if isinstance(x, list) else ())
            
            # Create the 'categories_list' copy *from the new tuple column*
            df["categories_list"] = df["categories"].copy()
        
        return df

# ============================================================================
# DATA FILTERING & PRE-COMPUTATION (FOR OPTIMIZATION)
# ============================================================================

@st.cache_data
def get_exploded_categories(df: pd.DataFrame, exclude_kitap: bool = True) -> pd.DataFrame:
    """(Cached) Explode categories and optionally exclude 'Kitap'."""
    # df.explode works fine with tuples
    exploded = df.explode("categories")
    if exclude_kitap:
        exploded = exploded[exploded["categories"] != "Kitap"]
    return exploded


@st.cache_data
def filter_no_author(df: pd.DataFrame) -> pd.DataFrame:
    """(Cached) Remove 'No Author' entries."""
    return df[df["author"] != "No Author"]


def get_primary_category(categories_tuple) -> str:
    """
    Extract primary category from tuple (previously list).
    --- FIX: Modified to accept tuples from the cached DataFrame ---
    """
    if isinstance(categories_tuple, tuple) and len(categories_tuple) > 0:
        if categories_tuple[0] != "Kitap":
            return categories_tuple[0]
        elif len(categories_tuple) > 1:
            return categories_tuple[1]
    return "Other"

# --- Optimization: Pre-calculate aggregations ---

@st.cache_data
def get_top_categories(_df: pd.DataFrame) -> pd.DataFrame:
    """(Cached) Get top 10 categories."""
    exploded = get_exploded_categories(_df)
    cat_freq = exploded["categories"].value_counts().head(10).reset_index()
    cat_freq.columns = ["Category", "Count"]
    return cat_freq

@st.cache_data
def get_top_categorical(_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """(Cached) Get top 10 for any categorical column."""
    data = filter_no_author(_df) if col == "author" else _df
    top_10 = data[col].value_counts().head(10).reset_index()
    top_10.columns = [col, "Count"]
    return top_10

@st.cache_data
def get_category_numerical_data(_df: pd.DataFrame, num_col: str) -> pd.DataFrame:
    """(Cached) Get data for category vs. numerical boxplots."""
    exploded = get_exploded_categories(_df)
    top_cats = exploded["categories"].value_counts().head(10).index
    filtered = exploded[exploded["categories"].isin(top_cats)]
    return filtered[["categories", num_col]]

@st.cache_data
def get_top_revenue_books(_df: pd.DataFrame) -> pd.DataFrame:
    """(Cached) Get top 10 revenue books with primary category."""
    top_books = _df.nlargest(10, "estimated_revenue").copy()
    top_books["Primary Category"] = top_books["categories"].apply(get_primary_category)
    return top_books

@st.cache_data
def get_top_revenue_authors(_df: pd.DataFrame) -> pd.DataFrame:
    """(Cached) Get top 10 revenue authors with primary category."""
    filtered_df = filter_no_author(_df)
    exploded = get_exploded_categories(filtered_df)
    
    top_authors_index = (
        filtered_df.groupby("author")["estimated_revenue"]
        .sum().nlargest(10).index
    )
    
    author_cats = {}
    for author in top_authors_index:
        author_data = exploded[exploded["author"] == author]
        if len(author_data) > 0 and len(author_data["categories"].value_counts()) > 0:
            author_cats[author] = author_data["categories"].value_counts().index[0]
        else:
            author_cats[author] = "Other"
            
    author_revenue = (
        filtered_df.groupby("author")["estimated_revenue"]
        .sum().nlargest(10).reset_index()
    )
    author_revenue["Primary Category"] = author_revenue["author"].map(author_cats)
    return author_revenue

@st.cache_data
def get_yearly_revenue(_df: pd.DataFrame) -> pd.DataFrame:
    """(Cached) Get total revenue by year."""
    return _df.groupby("publish_year")["estimated_revenue"].sum().reset_index()

# ============================================================================
# TEXT ANALYSIS
# ============================================================================

def tokenize_turkish_text(text_series: pd.Series) -> List[str]:
    """Tokenize Turkish text using regex (handles punctuation)."""
    text = " ".join(text_series.dropna().astype(str)).lower()
    words = re.findall(r'\b\w+\b', text)
    return [w for w in words if w not in TURKISH_STOP_WORDS and len(w) > 2]

@st.cache_data
def get_top_words(text_series: pd.Series, n: int = 50) -> pd.DataFrame:
    """(Cached) Get top N words after filtering."""
    words = tokenize_turkish_text(text_series)
    word_counts = Counter(words)
    return pd.DataFrame(word_counts.most_common(n), columns=["Word", "Count"])

@st.cache_data
def get_correlation_matrix(_df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """(Cached) Calculate correlation matrix."""
    available_cols = [c for c in num_cols if c in _df.columns]
    if available_cols:
        return _df[available_cols].corr()
    return pd.DataFrame()

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_histogram(data: pd.Series, title: str, use_log: bool = False) -> None:
    """Create and display histogram with box plot."""
    fig = px.histogram(
        data.dropna(), x=data.name, marginal="box",
        log_y=use_log, title=title, nbins=50
    )
    fig.update_xaxes(tickformat=",.0f")
    fig.update_yaxes(tickformat=",.0f")
    # --- FIX: Deprecation Warning ---
    st.plotly_chart(fig, width='stretch')


def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, 
                     color: str = None, horizontal: bool = True) -> None:
    """Create and display bar chart."""
    fig = px.bar(
        df, x=x, y=y, title=title,
        color=color or y,
        color_discrete_sequence=px.colors.qualitative.Set3,
        orientation='h' if horizontal else 'v'
    )
    # --- FIX: Deprecation Warning ---
    st.plotly_chart(fig, width='stretch')


def create_scatter(df: pd.DataFrame, x: str, y: str, title: str, 
                   use_log: bool = True) -> None:
    """Create and display scatter plot."""
    fig = px.scatter(
        df, x=x, y=y, title=title,
        log_x=use_log and df[x].min() > 0,
        log_y=use_log and df[y].min() > 0
    )
    fig.update_xaxes(tickformat=",.0f")
    fig.update_yaxes(tickformat=",.0f")
    # --- FIX: Deprecation Warning ---
    st.plotly_chart(fig, width='stretch')


def create_box_plot(df: pd.DataFrame, x: str, y: str, title: str, 
                    log_y: bool = True) -> None:
    """Create and display box plot."""
    fig = px.box(df, x=x, y=y, title=title, log_y=log_y, color=x)
    fig.update_yaxes(tickformat=",.0f")
    # --- FIX: Deprecation Warning ---
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# ANALYSIS SECTIONS
# ============================================================================

def show_dataset_overview(df: pd.DataFrame, show_raw: bool) -> None:
    """Display dataset overview section."""
    with st.expander("📋 Dataset Overview", expanded=show_raw):
        if show_raw:
            # --- FIX: Deprecation Warning ---
            st.dataframe(df.head(100), width='stretch')
        
        # Column information
        unique_counts = [
            df[col].astype(str).nunique() if df[col].dtype == 'object' 
            else df[col].nunique() for col in df.columns
        ]
        
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str)
                .str.replace("int64", "Integer")
                .str.replace("float64", "Float")
                .str.replace("object", "Text/Tuple") # Updated type
                .str.replace("datetime64[ns]", "Date"),
            "Non-Null": df.notnull().sum().values,
            "Unique": unique_counts,
            "Missing %": (df.isnull().sum() / len(df) * 100).round(2).values
        })
        # --- FIX: Deprecation Warning ---
        st.dataframe(info_df, width='stretch')
        
        st.subheader("Summary Statistics")
        # --- FIX: Deprecation Warning ---
        st.dataframe(df.describe(include="all"), width='stretch')


def show_univariate_analysis(df: pd.DataFrame) -> None:
    """Display univariate analysis section."""
    with st.expander("📊 Univariate Analysis", expanded=True):
        # Numerical variables
        st.markdown(f"### Numerical Variables (Full Dataset: {len(df):,} rows)")
        num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
        
        col1, col2 = st.columns(2)
        for idx, col in enumerate(num_cols):
            with col1 if idx % 2 == 0 else col2:
                use_log = col not in ["publish_year"]
                create_histogram(df[col], f"Distribution of {col}", use_log)
        
        # Categorical variables
        st.markdown(f"### Categorical Variables (Top 10, Full Dataset: {len(df):,} rows)")
        
        # Categories
        if "categories" in df.columns:
            # --- OPTIMIZATION: Use cached function ---
            cat_freq = get_top_categories(df)
            create_bar_chart(cat_freq, "Count", "Category", 
                           "Top 10 Categories", "Category")
        
        # Other categorical columns
        categorical_cols = ["author", "Publisher", "Language", "Cover Type"]
        for col in categorical_cols:
            if col in df.columns:
                # --- OPTIMIZATION: Use cached function ---
                top_10 = get_top_categorical(df, col)
                create_bar_chart(top_10, "Count", col, 
                               f"Top 10 {col}", col)


def show_bivariate_analysis(df: pd.DataFrame) -> None:
    """Display bivariate analysis section."""
    with st.expander("🔗 Bivariate Analysis"):
        # Numerical relationships
        st.markdown(f"### Numerical Relationships (Full Dataset: {len(df):,} rows)")
        scatter_pairs = [
            ("Kitapyurdu Price", "Purchase Count"),
            ("Page Count", "Kitapyurdu Price"),
            ("publish_year", "Kitapyurdu Price"),
            ("en", "boy")
        ]
        
        for x, y in scatter_pairs:
            if x in df.columns and y in df.columns:
                # Using full 'df' and updated title
                create_scatter(df, x, y, f"{x} vs {y} (Full Data)") 
        
        # Categorical vs Numerical
        st.markdown(f"### Categorical vs Numerical (Full Dataset: {len(df):,} rows)")
        if "categories" in df.columns:
            for num_col in ["Kitapyurdu Price", "Purchase Count"]:
                if num_col in df.columns:
                    filtered_data = get_category_numerical_data(df, num_col)
                    title = f"{num_col} by Top Categories"
                    create_box_plot(filtered_data, "categories", num_col, title)


def show_financial_analysis(df: pd.DataFrame) -> None:
    """Display financial analysis section."""
    with st.expander("💰 Financial Analysis"):
        # Top books - takes 2/3 of space
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### Top 10 Revenue Books (Full Dataset: {len(df):,} rows)")
            
            # --- OPTIMIZATION: Use cached function ---
            top_books = get_top_revenue_books(df)
            
            # Display table
            display_cols = ["Book Name", "estimated_revenue", "Kitapyurdu Price", 
                          "Purchase Count", "Primary Category"]
            
            display_df = top_books[display_cols].copy()
            display_df["Book Name"] = display_df["Book Name"].astype(str)
            display_df["Primary Category"] = display_df["Primary Category"].astype(str)

            st.dataframe(display_df, width='stretch')
            
            fig = px.bar(
                top_books, x="estimated_revenue", y="Book Name",
                color="Primary Category", title="Top 10 Books by Revenue",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown(f"### Top 10 Authors (Full Dataset: {len(df):,} rows)")
            
            author_revenue = get_top_revenue_authors(df)
            
            fig = px.bar(
                author_revenue, x="estimated_revenue", y="author",
                color="Primary Category", title="Top 10 Authors by Revenue",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, width='stretch')
        
        # Revenue trend
        if "publish_year" in df.columns:
            st.markdown("### Revenue Trend Over Time")
            yearly = get_yearly_revenue(df)
            fig = px.line(yearly, x="publish_year", y="estimated_revenue",
                         title="Revenue Trend Over Years")
            st.plotly_chart(fig, width='stretch')


def show_text_analysis(df: pd.DataFrame, num_cols: List[str]) -> None:
    """Display text analysis section."""
    with st.expander("📝 Text Analysis"):
        st.markdown(f"### Word Frequency Analysis (Full Dataset: {len(df):,} rows)")
        st.info("📌 Using regex tokenization for Turkish text (handles punctuation)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Words in Descriptions")
            if "Description" in df.columns:
                top_words = get_top_words(df["Description"], 30)
                create_bar_chart(top_words, "Count", "Word", 
                               "Most Common Words", "Word")
        
        with col2:
            st.markdown("#### Top Words in Titles")
            if "Book Name" in df.columns:
                top_words = get_top_words(df["Book Name"], 30)
                create_bar_chart(top_words, "Count", "Word", 
                               "Title Words", "Word")
        
        # Correlation heatmap
        st.markdown(f"### Feature Correlations (Full Dataset: {len(df):,} rows)")
        corr = get_correlation_matrix(df, num_cols)
        
        if not corr.empty:
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu", zmin=-1, zmax=1,
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No numeric columns found for correlation heatmap.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.title("📚 Kitapyurdu Books Dataset - Enhanced EDA")
    st.markdown("""
    Comprehensive exploratory data analysis using **full dataset** for all visualizations:
    - **Univariate**: Distribution of individual variables
    - **Bivariate**: Relationships between two variables  
    - **Financial**: Revenue and pricing insights
    - **Text Analysis**: Word frequency and correlations
    
    *All analyses use the complete dataset*
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        show_raw = st.checkbox("Show raw data preview", value=False)
        st.info("💡 All visualizations use the full dataset (no sampling)")
    
    # Load data
    try:
        df = load_data(SAMPLED_BOOK_DATA_FILE_PATH)
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at '{SAMPLED_BOOK_DATA_FILE_PATH}'. Please check the path.")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.exception(e)
        return
    
    # Show all analysis sections with full data
    show_dataset_overview(df, show_raw)
    
    num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    show_univariate_analysis(df)
    show_bivariate_analysis(df)
    show_financial_analysis(df)
    show_text_analysis(df, num_cols)


if __name__ == "__main__":
    main()