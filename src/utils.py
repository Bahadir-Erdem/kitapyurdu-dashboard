import pandas as pd
from config import SAMPLE_SIZE, BOOK_DATA_FILE_PATH, SAMPLED_BOOK_DATA_FILE_PATH

COLUMN_RENAME = {
    "Liste Fiyatı:": "List Price",
    "Yayın Tarihi:": "Publication Date",
    "Sayfa Sayısı:": "Page Count",
    "Cilt Tipi:": "Cover Type",
    "Kağıt Cinsi:": "Paper Type",
    "Dil:": "Language",
    "kitapyurdu_price": "Kitapyurdu Price",
    "purchase_count": "Purchase Count",
    "comment_count": "Comment Count",
    "favorite_count": "Favorite Count",
    "going_to_read_count": "Going to Read Count",
    "reading_count": "Reading Count",
    "read_count": "Read Count",
    "book_name": "Book Name",
    "book_description_text": "Description",
    "producers": "Publisher",
    "Boyut:": "Dimensions",
}


def drop_high_null_columns(df: pd.DataFrame, threshold: int = 80) -> pd.DataFrame:
    """Drop columns with null percentage above threshold."""
    null_pct = (df.isna().sum() / len(df) * 100).round()
    cols_to_drop = null_pct[null_pct > threshold].index
    return df.drop(columns=cols_to_drop)


def flatten_nested_data(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested book_info_table column."""
    if "book_info_table" not in df.columns:
        return df

    info_df = pd.json_normalize(df["book_info_table"])
    return pd.concat([df.drop("book_info_table", axis=1), info_df], axis=1)


def parse_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Dimensions into width (en) and height (boy)."""
    if "Dimensions" not in df.columns:
        return df

    split_df = df["Dimensions"].str.split("x", expand=True)
    if split_df.shape[1] >= 2:
        df["en"] = pd.to_numeric(
            split_df[0].str.strip().str.replace(" cm", ""), errors="coerce"
        )
        df["boy"] = pd.to_numeric(
            split_df[1].str.strip().str.replace(" cm", ""), errors="coerce"
        )

    return df.drop("Dimensions", axis=1)


def clean_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert specified columns to numeric."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_language_column(df: pd.DataFrame) -> pd.DataFrame:
    """Extract primary language from Language column."""
    if "Language" not in df.columns:
        return df

    df["Language"] = (
        df["Language"]
        .str.upper()
        .str.strip()
        .str.split(",")
        .str[0]
        .str.strip()
        .str.title()
    )
    return df


def process_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure categories is a list."""
    df["categories"] = df["categories"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return df


def parse_dates_and_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Parse publication dates and clean price columns."""
    # List Price
    if "List Price" in df.columns:
        df["List Price"] = pd.to_numeric(
            df["List Price"].str.replace(",", ""), errors="coerce"
        )

    # Page Count
    if "Page Count" in df.columns:
        df["Page Count"] = pd.to_numeric(df["Page Count"], errors="coerce")

    # Publication Date
    if "Publication Date" in df.columns:
        df["Publication Date"] = pd.to_datetime(
            df["Publication Date"], errors="coerce", dayfirst=True
        )
        df["publish_year"] = df["Publication Date"].dt.year.astype("float")

    return df


def calculate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate estimated revenue."""
    df["estimated_revenue"] = df["Kitapyurdu Price"] * df["Purchase Count"]
    return df


def load_and_process_data(file_path: str) -> pd.DataFrame:
    df = pd.read_json(file_path, lines=True, encoding="utf-8")

    df = (
        df.pipe(flatten_nested_data)
        .drop(columns=["url", "id"], errors="ignore")
        .pipe(drop_high_null_columns, 80)
        .dropna()
        .rename(columns=COLUMN_RENAME)
        .pipe(parse_dimensions)
        .pipe(process_categories)
        .pipe(
            clean_numeric_cols,
            [
                "Purchase Count",
                "Comment Count",
                "Kitapyurdu Price",
                "Going to Read Count",
                "Reading Count",
                "Read Count",
                "en",
                "boy",
            ],
        )
        .pipe(parse_dates_and_prices)
        .pipe(normalize_language_column)
        .pipe(calculate_revenue)
    )

    return df

def sample_raw_book_data():
    df = load_and_process_data(BOOK_DATA_FILE_PATH)
    df_sampled = df.sample(n=SAMPLE_SIZE, random_state=42)
    df_sampled.to_parquet(SAMPLED_BOOK_DATA_FILE_PATH, compression="gzip")
