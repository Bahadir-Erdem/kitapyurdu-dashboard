from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import random
import numpy as np

from config import SAMPLED_BOOK_DATA_FILE_PATH


TRANSLATIONS = {
    "en": {
        "title": "💰 Kitapyurdu Business Intelligence Dashboard",
        "subtitle": "**Interactive filtering for targeted business insights.**\nUse the sidebar filters to analyze specific categories and time periods.",
        "filters_header": "🎯 Smart Filters",
        "filters_desc": "**Filter data to get targeted insights**",
        "category_filter": "📚 Category Filter",
        "select_categories": "Select Categories",
        "all_categories": "All Categories",
        "time_period": "📅 Time Period",
        "publication_year": "Publication Year Range",
        "reset_filters": "🔄 Reset All Filters",
        "showing_all": "📊 **Showing all data:** {count:,} books",
        "filters_active": "🔍 **Filters Active:** Showing {filtered:,} of {total:,} books ({pct:.1f}% filtered out)",
        "executive_summary": "📊 Summary",
        "total_revenue": "Total Revenue (Filtered)",
        "total_sales": "Total Books Sold",
        "avg_price": "Average Book Price",
        "avg_revenue_book": "Avg Revenue/Book",
        "vs_all": "vs all",
        "books": "books",
        "engagement_metrics": "📈 Engagement Metrics",
        "avg_comments": "Avg Comments/Book",
        "avg_readers": "Avg Readers/Book",
        "reading_now": "Reading Now",
        "want_to_read": "Want to Read",
        "completed": "Completed",
        "engagement_score": "Engagement Score",
        "top_performers": "🏆 Top Performers (Filtered View)",
        "top_books_revenue": "📚 Top Books by Revenue",
        "top_books_sales": "📈 Top Books by Sales",
        "top_books_engagement": "❤️ Most Engaging Books",
        "top_authors": "✍️ Top Authors",
        "sort_by": "Sort by:",
        "revenue": "Revenue",
        "sales_volume": "Sales Volume",
        "engagement": "Engagement",
        "price": "Price",
        "book_name": "Book Name",
        "author": "Author",
        "estimated_revenue": "Revenue",
        "purchase_count": "Sales",
        "comments": "Comments",
        "readers": "Readers",
        "publish_year": "Year",
        "log_scale_note": "📊 **Note:** Using logarithmic scale to better visualize differences",
        "total_revenue_author": "Total Revenue by Author",
        "revenue_per_book": "Revenue per Book by Author",
        "book_count": "Book Count",
        "pricing_insights": "💰 Pricing Strategy Insights",
        "revenue_by_price": "Revenue by Price Range",
        "sales_vs_revenue": "Sales Volume vs Revenue",
        "price_range": "Price Range",
        "optimal_price": "💡 **Optimal Price Range:** {range} generates {revenue} with {count} books available.",
        "trends_header": "📈 Trends Over Time",
        "no_trend_data": "No trend data available for current filters",
        "revenue_trend": "Revenue Trend Over Years",
        "year": "Year",
        "books_published": "Books Published by Year",
        "avg_price_trend": "Average Price Trend",
        "dashboard_view": "📊 Panel Görünümü:",
        "complete_overview": "Complete Overview",
        "performance_focus": "Performance Focus",
        "engagement_focus": "Engagement Deep Dive",
        "pricing_analysis": "Pricing Analysis",
        "engagement_analysis": "💝 Engagement Analysis",
        "most_commented": "Most Commented Books",
        "most_wanted": "Most Wanted Books (To Read)",
        "most_read": "Most Read Books",
        "engagement_overview": "Reader Engagement Overview",
        "total_engagement": "Total Engagement",
        "engagement_by_category": "Engagement by Category",
        "data_preview": "Data Preview (First 20 Rows)",
    },
    "tr": {
        "title": "💰 Kitapyurdu İş Zekası Paneli",
        "subtitle": "**Hedefli iş içgörüleri için interaktif filtreleme.**\nBelirli kategorileri ve zaman dilimlerini analiz etmek için yan çubuk filtrelerini kullanın.",
        "filters_header": "🎯 Akıllı Filtreler",
        "filters_desc": "**Hedefli içgörüleri için veriyi filtreleyin**",
        "category_filter": "📚 Kategori Filtresi",
        "select_categories": "Kategori Seçin",
        "all_categories": "Tüm Kategoriler",
        "time_period": "📅 Zaman Dilimi",
        "publication_year": "Yayın Yılı Aralığı",
        "reset_filters": "🔄 Tüm Filtreleri Sıfırla",
        "showing_all": "📊 **Tüm veri gösteriliyor:** {count:,} kitap",
        "filters_active": "🔍 **Filtreler Aktif:** {total:,} kitaptan {filtered:,} tanesi gösteriliyor (%{pct:.1f} filtrelendi)",
        "executive_summary": "📊 Özet",
        "total_revenue": "Toplam Gelir (Filtrelenmiş)",
        "total_sales": "Toplam Satılan Kitap",
        "avg_price": "Ortalama Kitap Fiyatı",
        "avg_revenue_book": "Ort. Gelir/Kitap",
        "vs_all": "tümüne göre",
        "books": "kitap",
        "engagement_metrics": "📈 Etkileşim Metrikleri",
        "avg_comments": "Ort. Yorum/Kitap",
        "avg_readers": "Ort. Okuyucu/Kitap",
        "reading_now": "Şu an Okuyor",
        "want_to_read": "Okumak İstiyor",
        "completed": "Tamamlandı",
        "engagement_score": "Etkileşim Skoru",
        "top_performers": "🏆 En İyi Performanslar (Filtrelenmiş Görünüm)",
        "top_books_revenue": "📚 Gelire Göre En İyi Kitaplar",
        "top_books_sales": "📈 Satışa Göre En İyi Kitaplar",
        "top_books_engagement": "❤️ En Çok Etkileşim Alan Kitaplar",
        "top_authors": "✍️ En İyi Yazarlar",
        "sort_by": "Sıralama:",
        "revenue": "Gelir",
        "sales_volume": "Satış Hacmi",
        "engagement": "Etkileşim",
        "price": "Fiyat",
        "book_name": "Kitap Adı",
        "author": "Yazar",
        "estimated_revenue": "Gelir",
        "purchase_count": "Satış",
        "comments": "Yorumlar",
        "readers": "Okuyucular",
        "publish_year": "Yıl",
        "log_scale_note": "📊 **Not:** Farkları daha iyi görmek için logaritmik ölçek kullanılıyor",
        "total_revenue_author": "Yazara Göre Toplam Gelir",
        "revenue_per_book": "Yazara Göre Kitap Başına Gelir",
        "book_count": "Kitap Sayısı",
        "pricing_insights": "💰 Fiyatlandırma Stratejisi İçgörüleri",
        "revenue_by_price": "Fiyat Aralığına Göre Gelir",
        "sales_vs_revenue": "Satış Hacmi vs Gelir",
        "price_range": "Fiyat Aralığı",
        "optimal_price": "💡 **Optimal Fiyat Aralığı:** {range} aralığı {revenue} gelir üretiyor ve {count} kitap mevcut.",
        "trends_header": "📈 Zaman İçinde Trendler",
        "no_trend_data": "Mevcut filtreler için trend verisi yok",
        "revenue_trend": "Yıllara Göre Gelir Trendi",
        "year": "Yıl",
        "books_published": "Yıla Göre Yayınlanan Kitaplar",
        "avg_price_trend": "Ortalama Fiyat Trendi",
        "dashboard_view": "📊 Panel Görünümü:",
        "complete_overview": "Tam Genel Bakış",
        "performance_focus": "Performans Odaklı",
        "engagement_focus": "Etkileşim Odaklı",
        "pricing_analysis": "Fiyatlandırma Analizi",
        "engagement_analysis": "💝 Etkileşim Analizi",
        "most_commented": "En Çok Yorumlanan Kitaplar",
        "most_wanted": "En Çok İstenen Kitaplar (Okunacak)",
        "most_read": "En Çok Okunan Kitaplar",
        "engagement_overview": "Okuyucu Etkileşim Genel Bakış",
        "total_engagement": "Toplam Etkileşim",
        "engagement_by_category": "Kategoriye Göre Etkileşim",
        "data_preview": "Veri Önizlemesi (İlk 20 Satır)",
    },
}

st.set_page_config(
    page_title="Kitapyurdu BI Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "language" not in st.session_state:
    st.session_state.language = "tr"


def t(key: str) -> str:
    return TRANSLATIONS[st.session_state.language].get(key, key)


CACHE_TTL = 3600


@st.cache_data(ttl=CACHE_TTL)
def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(str(file_path), engine="fastparquet")
    df["estimated_revenue"] = (df["Kitapyurdu Price"] * df["Purchase Count"]).round(2)

    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(
            lambda x: tuple(x) if isinstance(x, list) else ()
        )

    if "Favorite Count" not in df.columns:
        df["Favorite Count"] = 0

    return df


def get_primary_category(categories_tuple) -> str:
    if isinstance(categories_tuple, tuple) and len(categories_tuple) > 0:
        if categories_tuple[0] != "Kitap":
            return categories_tuple[0]
        elif len(categories_tuple) > 1:
            return categories_tuple[1]
    return "Other"


def format_currency(value: float) -> str:
    return f"₺{value:,.2f}"


def format_number(value: float) -> str:
    lang = st.session_state.language

    if value >= 1_000_000_000:
        if lang == "en":
            return f"{value / 1_000_000_000:.2f} Billion"
        else:
            return f"{value / 1_000_000_000:.2f} Milyar"
    elif value >= 1_000_000:
        if lang == "en":
            return f"{value / 1_000_000:.2f} Million"
        else:
            return f"{value / 1_000_000:.2f} Milyon"
    elif value >= 1_000:
        if lang == "en":
            return f"{value / 1_000:.2f} Thousand"
        else:
            return f"{value / 1_000:.2f} Bin"
    else:
        return f"{value:.2f}"


def get_all_categories(df: pd.DataFrame) -> List[str]:
    all_cats = set()
    for cats in df["categories"]:
        if isinstance(cats, tuple):
            all_cats.update(cats)
    all_cats.discard("Kitap")
    return sorted(list(all_cats))


def apply_filters(
    df: pd.DataFrame, categories: List[str], year_range: Tuple[int, int]
) -> pd.DataFrame:
    filtered_df = df.copy()

    if categories and t("all_categories") not in categories:
        filtered_df = filtered_df[
            filtered_df["categories"].apply(
                lambda x: any(cat in x for cat in categories)
            )
        ]

    if "publish_year" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["publish_year"] >= year_range[0])
            & (filtered_df["publish_year"] <= year_range[1])
        ]

    return filtered_df


def calculate_engagement_score(row) -> float:
    comments = row.get("Comment Count", 0)
    readers = (
        row.get("Read Count", 0)
        + row.get("Reading Count", 0)
        + row.get("Going to Read Count", 0)
    )

    return (comments * 3) + (readers * 1)


def calculate_market_overview(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "total_books": 0,
            "total_revenue": 0,
            "avg_price": 0,
            "avg_sales": 0,
            "total_sales": 0,
            "avg_revenue_per_book": 0,
            "avg_comments": 0,
            "total_readers": 0,
            "avg_readers": 0,
        }
    return {
        "total_books": len(df),
        "total_revenue": df["estimated_revenue"].sum(),
        "avg_price": df["Kitapyurdu Price"].mean(),
        "avg_sales": df["Purchase Count"].mean(),
        "total_sales": df["Purchase Count"].sum(),
        "avg_revenue_per_book": df["estimated_revenue"].mean(),
        "avg_comments": df["Comment Count"].mean(),
        "total_readers": df["Read Count"].sum()
        + df["Reading Count"].sum()
        + df["Going to Read Count"].sum(),
        "avg_readers": (
            df["Read Count"] + df["Reading Count"] + df["Going to Read Count"]
        ).mean(),
    }


def get_top_books(
    df: pd.DataFrame,
    metric: str = "revenue",
    top_n: int = 10,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy["Engagement Score"] = df_copy.apply(calculate_engagement_score, axis=1)

    if metric == "revenue":
        sort_col = "estimated_revenue"
    elif metric == "sales":
        sort_col = "Purchase Count"
    elif metric == "engagement":
        sort_col = "Engagement Score"
    elif metric == "comments":
        sort_col = "Comment Count"
    elif metric == "wanted":
        sort_col = "Going to Read Count"
    elif metric == "read":
        sort_col = "Read Count"
    else:
        sort_col = "estimated_revenue"

    cols_to_keep = [
        "Book Name",
        "author",
        "estimated_revenue",
        "Purchase Count",
        "Kitapyurdu Price",
        "publish_year",
        "Comment Count",
        "Read Count",
        "Reading Count",
        "Going to Read Count",
        "Engagement Score",
        "categories",
    ]

    cols_to_keep = [col for col in cols_to_keep if col in df_copy.columns]

    if sort_col not in df_copy.columns:
        return pd.DataFrame(
            columns=[col for col in cols_to_keep if col != "categories"]
            + ["Primary Category"]
        )

    top = df_copy.nlargest(top_n, sort_col)[cols_to_keep].copy()

    if "categories" in top.columns:
        top["Primary Category"] = top["categories"].apply(get_primary_category)
        top = top.drop("categories", axis=1)
    else:
        top["Primary Category"] = "Other"

    return top


def get_top_authors(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if df.empty or "author" not in df.columns:
        return pd.DataFrame()

    agg_dict = {
        "estimated_revenue": "sum",
        "Purchase Count": "sum",
        "Book Name": "count",
        "Kitapyurdu Price": "mean",
        "Comment Count": "sum",
        "Read Count": "sum",
    }

    cols_to_agg = {k: v for k, v in agg_dict.items() if k in df.columns}

    if "Book Name" not in df.columns:
        cols_to_agg["author"] = "count"  # Need a column to count books

    author_stats = (
        df[df["author"] != "No Author"].groupby("author").agg(cols_to_agg).round(2)
    )

    if author_stats.empty:
        return pd.DataFrame()

    author_stats.columns = [
        "Total Revenue",
        "Total Sales",
        "Book Count",
        "Avg Price",
        "Total Comments",
        "Total Readers",
    ]
    author_stats["Revenue per Book"] = (
        author_stats["Total Revenue"] / author_stats["Book Count"]
    ).round(2)
    author_stats["Engagement per Book"] = (
        (author_stats["Total Comments"] * 3 + author_stats["Total Readers"])
        / author_stats["Book Count"]
    ).round(2)

    return (
        author_stats.sort_values("Total Revenue", ascending=False)
        .head(top_n)
        .reset_index()
    )


def get_pricing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Kitapyurdu Price" not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy["Price Range"] = pd.cut(
        df_copy["Kitapyurdu Price"],
        bins=[0, 50, 100, 150, 200, 300, 500, 1000, float("inf")],
        labels=[
            "0-50₺",
            "50-100₺",
            "100-150₺",
            "150-200₺",
            "200-300₺",
            "300-500₺",
            "500-1000₺",
            "1000₺+",
        ],
    )

    pricing = (
        df_copy.groupby("Price Range", observed=True)
        .agg(
            {
                "Purchase Count": ["sum", "mean"],
                "estimated_revenue": "sum",
                "Book Name": "count",
            }
        )
        .round(2)
    )

    pricing.columns = [
        "Total Sales",
        "Avg Sales per Book",
        "Total Revenue",
        "Book Count",
    ]

    return pricing.reset_index().sort_values("Total Revenue", ascending=False)


def get_yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    if "publish_year" not in df.columns or df.empty:
        return pd.DataFrame()

    trends = (
        df[df["publish_year"] >= 2000]
        .groupby("publish_year")
        .agg(
            {
                "estimated_revenue": "sum",
                "Purchase Count": "sum",
                "Book Name": "count",
                "Kitapyurdu Price": "mean",
            }
        )
        .round(2)
    )

    if trends.empty:
        return pd.DataFrame()

    trends.columns = ["Revenue", "Sales", "Books Published", "Avg Price"]
    return trends.reset_index()


def create_sidebar_filters(df: pd.DataFrame) -> Tuple:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button(
            "🇬🇧 English",
            width="stretch",
            type="primary" if st.session_state.language == "en" else "secondary",
        ):
            st.session_state.language = "en"
            st.rerun()
    with col2:
        if st.sidebar.button(
            "🇹🇷 Türkçe",
            width="stretch",
            type="primary" if st.session_state.language == "tr" else "secondary",
        ):
            st.session_state.language = "tr"
            st.rerun()

    st.sidebar.divider()

    st.sidebar.header(t("filters_header"))
    st.sidebar.markdown(t("filters_desc"))

    st.sidebar.subheader(t("category_filter"))
    all_categories = get_all_categories(df)

    category_options = [t("all_categories")] + all_categories
    selected_categories = st.sidebar.multiselect(
        t("select_categories"),
        options=category_options,
        default=[t("all_categories")],
        placeholder=t("select_categories"),
        help=t("select_categories"),
    )

    st.sidebar.subheader(t("time_period"))
    if "publish_year" in df.columns and not df.empty:
        min_year = int(df["publish_year"].min())
        max_year = int(df["publish_year"].max())
        default_min = max(min_year, 2010)
        if default_min > max_year:
            default_min = min_year

        year_range = st.sidebar.slider(
            t("publication_year"),
            min_value=min_year,
            max_value=max_year,
            value=(default_min, max_year),
        )
    else:
        year_range = (2010, 2024)

    if st.sidebar.button(t("reset_filters"), width="stretch"):
        st.rerun()

    return selected_categories, year_range


def show_filter_summary(original_count: int, filtered_count: int, filters_active: bool):
    if filters_active:
        reduction_pct = (
            (original_count - filtered_count) / original_count * 100
            if original_count > 0
            else 0
        )
        st.info(
            t("filters_active").format(
                filtered=filtered_count, total=original_count, pct=reduction_pct
            )
        )
    else:
        st.success(t("showing_all").format(count=original_count))


def show_executive_summary(metrics: Dict, filtered_metrics: Dict, filters_active: bool):
    st.header(t("executive_summary"))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_val = (
            (filtered_metrics["total_revenue"] - metrics["total_revenue"])
            / metrics["total_revenue"]
            * 100
            if filters_active and metrics["total_revenue"] > 0
            else 0
        )
        delta = None if not filters_active else f"{delta_val:.1f}% {t('vs_all')}"
        st.metric(
            t("total_revenue"),
            format_number(filtered_metrics["total_revenue"]) + "₺",
            delta=delta,
        )

    with col2:
        delta = (
            None
            if not filters_active
            else f"{filtered_metrics['total_books']:,} {t('books')}"
        )
        st.metric(
            t("total_sales"),
            format_number(filtered_metrics["total_sales"]),
            delta=delta,
        )

    with col3:
        st.metric(t("avg_price"), format_currency(filtered_metrics["avg_price"]))

    with col4:
        st.metric(
            t("avg_revenue_book"), f"₺{filtered_metrics['avg_revenue_per_book']:,.0f}"
        )

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(t("avg_comments"), f"{filtered_metrics['avg_comments']:.2f}")

    with col2:
        st.metric(t("avg_readers"), f"{filtered_metrics['avg_readers']:.2f}")

    with col3:
        st.metric(
            t("total_engagement"), format_number(filtered_metrics["total_readers"])
        )


def show_top_performers(df: pd.DataFrame):
    st.header(t("top_performers"))

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            t("top_books_revenue"),
            t("top_books_sales"),
            t("top_books_engagement"),
            t("top_authors"),
        ]
    )

    with tab1:
        st.subheader(t("top_books_revenue"))
        top_revenue = get_top_books(df, "revenue", 10)

        if top_revenue.empty:
            st.warning(t("no_trend_data"))
        else:
            display_top = top_revenue.head(10).sort_values(
                "estimated_revenue", ascending=True
            )
            fig = px.bar(
                display_top,
                x="estimated_revenue",
                y="Book Name",
                color="Primary Category",
                title=t("top_books_revenue"),
                color_discrete_sequence=px.colors.qualitative.Pastel1,
                category_orders=display_top["estimated_revenue"].to_dict(),
                labels={
                    "estimated_revenue": t("revenue"),
                    "Book Name": t("book_name"),
                    "Primary Category": t("category_filter"),
                },
            ).update_layout(yaxis_categoryorder="total ascending")
            fig.update_traces(
                hovertemplate=f"<b>%{{y}}</b><br><b>{t('revenue')}:</b> ₺%{{x:,.0f}}<extra></extra>"
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader(t("top_books_sales"))
        top_sales = get_top_books(df, "sales", 10)

        if top_sales.empty:
            st.warning(t("no_trend_data"))
        else:
            display_top = top_sales.head(10).sort_values(
                "Purchase Count", ascending=True
            )
            fig = px.bar(
                display_top,
                x="Purchase Count",
                y="Book Name",
                color="Primary Category",
                title=t("top_books_sales"),
                color_discrete_sequence=px.colors.qualitative.Pastel2,
                labels={
                    "Purchase Count": t("sales_volume"),
                    "Book Name": t("book_name"),
                    "Primary Category": t("category_filter"),
                },
            ).update_layout(yaxis_categoryorder="total ascending")
            fig.update_traces(
                hovertemplate=f"<b>%{{y}}</b><br><b>{t('sales_volume')}:</b> %{{x:,.0f}}<extra></extra>"
            )
            st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader(t("top_books_engagement"))

        if st.session_state.language == "en":
            st.info(
                """
            **📊 Engagement Score = (Comments × 3) + (Readers × 1)** This weighted formula prioritizes active engagement (comments) over passive interest.
            """
            )
        else:
            st.info(
                """
            **📊 Etkileşim Skoru = (Yorumlar × 3) + (Okuyucular × 1)**
            Bu ağırlıklı formül, aktif etkileşimi (yorumlar) pasif ilgiye göre önceliklendirir.
            """
            )

        top_engagement = get_top_books(df, "engagement", 10)

        if top_engagement.empty:
            st.warning(t("no_trend_data"))
        else:
            display_top = top_engagement.head(10).sort_values(
                "Engagement Score", ascending=True
            )
            fig = px.bar(
                display_top,
                x="Engagement Score",
                y="Book Name",
                color="Primary Category",
                title=t("top_books_engagement"),
                color_discrete_sequence=px.colors.qualitative.Set3,
                labels={
                    "Engagement Score": t("engagement_score"),
                    "Book Name": t("book_name"),
                    "Primary Category": t("category_filter"),
                },
            ).update_layout(yaxis_categoryorder="total ascending")
            fig.update_traces(
                hovertemplate=f"<b>%{{y}}</b><br><b>{t('engagement_score')}:</b> %{{x:,.0f}}<extra></extra>"
            )
            st.plotly_chart(fig, width="stretch")

    with tab4:
        st.subheader(t("top_authors"))

        authors = get_top_authors(df, 15)

        if authors.empty:
            st.warning(t("no_trend_data"))
        else:
            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.bar(
                    authors.head(10).sort_values("Total Revenue", ascending=True),
                    x="Total Revenue",
                    y="author",
                    title=t("total_revenue_author"),
                    color_discrete_sequence=["#1f77b4"],
                    labels={
                        "Total Revenue": t("total_revenue"),
                        "author": t("author"),
                    },
                ).update_layout(yaxis_categoryorder="total ascending")
                fig1.update_traces(
                    hovertemplate=f"<b>%{{y}}</b><br><b>{t('total_revenue')}:</b> ₺%{{x:,.0f}}<extra></extra>"
                )
                st.plotly_chart(fig1, width="stretch")

            with col2:
                fig2 = px.bar(
                    authors.head(10).sort_values("Revenue per Book", ascending=True),
                    x="Revenue per Book",
                    y="author",
                    title=t("revenue_per_book"),
                    color_discrete_sequence=["#2ca02c"],
                    labels={
                        "Revenue per Book": t("revenue_per_book"),
                        "author": t("author"),
                    },
                ).update_layout(yaxis_categoryorder="total ascending")
                fig2.update_traces(
                    hovertemplate=f"<b>%{{y}}</b><br><b>{t('revenue_per_book')}:</b> ₺%{{x:,.0f}}<extra></extra>"
                )
                st.plotly_chart(fig2, width="stretch")


def show_engagement_analysis(df: pd.DataFrame):
    st.header(t("engagement_analysis"))

    tab1, tab2, tab3 = st.tabs([t("most_commented"), t("most_wanted"), t("most_read")])

    with tab1:
        st.subheader(t("most_commented"))
        top_commented = get_top_books(df, "comments", 10)

        if top_commented.empty:
            st.warning(t("no_trend_data"))
        else:
            display_top = top_commented.head(10).sort_values(
                "Comment Count", ascending=True
            )
            fig = px.bar(
                display_top,
                x="Comment Count",
                y="Book Name",
                color="Primary Category",
                title=t("most_commented"),
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={
                    "Comment Count": t("comments"),
                    "Book Name": t("book_name"),
                    "Primary Category": t("category_filter"),
                },
            ).update_layout(yaxis_categoryorder="total ascending")
            fig.update_traces(
                hovertemplate=f"<b>%{{y}}</b><br><b>{t('comments')}:</b> %{{x:,.0f}}<extra></extra>"
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader(t("most_wanted"))
        top_wanted = get_top_books(df, "wanted", 10)

        if top_wanted.empty:
            st.warning(t("no_trend_data"))
        else:
            display_top = top_wanted.head(10).sort_values(
                "Going to Read Count", ascending=True
            )
            fig = px.bar(
                display_top,
                x="Going to Read Count",
                y="Book Name",
                color="Primary Category",
                title=t("most_wanted"),
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={
                    "Going to Read Count": t("want_to_read"),
                    "Book Name": t("book_name"),
                    "Primary Category": t("category_filter"),
                },
            ).update_layout(yaxis_categoryorder="total ascending")
            fig.update_traces(
                hovertemplate=f"<b>%{{y}}</b><br><b>{t('want_to_read')}:</b> %{{x:,.0f}}<extra></tra>"
            )
            st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader(t("most_read"))
        top_read = get_top_books(df, "read", 10)

        if top_read.empty:
            st.warning(t("no_trend_data"))
        else:
            display_top = top_read.head(10).sort_values("Read Count", ascending=True)
            fig = px.bar(
                display_top,
                x="Read Count",
                y="Book Name",
                color="Primary Category",
                title=t("most_read"),
                color_discrete_sequence=px.colors.qualitative.Set3,
                labels={
                    "Read Count": t("completed"),
                    "Book Name": t("book_name"),
                    "Primary Category": t("category_filter"),
                },
            ).update_layout(yaxis_categoryorder="total ascending")
            fig.update_traces(
                hovertemplate=f"<b>%{{y}}</b><br><b>{t('completed')}:</b> %{{x:,.0f}}<extra></extra>"
            )
            st.plotly_chart(fig, width="stretch")


def show_pricing_insights(df: pd.DataFrame):
    st.header(t("pricing_insights"))

    pricing = get_pricing_analysis(df)

    if pricing.empty:
        st.warning(t("no_trend_data"))
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.bar(
            pricing,
            x="Price Range",
            y="Total Revenue",
            title=t("revenue_by_price"),
            color="Total Revenue",
            color_continuous_scale="Viridis",
            labels={
                "Price Range": t("price_range"),
                "Total Revenue": t("total_revenue"),
            },
        )
        fig.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br><b>{t('total_revenue')}:</b> ₺%{{y:,.0f}}<extra></extra>"
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.scatter(
            pricing,
            x="Avg Sales per Book",
            y="Total Revenue",
            size="Book Count",
            text="Price Range",
            title=t("sales_vs_revenue"),
            color="Price Range",
            labels={
                "Avg Sales per Book": t("avg_revenue_book"),
                "Total Revenue": t("total_revenue"),
                "Price Range": t("price_range"),
            },
        )
        fig.update_traces(
            textposition="top center",
            hovertemplate=f"<b>{t('price_range')}:</b> %{{text}}<br>"
            + f"<b>{t('total_revenue')}:</b> ₺%{{y:,.0f}}<br>"
            + f"<b>{t('avg_sales')}:</b> %{{x:,.2f}}<br>"
            + f"<b>{t('book_count')}:</b> %{{marker.size:,.0f}}<extra></extra>",
        )
        st.plotly_chart(fig, width="stretch")

    best_range = pricing.iloc[0]
    st.info(
        t("optimal_price").format(
            range=best_range["Price Range"],
            revenue=format_number(best_range["Total Revenue"]),
            count=int(best_range["Book Count"]),
        )
    )


def show_trends(df: pd.DataFrame):
    st.header(t("trends_header"))

    trends = get_yearly_trends(df)

    if trends.empty:
        st.warning(t("no_trend_data"))
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trends["publish_year"],
            y=trends["Revenue"],
            name=t("revenue"),
            mode="lines+markers",
            line=dict(color="green", width=3),
            hovertemplate=f"<b>{t('revenue')}:</b> ₺%{{y:,.0f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=t("revenue_trend"),
        xaxis={"title": t("year")},
        yaxis={"title": t("revenue") + " (₺)"},
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.bar(
            trends,
            x="publish_year",
            y="Books Published",
            title=t("books_published"),
            labels={
                "publish_year": t("year"),
                "Books Published": t("books_published"),
            },
        )
        fig2.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br><b>{t('books_published')}:</b> %{{y:,.0f}}<extra></extra>"
        )
        st.plotly_chart(fig2, width="stretch")

    with col2:
        fig3 = px.line(
            trends,
            x="publish_year",
            y="Avg Price",
            title=t("avg_price_trend"),
            markers=True,
            labels={
                "publish_year": t("year"),
                "Avg Price": t("avg_price"),
            },
        )
        fig3.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br><b>{t('avg_price')}:</b> ₺%{{y:,.2f}}<extra></extra>"
        )
        st.plotly_chart(fig3, width="stretch")


def main():
    st.title(t("title"))
    st.markdown(t("subtitle"))

    df_full = load_data(SAMPLED_BOOK_DATA_FILE_PATH)

    selected_categories, year_range = create_sidebar_filters(df_full)

    if not year_range or len(year_range) < 2 or year_range[0] > year_range[1]:
        year_range = (2010, 2024)

    default_min_year = df_full["publish_year"].min() if not df_full.empty else 2010
    default_max_year = df_full["publish_year"].max() if not df_full.empty else 2024

    filters_active = t("all_categories") not in selected_categories or (
        "publish_year" in df_full.columns
        and not df_full.empty
        and year_range
        != (
            int(default_min_year),
            int(default_max_year),
        )
    )

    df_filtered = apply_filters(df_full, selected_categories, year_range)

    show_filter_summary(len(df_full), len(df_filtered), filters_active)

    metrics_full = calculate_market_overview(df_full)
    metrics_filtered = calculate_market_overview(df_filtered)

    if df_filtered.empty:
        st.warning(
            "No data matches the current filters. Please adjust your selection."
            if st.session_state.language == "en"
            else "Mevcut filtrelere uyan veri bulunamadı. Lütfen seçiminizi değiştirin."
        )
    else:
        show_executive_summary(metrics_full, metrics_filtered, filters_active)
        st.divider()

        show_top_performers(df_filtered)

        st.divider()

        show_engagement_analysis(df_filtered)

        st.divider()
        show_pricing_insights(df_filtered)
        st.divider()
        show_trends(df_filtered)

    st.divider()
    with st.expander(t("data_preview"), expanded=False):
        st.dataframe(df_full.head(20))


if __name__ == "__main__":
    main()
