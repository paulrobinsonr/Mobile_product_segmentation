# ============================================================
# MOBILE PRODUCT SEGMENTATION & RECOMMENDATION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Product Intelligence",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #080D18;
        color: #E5E7EB;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0D1424 0%,
            #0A1020 100%
        );

        border-right: 1px solid #1D2A44;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }

    /* Sidebar title */

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .sidebar-logo-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;

        background: linear-gradient(
            135deg,
            #2563EB,
            #7C3AED
        );

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 23px;

        box-shadow:
            0 8px 20px rgba(37, 99, 235, 0.25);
    }

    .sidebar-title {
        color: #F8FAFC;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    .sidebar-subtitle {
        color: #64748B;
        font-size: 12px;
        margin-bottom: 28px;
    }

    /* Navigation label */

    .nav-label {
        color: #64748B;
        font-size: 11px;
        font-weight: 700;

        text-transform: uppercase;
        letter-spacing: 1.2px;

        margin: 15px 0 10px 5px;
    }

    /* Radio navigation */

    div[data-testid="stRadio"] > label {
        display: none;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px;
    }

    div[data-testid="stRadio"] label {
        background: transparent !important;

        border-radius: 10px;

        padding: 11px 12px !important;

        transition: all 0.2s ease;
    }

    div[data-testid="stRadio"] label:hover {
        background: #131E33 !important;
    }

    div[data-testid="stRadio"] label p {
        color: #CBD5E1 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Hide radio circles */

    div[data-testid="stRadio"] label div:first-child {
        display: none;
    }

    /* ======================================================
       MAIN TITLE
       ====================================================== */

    .main-title {
        font-size: 42px;
        font-weight: 750;

        letter-spacing: -1.5px;

        color: #F8FAFC;

        margin-bottom: 5px;
    }

    .main-title span {
        background: linear-gradient(
            90deg,
            #60A5FA,
            #A78BFA
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-subtitle {
        color: #94A3B8;
        font-size: 15px;
        margin-bottom: 30px;
    }

    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 25px;
        margin-bottom: 18px;
    }

    .section-description {
        color: #94A3B8;
        font-size: 14px;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* ======================================================
       METRIC CARDS
       ====================================================== */

    div[data-testid="stMetric"] {

        background: linear-gradient(
            145deg,
            #101A2D,
            #0D1628
        );

        border: 1px solid #1E3152;

        border-radius: 15px;

        padding: 20px 20px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.20);
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 13px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }

    /* ======================================================
       CARDS
       ====================================================== */

    .custom-card {

        background: linear-gradient(
            145deg,
            #101A2D,
            #0D1628
        );

        border: 1px solid #1E3152;

        border-radius: 15px;

        padding: 20px;

        margin-bottom: 15px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.18);
    }

    .card-title {
        color: #F8FAFC;
        font-size: 16px;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .card-text {
        color: #94A3B8;
        font-size: 13px;
    }

    /* ======================================================
       RECOMMENDATION CARD
       ====================================================== */

    .recommendation-card {

        background: linear-gradient(
            135deg,
            #101C35,
            #15142F
        );

        border: 1px solid #3949AB;

        border-radius: 15px;

        padding: 18px;

        margin-bottom: 12px;

        transition: transform 0.2s ease;
    }

    .recommendation-card:hover {
        transform: translateY(-2px);

        border-color: #6366F1;
    }

    .recommendation-title {
        color: #60A5FA;
        font-size: 17px;
        font-weight: 700;
    }

    .recommendation-brand {
        color: #CBD5E1;
        font-size: 13px;
        margin-top: 3px;
    }

    .recommendation-price {
        color: #F8FAFC;
        font-size: 20px;
        font-weight: 700;
    }

    .similarity-score {
        color: #34D399;
        font-size: 13px;
        font-weight: 650;
    }

    /* ======================================================
       SEGMENT BADGES
       ====================================================== */

    .segment-badge {

        display: inline-block;

        padding: 5px 10px;

        border-radius: 20px;

        background: #172554;

        color: #93C5FD;

        font-size: 11px;

        font-weight: 650;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {

        background: linear-gradient(
            90deg,
            #2563EB,
            #7C3AED
        );

        color: white;

        border: none;

        border-radius: 9px;

        font-weight: 600;

        padding: 9px 18px;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {

        transform: translateY(-1px);

        box-shadow:
            0 8px 20px rgba(99, 102, 241, 0.25);
    }

    /* ======================================================
       SELECT BOX
       ====================================================== */

    div[data-baseweb="select"] > div {

        background-color: #101827;

        border: 1px solid #273957;

        border-radius: 9px;
    }

    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {

        border: 1px solid #1E3152;

        border-radius: 12px;

        overflow: hidden;
    }

    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {

        color: #94A3B8;

        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {

        color: #60A5FA !important;
    }

    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {
        border-color: #1D2A44 !important;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        text-align: center;

        color: #475569;

        font-size: 12px;

        padding-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_models():

    product_features = joblib.load(
        DATA_DIR / "product_features.pkl"
    )

    kmeans = joblib.load(
        DATA_DIR / "kmeans_model.pkl"
    )

    scaler = joblib.load(
        DATA_DIR / "scaler.pkl"
    )

    similarity_matrix = joblib.load(
        DATA_DIR / "similarity_matrix.pkl"
    )

    clustering_features = joblib.load(
        DATA_DIR / "clustering_features.pkl"
    )

    return (
        product_features,
        kmeans,
        scaler,
        similarity_matrix,
        clustering_features
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    (
        product_features,
        kmeans,
        scaler,
        similarity_matrix,
        clustering_features
    ) = load_models()

except Exception as e:

    st.error(
        "Unable to load model files. "
        "Please make sure all .pkl files are inside the data folder."
    )

    st.exception(e)

    st.stop()


# ============================================================
# LOAD RAW DATASET
# ============================================================

@st.cache_data
def load_raw_data():

    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        return None

    return pd.read_csv(csv_files[0])


raw_df = load_raw_data()


# ============================================================
# PREPARE PRODUCT DATA
# ============================================================

products = product_features.copy()

products = products.reset_index(drop=True)


# ============================================================
# SEGMENT NAMES
# ============================================================

segment_names = {

    0: "Value Segment",

    1: "Mid-Range Segment",

    2: "Premium High-Rated Segment",

    3: "Premium Lower-Rated Segment"
}


if "cluster" in products.columns:

    products["segment"] = products["cluster"].map(
        segment_names
    ).fillna(
        products["cluster"].apply(
            lambda x: f"Segment {x}"
        )
    )

else:

    products["cluster"] = 0

    products["segment"] = "Unclassified"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">

        <div class="sidebar-logo-icon">
            📱
        </div>

        <div>
            <div class="sidebar-title">
                Mobile Intelligence
            </div>

            <div style="
                color:#64748B;
                font-size:11px;
                margin-top:2px;
            ">
                ML Analytics Platform
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Product analytics & recommendations'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nav-label">Workspace</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "📊  Dashboard",
            "🎯  Product Segmentation",
            "✨  Recommendation System"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<div class="nav-label">Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background:#101A2D;
            border:1px solid #1E3152;
            border-radius:10px;
            padding:12px;
        ">

        <div style="
            color:#CBD5E1;
            font-size:13px;
            font-weight:600;
        ">
        🤖 K-Means Clustering
        </div>

        <div style="
            color:#64748B;
            font-size:11px;
            margin-top:5px;
        ">
        4 product segments
        </div>

        <div style="
            color:#CBD5E1;
            font-size:13px;
            font-weight:600;
            margin-top:12px;
        ">
        🔍 Cosine Similarity
        </div>

        <div style="
            color:#64748B;
            font-size:11px;
            margin-top:5px;
        ">
        Product recommendations
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="footer">Mobile Product Intelligence<br>'
        'Machine Learning Project</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "📊  Dashboard":

    st.markdown(
        '<div class="main-title">'
        '📱 Mobile Product <span>Intelligence</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Machine learning based product segmentation and '
        'recommendation platform'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_products = len(products)

    total_brands = products["brand"].nunique()

    avg_price = products["price_usd"].mean()

    avg_rating = products["average_rating"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📦 Total Products",
            f"{total_products:,}"
        )

    with c2:
        st.metric(
            "🏷️ Total Brands",
            f"{total_brands}"
        )

    with c3:
        st.metric(
            "💰 Average Price",
            f"${avg_price:,.2f}"
        )

    with c4:
        st.metric(
            "⭐ Average Rating",
            f"{avg_rating:.2f}/5"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # BRAND ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📊 Product Market Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Products by brand

    with col1:

        brand_counts = (
            products
            .groupby("brand")
            .size()
            .reset_index(name="product_count")
            .sort_values(
                "product_count",
                ascending=False
            )
        )

        fig = px.bar(
            brand_counts,
            x="brand",
            y="product_count",
            title="Products by Brand",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            title_font_color="#F8FAFC",
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Average price by brand

    with col2:

        price_brand = (
            products
            .groupby("brand")["price_usd"]
            .mean()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig = px.bar(
            price_brand,
            x="brand",
            y="price_usd",
            title="Average Price by Brand",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            title_font_color="#F8FAFC",
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),
            yaxis_title="Price (USD)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # RATING + PRICE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        rating_brand = (
            products
            .groupby("brand")["average_rating"]
            .mean()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig = px.bar(
            rating_brand,
            x="brand",
            y="average_rating",
            title="Average Customer Rating by Brand",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            title_font_color="#F8FAFC",
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),
            yaxis_title="Rating"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            products,
            x="price_usd",
            y="average_rating",
            color="segment",
            hover_name="model",
            hover_data=[
                "brand",
                "overall_quality_score"
            ],
            title="Price vs Customer Rating",
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            title_font_color="#F8FAFC",
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 2 — PRODUCT SEGMENTATION
# ============================================================

elif page == "🎯  Product Segmentation":

    st.markdown(
        '<div class="main-title">'
        '🎯 Product <span>Segmentation</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'K-Means clustering groups mobile products according '
        'to price, ratings and quality characteristics.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SEGMENT SUMMARY
    # --------------------------------------------------------

    segment_summary = (

        products
        .groupby(
            ["cluster", "segment"],
            as_index=False
        )
        .agg(
            product_count=("model", "count"),
            avg_price=("price_usd", "mean"),
            avg_rating=("average_rating", "mean"),
            avg_quality=("overall_quality_score", "mean")
        )
        .sort_values("cluster")
    )

    st.markdown(
        '<div class="section-title">'
        '📌 Segment Overview'
        '</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(
        len(segment_summary)
    )

    for col, (_, row) in zip(
        cols,
        segment_summary.iterrows()
    ):

        with col:

            st.metric(
                row["segment"],
                f'{int(row["product_count"])} products'
            )

            st.caption(
                f'Avg. price: ${row["avg_price"]:,.0f}'
            )

    # --------------------------------------------------------
    # CLUSTER VISUALIZATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📈 Segment Visualization'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.scatter(
        products,
        x="price_usd",
        y="average_rating",
        color="segment",
        size="overall_quality_score",
        hover_name="model",
        hover_data=[
            "brand",
            "overall_quality_score"
        ],
        title="Mobile Product Segments",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        title_font_color="#F8FAFC"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # SEGMENT TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📋 Segment Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    display_summary = segment_summary.copy()

    display_summary["avg_price"] = (
        display_summary["avg_price"]
        .round(2)
    )

    display_summary["avg_rating"] = (
        display_summary["avg_rating"]
        .round(2)
    )

    display_summary["avg_quality"] = (
        display_summary["avg_quality"]
        .round(2)
    )

    display_summary.columns = [
        "Cluster",
        "Segment",
        "Products",
        "Average Price",
        "Average Rating",
        "Quality Score"
    ]

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # PRODUCT ASSIGNMENTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📱 Products by Segment'
        '</div>',
        unsafe_allow_html=True
    )

    selected_segment = st.selectbox(
        "Select a segment",
        sorted(
            products["segment"].unique()
        )
    )

    segment_products = products[
        products["segment"] == selected_segment
    ].copy()

    segment_products = segment_products[
        [
            "brand",
            "model",
            "price_usd",
            "average_rating",
            "overall_quality_score",
            "segment"
        ]
    ]

    segment_products = segment_products.sort_values(
        "price_usd"
    )

    st.dataframe(
        segment_products,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — RECOMMENDATION SYSTEM
# ============================================================

elif page == "✨  Recommendation System":

    st.markdown(
        '<div class="main-title">'
        '✨ Product <span>Recommendations</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Find products with similar characteristics using '
        'Cosine Similarity.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PRODUCT SELECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🔍 Find Similar Products'
        '</div>',
        unsafe_allow_html=True
    )

    product_names = products["model"].tolist()

    selected_product = st.selectbox(
        "Select a mobile product",
        product_names
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=min(10, len(products) - 1),
        value=5
    )

    # --------------------------------------------------------
    # SELECTED PRODUCT
    # --------------------------------------------------------

    selected_index = products[
        products["model"] == selected_product
    ].index[0]

    selected_data = products.iloc[
        selected_index
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Brand",
            selected_data["brand"]
        )

    with col2:

        st.metric(
            "Price",
            f'${selected_data["price_usd"]:,.2f}'
        )

    with col3:

        st.metric(
            "Rating",
            f'{selected_data["average_rating"]:.2f}'
        )

    with col4:

        st.metric(
            "Segment",
            selected_data["segment"]
        )

    # --------------------------------------------------------
    # RECOMMENDATION CALCULATION
    # --------------------------------------------------------

    similarity_scores = similarity_matrix[
        selected_index
    ]

    sorted_indices = (
        similarity_scores
        .argsort()[::-1]
    )

    recommendation_indices = [
        i
        for i in sorted_indices
        if i != selected_index
    ][
        :number_of_recommendations
    ]

    recommendations = products.iloc[
        recommendation_indices
    ].copy()

    recommendations[
        "similarity_score"
    ] = similarity_scores[
        recommendation_indices
    ]

    # --------------------------------------------------------
    # RECOMMENDATION RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🎯 Recommended Products'
        '</div>',
        unsafe_allow_html=True
    )

    for _, row in recommendations.iterrows():

        st.markdown(
            f"""
            <div class="recommendation-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div>

                        <div class="recommendation-title">
                            {row['model']}
                        </div>

                        <div class="recommendation-brand">
                            {row['brand']}
                        </div>

                    </div>

                    <div class="similarity-score">
                        {row['similarity_score'] * 100:.1f}% similar
                    </div>

                </div>

                <div style="
                    display:flex;
                    gap:30px;
                    margin-top:14px;
                    align-items:center;
                ">

                    <div class="recommendation-price">
                        ${row['price_usd']:,.2f}
                    </div>

                    <div>
                        ⭐ {row['average_rating']:.2f}
                    </div>

                    <div>
                        Quality: {row['overall_quality_score']:.2f}
                    </div>

                    <div class="segment-badge">
                        {row['segment']}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RECOMMENDATION TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📋 Recommendation Details'
        '</div>',
        unsafe_allow_html=True
    )

    recommendation_table = recommendations[
        [
            "brand",
            "model",
            "price_usd",
            "average_rating",
            "overall_quality_score",
            "segment",
            "similarity_score"
        ]
    ].copy()

    recommendation_table[
        "price_usd"
    ] = recommendation_table[
        "price_usd"
    ].round(2)

    recommendation_table[
        "average_rating"
    ] = recommendation_table[
        "average_rating"
    ].round(2)

    recommendation_table[
        "overall_quality_score"
    ] = recommendation_table[
        "overall_quality_score"
    ].round(2)

    recommendation_table[
        "similarity_score"
    ] = (
        recommendation_table[
            "similarity_score"
        ] * 100
    ).round(2)

    recommendation_table.columns = [
        "Brand",
        "Model",
        "Price (USD)",
        "Rating",
        "Quality Score",
        "Segment",
        "Similarity (%)"
    ]

    st.dataframe(
        recommendation_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        📱 Mobile Product Segmentation & Recommendation System
        <br>
        Built using Python • Pandas • Scikit-learn • Plotly • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)