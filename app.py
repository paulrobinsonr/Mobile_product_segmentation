import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Product Segmentation",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_product_data():

    return pd.read_csv(
        "data/product_features.csv"
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    scaler = joblib.load(
        "data/scaler.pkl"
    )

    kmeans = joblib.load(
        "data/kmeans_model.pkl"
    )

    similarity_matrix = joblib.load(
        "data/similarity_matrix.pkl"
    )

    clustering_features = joblib.load(
        "data/clustering_features.pkl"
    )

    return (
        scaler,
        kmeans,
        similarity_matrix,
        clustering_features
    )


# ============================================================
# INITIALIZE
# ============================================================

product_features = load_product_data()

(
    scaler,
    kmeans,
    similarity_matrix,
    clustering_features
) = load_models()


# ============================================================
# TITLE
# ============================================================

st.title(
    "📱 Mobile Product Segmentation & Recommendation System"
)

st.markdown(
    """
    ### Machine Learning Based Mobile Product Analysis

    This application uses **K-Means clustering** and
    **Cosine Similarity** to segment mobile products and
    recommend similar products based on product characteristics.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "📊 Dashboard",
        "🎯 Product Segmentation",
        "🤖 Recommendation System"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.header("📊 Mobile Product Dashboard")

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Products",
        product_features["model"].nunique()
    )

    col2.metric(
        "Total Brands",
        product_features["brand"].nunique()
    )

    col3.metric(
        "Average Price",
        f"${product_features['price_usd'].mean():,.2f}"
    )

    col4.metric(
        "Average Rating",
        f"{product_features['average_rating'].mean():.2f}/5"
    )

    st.divider()

    # --------------------------------------------------------
    # BRAND ANALYSIS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📱 Products by Brand")

        brand_counts = (
            product_features
            .groupby("brand")["model"]
            .count()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            x=brand_counts.index,
            y=brand_counts.values,
            ax=ax
        )

        ax.set_xlabel("Brand")
        ax.set_ylabel("Number of Products")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.subheader("💰 Average Price by Brand")

        avg_price = (
            product_features
            .groupby("brand")["price_usd"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            x=avg_price.index,
            y=avg_price.values,
            ax=ax
        )

        ax.set_xlabel("Brand")
        ax.set_ylabel("Average Price (USD)")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # RATING ANALYSIS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⭐ Average Rating by Brand")

        avg_rating = (
            product_features
            .groupby("brand")["average_rating"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            x=avg_rating.index,
            y=avg_rating.values,
            ax=ax
        )

        ax.set_xlabel("Brand")
        ax.set_ylabel("Average Rating")
        ax.set_ylim(0, 5)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.subheader("🎯 Product Segment Distribution")

        segment_counts = (
            product_features["segment"]
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            x=segment_counts.values,
            y=segment_counts.index,
            ax=ax
        )

        ax.set_xlabel("Number of Products")
        ax.set_ylabel("Segment")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # PRODUCT TABLE
    # --------------------------------------------------------

    st.subheader("📋 Product Overview")

    display_columns = [
        "brand",
        "model",
        "price_usd",
        "average_rating",
        "overall_quality_score",
        "segment"
    ]

    display_df = product_features[
        display_columns
    ].copy()

    display_df["price_usd"] = (
        display_df["price_usd"].round(2)
    )

    display_df["average_rating"] = (
        display_df["average_rating"].round(2)
    )

    display_df["overall_quality_score"] = (
        display_df["overall_quality_score"].round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PRODUCT SEGMENTATION
# ============================================================

elif page == "🎯 Product Segmentation":

    st.header("🎯 Mobile Product Segmentation")

    st.write(
        """
        Select a mobile product to view its cluster assignment,
        business segment, price, rating, and feature profile.
        """
    )

    # --------------------------------------------------------
    # PRODUCT SELECTION
    # --------------------------------------------------------

    selected_product = st.selectbox(
        "Select Mobile Product",
        product_features["model"].tolist()
    )

    selected_data = product_features[
        product_features["model"] == selected_product
    ]

    product = selected_data.iloc[0]

    # --------------------------------------------------------
    # PRODUCT SUMMARY
    # --------------------------------------------------------

    st.subheader(
        f"📱 {product['brand']} {product['model']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Brand",
        product["brand"]
    )

    col2.metric(
        "Price",
        f"${product['price_usd']:,.2f}"
    )

    col3.metric(
        "Rating",
        f"{product['average_rating']:.2f}/5"
    )

    col4.metric(
        "Segment",
        product["segment"]
    )

    st.divider()

    # --------------------------------------------------------
    # PRODUCT FEATURES
    # --------------------------------------------------------

    st.subheader("📊 Product Feature Profile")

    feature_data = pd.DataFrame(
        {
            "Feature": [
                "Battery",
                "Camera",
                "Performance",
                "Design",
                "Display"
            ],
            "Score": [
                product["battery_score"],
                product["camera_score"],
                product["performance_score"],
                product["design_score"],
                product["display_score"]
            ]
        }
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        data=feature_data,
        x="Feature",
        y="Score",
        ax=ax
    )

    ax.set_ylim(0, 5)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Feature Profile - {product['model']}"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # ADDITIONAL INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("📌 Segment Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Cluster ID:** {int(product['cluster'])}"
        )

        st.write(
            f"**Business Segment:** {product['segment']}"
        )

    with col2:

        st.write(
            f"**Review Count:** "
            f"{int(product['review_count']):,}"
        )

        st.write(
            f"**Helpful Votes:** "
            f"{product['helpful_votes']:.2f}"
        )


# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

elif page == "🤖 Recommendation System":

    st.header(
        "🤖 Mobile Product Recommendation System"
    )

    st.write(
        """
        Select a mobile product and the system will recommend
        products with similar characteristics using
        **Cosine Similarity**.
        """
    )

    # --------------------------------------------------------
    # PRODUCT SELECTION
    # --------------------------------------------------------

    selected_product = st.selectbox(
        "Select Mobile Product",
        product_features["model"].tolist()
    )

    number_of_recommendations = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    # --------------------------------------------------------
    # RECOMMEND BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Find Similar Products",
        use_container_width=True
    ):

        product_index = product_features[
            product_features["model"] == selected_product
        ].index[0]

        similarity_scores = (
            similarity_matrix[product_index]
        )

        sorted_indices = (
            similarity_scores
            .argsort()[::-1]
        )

        # Remove selected product
        sorted_indices = [
            index
            for index in sorted_indices
            if index != product_index
        ]

        top_indices = sorted_indices[
            :number_of_recommendations
        ]

        recommendations = (
            product_features
            .iloc[top_indices]
            [
                [
                    "brand",
                    "model",
                    "price_usd",
                    "average_rating",
                    "overall_quality_score",
                    "segment"
                ]
            ]
            .copy()
        )

        recommendations[
            "similarity_score"
        ] = similarity_scores[top_indices]

        recommendations[
            "similarity_score"
        ] = recommendations[
            "similarity_score"
        ].round(3)

        recommendations[
            "price_usd"
        ] = recommendations[
            "price_usd"
        ].round(2)

        recommendations[
            "average_rating"
        ] = recommendations[
            "average_rating"
        ].round(2)

        recommendations[
            "overall_quality_score"
        ] = recommendations[
            "overall_quality_score"
        ].round(2)

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.success(
            f"Recommendations generated for "
            f"**{selected_product}**"
        )

        st.subheader(
            "📱 Recommended Products"
        )

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # SIMILARITY CHART
        # ----------------------------------------------------

        st.subheader(
            "📊 Recommendation Similarity"
        )

        chart_data = (
            recommendations[
                [
                    "model",
                    "similarity_score"
                ]
            ]
            .set_index("model")
        )

        st.bar_chart(
            chart_data
        )


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

# st.sidebar.divider()

# st.sidebar.markdown(
#     """
#     ### Project Information

#     **Dataset:** Global Mobile Reviews Dataset

#     **Machine Learning:**
#     - K-Means Clustering
#     - Cosine Similarity
#     - PCA

#     **Libraries:**
#     - Pandas
#     - NumPy
#     - Scikit-learn
#     - Matplotlib
#     - Seaborn
#     - Streamlit
#     """
# )

# st.sidebar.caption(
#     "Mobile Product Segmentation & Recommendation System"
# )