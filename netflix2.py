# netflix_dashboard_advanced.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Netflix Global Dashboard", layout="wide")

# -------------------------------
# Utility
# -------------------------------
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="latin1")
    for col in ["type", "country", "listed_in", "release_year", "title"]:
        if col not in df.columns:
            df[col] = np.nan
    df["country"] = df["country"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    df["type"] = df["type"].fillna("Unknown")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    df["title"] = df["title"].fillna("Unknown Title")
    return df

def explode_col(df, col):
    return df[col].astype(str).str.split(", ").explode().str.strip()

# -------------------------------
# Data & Visual Classes
# -------------------------------
class NetflixData:
    def __init__(self, path):
        self.df = load_data(path)

    def filter_df(self, types, countries, genres, years):
        dff = self.df.copy()
        if "All" not in types:
            dff = dff[dff["type"].isin(types)]
        if "All" not in countries:
            mask = dff["country"].apply(lambda c: any(x.strip() in countries for x in str(c).split(",")))
            dff = dff[mask]
        if "All" not in genres:
            mask_g = dff["listed_in"].apply(lambda c: any(x.strip() in genres for x in str(c).split(",")))
            dff = dff[mask_g]
        dff = dff[dff["release_year"].between(years[0], years[1])]
        return dff

    def get_year_range(self):
        y = self.df["release_year"].dropna()
        return (int(y.min()), int(y.max()))

class NetflixVisualizer:
    def __init__(self, df):
        self.df = df

    def metric_panel(self, metric):
        col1, col2, col3, col4 = st.columns(4)
        total = len(self.df)
        if total == 0:
            for c in [col1, col2, col3, col4]:
                c.metric("No data", "—")
            return

        top_country = explode_col(self.df, "country").value_counts().idxmax()
        top_genre = explode_col(self.df, "listed_in").value_counts().idxmax()
        avg_year = int(self.df["release_year"].dropna().mean()) if not self.df["release_year"].dropna().empty else 0
        unique_genres = explode_col(self.df, "listed_in").nunique()
        num_countries = explode_col(self.df, "country").nunique()

        if metric == "Total Titles":
            val = total
        elif metric == "Average Release Year":
            val = avg_year
        elif metric == "Number of Unique Genres":
            val = unique_genres
        else:
            val = num_countries

        col1.metric("Selected Metric", f"{metric}")
        col2.metric("Value", f"{val}")
        col3.metric("Top Country", top_country)
        col4.metric("Top Genre", top_genre)

    def pie_type(self):
        if self.df.empty:
            st.info("No data to display.")
            return
        fig = px.pie(self.df, names="type", hole=0.4, title="Movies vs TV Shows")
        st.plotly_chart(fig, use_container_width=True)

    def country_pie(self, top_n=10):
        countries = explode_col(self.df, "country").value_counts().head(top_n)
        fig = px.pie(names=countries.index, values=countries.values,
                     title=f"Top {top_n} Countries by Share of Netflix Titles")
        st.plotly_chart(fig, use_container_width=True)

    def map_view(self):
        if self.df.empty:
            st.info("No data available for map.")
            return
        countries = explode_col(self.df, "country").value_counts().reset_index()
        countries.columns = ["country", "count"]
        fig = px.choropleth(
            countries,
            locations="country",
            locationmode="country names",
            color="count",
            color_continuous_scale="Reds",
            title="Global Netflix Content Density by Country",
        )
        st.plotly_chart(fig, use_container_width=True)

    def per_country_panel(self, country):
        st.subheader(f"📍 Country Detail: {country}")
        dfc = self.df[self.df["country"].str.contains(country, na=False)]
        total = len(dfc)
        if total == 0:
            st.write("No data available for this country.")
            return
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Total Titles", total)
            top_genres = explode_col(dfc, "listed_in").value_counts().head(5)
            st.write("**Top 5 Genres:**")
            st.dataframe(top_genres.rename("Count"))
        with c2:
            fig = px.pie(dfc, names="type", hole=0.4, title=f"Type Ratio in {country}")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# App
# -------------------------------
class NetflixApp:
    def __init__(self, path):
        self.data = NetflixData(path)

    def sidebar(self):
        st.sidebar.header("Filters")
        df = self.data.df
        types = ["All"] + sorted(df["type"].dropna().unique().tolist())
        countries = ["All"] + explode_col(df, "country").dropna().unique().tolist()
        genres = ["All"] + explode_col(df, "listed_in").dropna().unique().tolist()
        y_min, y_max = self.data.get_year_range()

        selected_types = st.sidebar.multiselect("Type", types, default=["All"])
        selected_countries = st.sidebar.multiselect("Countries", countries[:50], default=["All"])
        selected_genres = st.sidebar.multiselect("Genres", genres[:50], default=["All"])
        selected_years = st.sidebar.slider("Year Range", y_min, y_max, (y_min, y_max))
        top_n = st.sidebar.slider("Top N for charts", 5, 25, 10)
        metric = st.sidebar.selectbox("Select Metric to Compare", 
                                      ["Total Titles", "Average Release Year", 
                                       "Number of Unique Genres", "Number of Countries Represented"])
        selected_country_detail = st.sidebar.selectbox("Country Detail View", countries[1:50])
        return selected_types, selected_countries, selected_genres, selected_years, top_n, metric, selected_country_detail

    def header(self):
        col1, col2, col3 = st.columns([2, 6, 2])
        try:
            logo = Image.open("Netflix-Logo.jpg")
            icon = Image.open("film Background Removed.png")
            with col1:
                st.image(logo, width=180)
            with col2:
                st.markdown("<h1 style='text-align:center;'>🌍 Netflix Global Insights</h1>", unsafe_allow_html=True)
                st.markdown("<h4 style='text-align:center;'>Data Visualization Project</h4>", unsafe_allow_html=True)
            with col3:
                st.image(icon, width=100)
        except:
            st.markdown("<h1 style='text-align:center;'>🌍 Netflix Global Insights</h1>", unsafe_allow_html=True)

    def run(self):
        self.header()
        types, countries, genres, years, top_n, metric, country_detail = self.sidebar()
        dff = self.data.filter_df(types, countries, genres, years)
        vis = NetflixVisualizer(dff)

        # Metric Panel
        st.header("📊 Overview")
        vis.metric_panel(metric)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Graph Tabs
        st.header("🎬 Visual Insights")
        tab1, tab2, tab3, tab4 = st.tabs(["Type Ratio", "Country Share (Pie)", "Map View", "Per-Country Detail"])
        with tab1:
            vis.pie_type()
        with tab2:
            vis.country_pie(top_n=top_n)
        with tab3:
            vis.map_view()
        with tab4:
            vis.per_country_panel(country_detail)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app = NetflixApp("netflix_movies.csv")
    app.run()
