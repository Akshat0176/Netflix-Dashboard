# netflix_app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Netflix Global Dashboard", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# -------------------------------
# Utility
# -------------------------------
@st.cache_data
def load_data(filepath: str):
    """Loads, cleans, and prepares the data."""
    df = pd.read_csv(filepath, encoding="latin1")
    for col in ["director", "cast", "country", "rating"]:
        df[col] = df[col].fillna('Unknown')
    
    df['date_added'] = df['date_added'].fillna('Missing Date')
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')
    df['date_added'] = df['date_added'].fillna(pd.to_datetime('1900-01-01'))

    df['duration'] = df['duration'].fillna('Unknown')
    df[['duration_value', 'duration_type']] = df['duration'].str.split(' ', expand=True)
    df['duration_value'] = pd.to_numeric(df['duration_value'], errors='coerce').fillna(0).astype(int)
    df['duration_type'] = df['duration_type'].fillna('Unknown')
    df = df.drop('duration', axis=1)
    
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    df["title"] = df["title"].fillna("Unknown Title")
    
    return df

def explode_col(df, col):
    """Utility to safely split and explode multi-value columns."""
    return df[col].astype(str).str.split(", ").explode().str.strip()

# -------------------------------
# Data & Visual Classes
# -------------------------------
class NetflixData:
    """Manages data loading."""
    def __init__(self, path):
        self.df = load_data(path)
        self.df['duration_value'] = self.df['duration_value'].fillna(0)
        self.df['release_year'] = self.df['release_year'].fillna(self.df['release_year'].median())

class NetflixVisualizer:
    """Manages Plotly visualizations with consistent Streamlit-themed styling."""
    
    def __init__(self, df):
        self.df = df

    def scatter_plot(self, x_col, y_col):
        if self.df.empty:
            st.info("No data to display.")
            return

        fig = px.scatter(
            self.df, x=x_col, y=y_col, color='type', hover_data=['title', 'country'],
            title=f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}',
            template="streamlit", color_discrete_sequence=px.colors.sequential.Reds
        )
        fig.update_layout(plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig, use_container_width=True)

    def histogram(self, col):
        if self.df.empty:
            st.info("No data to display.")
            return 
            
        fig = px.histogram(
            self.df, x=col, color='type',
            title=f'Distribution of {col.replace("_", " ").title()}',
            marginal="box", height=400, template="streamlit",
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        fig.update_layout(plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig, use_container_width=True) 

    def bar_plot(self):
        if self.df.empty:
            st.info("No data to display.")
            return
            
        df_year_count = self.df['release_year'].value_counts().sort_index().reset_index()
        df_year_count.columns = ['Year', 'Count']

        fig = px.bar(
            df_year_count, x='Year', y='Count',
            title='Content Count by Release Year', color='Count',
            color_continuous_scale=px.colors.sequential.Reds, height=400,
            template="streamlit"
        )
        fig.update_layout(
            plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        st.plotly_chart(fig, use_container_width=True)

    def pie_country(self, top_n=10):
        """Creates a Pie Chart for the Top N Content Producing Countries."""
        if self.df.empty:
            st.info("No data to display.")
            return

        countries_counts = explode_col(self.df, "country").value_counts().head(top_n + 1)
        countries_counts = countries_counts[countries_counts.index != 'Unknown'].head(top_n)
        
        fig = px.pie(
            names=countries_counts.index, values=countries_counts.values, 
            hole=0.4, title=f"Top {len(countries_counts)} Countries Distribution", 
            height=400, template="streamlit", color_discrete_sequence=px.colors.sequential.Reds
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig, use_container_width=True)

    def map_view(self):
        # NOTE: This is the sole definition of map_view now.
        if self.df.empty:
            st.info("No data available for map.")
            return

        countries = explode_col(self.df, "country").value_counts().reset_index()
        countries.columns = ["country", "count"]
        countries = countries[countries['country'] != 'Unknown']

        fig = px.choropleth(
            countries, locations="country", locationmode="country names",
            color="count", hover_name="country",
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Global Netflix Content Density by Country",
            labels={'count': 'Number of Titles'}, template="streamlit"
        )

        fig.update_geos(
            showcountries=True, countrycolor="Gray", showocean=True,
            oceancolor="lightblue", showland=True, landcolor="white"
        )

        fig.update_layout(
            height=400, margin={"r": 0, "t": 40, "l": 0, "b": 0},
            coloraxis_colorbar_title="Titles",
            plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    def per_country_detail(self, country_list, key_suffix=""):
        # NOTE: This is the sole definition of per_country_detail now.
        col_s, col_d = st.columns([1, 2])
        
        with col_s:
            selected_country = st.selectbox(
                "Select Country", country_list, key=f"country_detail_select_{key_suffix}"
            )

        dfc = self.df[self.df["country"].str.contains(selected_country, na=False)]
        total = len(dfc)
        
        with col_d:
            st.markdown(f"### Details for: {selected_country}")
            if total == 0:
                st.write("No data available for this country.")
                return
                
            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("Total Titles", total)
                top_genres = explode_col(dfc, "listed_in").value_counts().head(5)
                st.write("**Top 5 Genres:**")
                st.dataframe(top_genres.rename("Count"), use_container_width=True)
                
            with c2:
                top_directors = dfc[dfc['director'] != 'Unknown']['director'].value_counts().head(3)
                st.write("**Top 3 Directors:**")
                st.dataframe(top_directors.rename("Titles"), use_container_width=True)
                
            fig = px.pie(
                dfc, names="type", hole=0.4, 
                title=f"Type Ratio in {selected_country}", template="streamlit",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
            st.plotly_chart(fig, use_container_width=True)

class PredictionPage:
    """Manages the Machine Learning Prediction page."""
    
    def __init__(self, data_df):
        self.df = data_df

    def plot_regression_performance(self, y_test, y_pred, target_col):
        """Plots Predicted vs Actual values."""
        df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_results['Actual'], y=df_results['Predicted'], mode='markers',
            name='Predictions', marker=dict(opacity=0.5)
        ))
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], mode='lines',
            name='Ideal Prediction (y=x)', line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Actual vs. Predicted {target_col.replace("_", " ").title()}',
            xaxis_title=f'Actual {target_col.replace("_", " ").title()}',
            yaxis_title=f'Predicted {target_col.replace("_", " ").title()}',
            height=500, showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def run_prediction(self):
        st.title(" Predict Content Trends with Regression")
        st.markdown("Use Regression to predict content attributes based on existing data.")

        numerical_cols = ['release_year', 'duration_value']
        categorical_cols = ['type', 'rating', 'country']
        
        df_model = self.df.drop(columns=['show_id', 'title', 'director', 'cast', 'date_added', 'listed_in', 'description', 'duration_type'], errors='ignore')
        
        target_col = st.selectbox("Select Target Variable (What to Predict)", numerical_cols, index=0)
        
        model_features = [col for col in df_model.columns if col != target_col]

        for col in categorical_cols:
            if col in model_features:
                top_categories = df_model[col].value_counts().head(100).index
                df_model[col] = df_model[col].apply(lambda x: x if x in top_categories else 'Other')
        
        df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.drop(columns=[target_col], errors='ignore')
        y = df_encoded[target_col]

        if X.empty or y.empty:
            st.warning("Data cleaning failed to produce valid features or target.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.info(f"Training model to predict **{target_col}** using **{X_train.shape[1]}** features. Test size is 20%.")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Model Performance Scores")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        col_m2.metric("R-squared (R²)", f"{r2:.2f}")

        st.markdown("---")

        st.markdown(f"**Mean Absolute Error (MAE):** **{mae:.2f}**")
        st.markdown(f"**R-squared (R²):** **{r2:.2f}**")
        st.markdown("---")

        st.subheader("Actual vs. Predicted Values Plot")
        self.plot_regression_performance(y_test, y_pred, target_col)


# -------------------------------
# App Pages
# -------------------------------
def dashboard_page(data_instance):
    """The main visualization page, divided into two navigable tabs."""
    
    dff = data_instance.df
    vis = NetflixVisualizer(dff)

    st.header("📊Overview")
    
    total_records = len(dff)
    top_country = explode_col(dff, "country").value_counts().idxmax()
    top_genre = explode_col(dff, "listed_in").value_counts().idxmax()
    top_type = explode_col(dff, "type").value_counts().idxmax() 
        
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(label="Total Titles", value=f"{total_records:,}")
    col_b.metric(label="Top Content Country", value=top_country)
    col_c.metric(label="Top Genre", value=top_genre)
    col_d.metric(label="Top Content Type", value=top_type)
    
    st.markdown("---")

    tab1, tab2 = st.tabs(["Visualizations", "📍 Country Deep Dive"])

    # =================================================================
    # TAB 1: GLOBAL VISUALIZATIONS (GRAPHS)
    # =================================================================
    with tab1:
        
        st.subheader("Map of Content Distribution")
        st.markdown("Global Netflix Content Density by Country")
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            vis.map_view()
        
        with col_vis2:
            st.subheader("Content Count by Release Year")
            vis.bar_plot()
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col_vis3, col_vis4, col_vis5 = st.columns(3)
        numerical_cols = ['release_year', 'duration_value']
        
        with col_vis3:
            st.markdown("##### Top Countries Distribution")
            vis.pie_country(top_n=10)
            
        with col_vis4:
            st.markdown("##### Content Distribution")
            hist_col = st.selectbox("Select Histogram Axis", numerical_cols, index=0, key="hist_select_tab1")
            vis.histogram(hist_col)
            
        with col_vis5:
            st.markdown("#### Scatter Plots: Explore Relationships")
            scatter_x = st.selectbox("Scatter X-Axis", numerical_cols, index=0, key="scatter_x_select_tab1")
            scatter_y = st.selectbox("Scatter Y-Axis", numerical_cols, index=1, key="scatter_y_select_tab1")
            vis.scatter_plot(scatter_x, scatter_y)

    # =================================================================
    # TAB 2: COUNTRY DETAIL VIEW
    # =================================================================
    with tab2:
        st.markdown("### Content Statistics by Country in detail")
        country_list = [c for c in explode_col(dff, "country").dropna().unique().tolist() if c != "All" and c != "Unknown"]
        vis.per_country_detail(country_list, key_suffix="tab2")


# -------------------------------
# Main App Structure
# -------------------------------
class NetflixApp:
    def __init__(self, path):
        self.data = NetflixData(path)
        self.pages = {
            "Dashboard": dashboard_page,
            "Prediction": PredictionPage 
        }

    def header(self):
        """Displays the application header."""
        images_loaded = False 
        try: 
            logo = Image.open("Netflix-logo.jpg")
            icon = Image.open("film Background Removed.png")
            images_loaded = True
        except FileNotFoundError:
            st.warning("Image files not found!")

        col1, col2, col3 = st.columns([2, 6, 2])
        with col1:
            if images_loaded:
                st.image(logo, width=180)
        with col2:
            st.markdown("<h1 style='text-align:center;'>🌍 Netflix Global Insights</h1>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Data Visualization Project</h4>", unsafe_allow_html=True)
        with col3:
            if images_loaded:
                st.image(icon, width=100)

    def run(self):
        """Runs the main application logic with sidebar navigation."""
        self.header()

        st.sidebar.header("Navigation")
        
        for page_name in self.pages.keys():
            is_active = st.session_state.page == page_name
            
            if st.sidebar.button(page_name, use_container_width=True, help=f"Go to {page_name} Page"):
                st.session_state.page = page_name
        
        if st.session_state.page == "Dashboard":
            dashboard_page(self.data)
        elif st.session_state.page == "Prediction":
            prediction_app = self.pages["Prediction"](self.data.df)
            prediction_app.run_prediction()


# Run
if __name__ == "__main__":
    app = NetflixApp("netflix_movies.csv")
    app.run()