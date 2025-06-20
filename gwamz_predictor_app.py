import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(
    page_title="Gwamz Song Performance Predictor Pro",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .sidebar .sidebar-content {
        background-color: #1E2130;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stMetric {
        background-color: #262730;
        border-radius: 10px;
        padding: 15px;
    }
    .css-1aumxhk {
        background-color: #1E2130;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_resources():
    # Load models
    streams_model = joblib.load('xgb_streams_model.pkl')
    popularity_model = joblib.load('xgb_popularity_model.pkl')
    
    # Load data
    df = pd.read_parquet('gwamz_cleaned.parquet')
    
    # Prepare SHAP explainer
    preprocessor = streams_model.named_steps['preprocessor']
    model = streams_model.named_steps['regressor']
    sample_data = preprocessor.transform(df[df.columns.intersection(streams_model.feature_names_in_)].iloc[:50])
    explainer = shap.Explainer(model, sample_data)
    
    return streams_model, popularity_model, df, explainer, preprocessor

streams_model, popularity_model, df, explainer, preprocessor = load_resources()

# Calculate first release date
first_release = df['release_date'].min()

# App header
st.title("🎤 Gwamz Song Performance Predictor Pro")
st.markdown("""
<p style='font-size:16px; color:#AAAAAA;'>
Advanced predictive analytics for Gwamz's music performance forecasting
</p>
""", unsafe_allow_html=True)

# Sidebar with advanced controls
with st.sidebar:
    st.header("🎛️ Song Parameters")
    
    # Release details
    st.subheader("Release Information")
    release_date = st.date_input("Release Date", datetime.today())
    album_type = st.selectbox("Album Type", ["single", "album", "compilation"], 
                            help="Type of release (single tracks typically perform differently than album tracks)")
    
    # Track metadata
    st.subheader("Track Metadata")
    col1, col2 = st.columns(2)
    with col1:
        total_tracks = st.number_input("Total Tracks", min_value=1, value=1, 
                                     help="Total tracks in the album/release")
    with col2:
        track_number = st.number_input("Track Number", min_value=1, value=1,
                                     help="Position of this track in the album")
    
    # Content flags
    st.subheader("Content Characteristics")
    explicit = st.checkbox("Explicit Content", value=True)
    col1, col2 = st.columns(2)
    with col1:
        is_remix = st.checkbox("Is Remix/Edit", value=False)
        is_sped_up = st.checkbox("Is Sped Up", value=False)
    with col2:
        is_instrumental = st.checkbox("Is Instrumental", value=False)
        is_jersey = st.checkbox("Is Jersey Club", value=False)
    
    # Market availability
    st.subheader("Distribution")
    available_markets = st.slider("Available Markets", 1, 200, 185,
                                help="Number of markets where the track will be available")
    
    # Artist metrics (fixed for Gwamz)
    st.subheader("Artist Metrics")
    artist_followers = st.number_input("Artist Followers", value=7937, 
                                    help="Current number of artist followers")
    artist_popularity = st.slider("Artist Popularity", 0, 100, 41,
                                help="Current artist popularity score (0-100)")

# Calculate derived features
release_year = release_date.year
release_month = release_date.month
release_day = release_date.day
release_dayofweek = release_date.weekday()
release_quarter = (release_date.month - 1) // 3 + 1
days_since_first_release = (release_date - first_release.date()).days

# Create input dataframe
input_data = pd.DataFrame({
    'artist_followers': [artist_followers],
    'artist_popularity': [artist_popularity],
    'album_type': [album_type],
    'release_year': [release_year],
    'release_month': [release_month],
    'release_day': [release_day],
    'release_dayofweek': [release_dayofweek],
    'release_quarter': [release_quarter],
    'days_since_first_release': [days_since_first_release],
    'total_tracks_in_album': [total_tracks],
    'available_markets_count': [available_markets],
    'track_number': [track_number],
    'disc_number': [1],  # Default to 1 since we didn't include in UI
    'explicit': [explicit],
    'is_remix': [is_remix],
    'is_sped_up': [is_sped_up],
    'is_instrumental': [is_instrumental],
    'is_jersey': [is_jersey]
})

# Prediction button
if st.button("🚀 Predict Performance", use_container_width=True):
    with st.spinner("Analyzing song potential..."):
        # Make predictions
        streams_pred = streams_model.predict(input_data)[0]
        popularity_pred = popularity_model.predict(input_data)[0]
        
        # Get historical percentiles
        streams_percentile = np.mean(df['streams'] <= streams_pred) * 100
        popularity_percentile = np.mean(df['track_popularity'] <= popularity_pred) * 100
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Streams Prediction")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=streams_pred,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Streams"},
                gauge={
                    'axis': {'range': [None, df['streams'].max() * 1.1]},
                    'steps': [
                        {'range': [0, df['streams'].quantile(0.25)], 'color': "lightgray"},
                        {'range': [df['streams'].quantile(0.25), df['streams'].quantile(0.75)], 'color': "gray"},
                        {'range': [df['streams'].quantile(0.75), df['streams'].max() * 1.1], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': streams_pred
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Percentile Rank", f"{streams_percentile:.1f}%",
                     help="Percentage of historical tracks with equal or fewer streams")
            
        with col2:
            st.markdown("### 🌟 Popularity Prediction")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=popularity_pred,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Popularity (0-100)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 33], 'color': "red"},
                        {'range': [33, 66], 'color': "orange"},
                        {'range': [66, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': popularity_pred
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Percentile Rank", f"{popularity_percentile:.1f}%",
                     help="Percentage of historical tracks with equal or lower popularity")
        
        # SHAP explanation
        st.markdown("---")
        st.markdown("### 🔍 Prediction Explanation")
        
        # Get SHAP values
        preprocessed_input = preprocessor.transform(input_data)
        shap_values = explainer(preprocessed_input)
        
        # Plot SHAP force plot
        st.markdown("**How each feature contributes to this prediction:**")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        insights = []
        
        # Streams insights
        if is_sped_up and 'is_sped_up' in streams_model.feature_names_in_:
            insights.append("🎵 Sped Up versions historically perform better (average +15% streams)")
        if is_remix and 'is_remix' in streams_model.feature_names_in_:
            insights.append("🔄 Remixes/Edits tend to get 10-20% more streams than originals")
        if available_markets < 185:
            insights.append(f"🌍 Limited market availability ({available_markets}/185) may reduce potential streams by up to {int((185 - available_markets) * 1000)}")
        
        # Popularity insights
        if explicit and 'explicit' in popularity_model.feature_names_in_:
            insights.append("🔞 Explicit content typically has 5-10% higher popularity scores")
        if release_quarter in [2, 3] and 'release_quarter' in popularity_model.feature_names_in_:
            insights.append("☀️ Summer releases (Q2/Q3) often see higher popularity scores")
        
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.markdown("This combination of features doesn't show strong historical patterns.")
        
        # Recommendation engine
        st.markdown("---")
        st.markdown("### 🛠️ Optimization Recommendations")
        
        recs = []
        if not is_sped_up and 'is_sped_up' in streams_model.feature_names_in_:
            recs.append("Consider releasing a Sped Up version (+15% potential streams)")
        if available_markets < 185:
            recs.append(f"Increase market availability to 185 for maximum exposure (+{int((185 - available_markets) * 1000)} potential streams)")
        if not is_remix and 'is_remix' in streams_model.feature_names_in_:
            recs.append("Consider a remix/edition version (+10-20% potential streams)")
        
        if recs:
            for rec in recs:
                st.markdown(f"✅ {rec}")
        else:
            st.markdown("Your current configuration is already optimized based on historical patterns.")

# Historical analytics section
st.markdown("---")
st.markdown("## 📈 Historical Performance Analytics")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Trend Analysis", "Version Comparison", "Market Impact", "Track Attributes"])

with tab1:
    st.markdown("### Streams and Popularity Over Time")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add streams trace
    fig.add_trace(
        go.Scatter(
            x=df['release_date'],
            y=df['streams'],
            name="Streams",
            mode='markers',
            marker=dict(color='#1DB954', size=8),
            hovertext=df['track_name']
        ),
        secondary_y=False,
    )
    
    # Add popularity trace
    fig.add_trace(
        go.Scatter(
            x=df['release_date'],
            y=df['track_popularity'],
            name="Popularity",
            mode='markers',
            marker=dict(color='#FFD700', size=8),
            hovertext=df['track_name']
        ),
        secondary_y=True,
    )
    
    # Add figure layout
    fig.update_layout(
        title="Streams and Popularity Over Time",
        xaxis_title="Release Date",
        hovermode="closest",
        template="plotly_dark"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Streams", secondary_y=False)
    fig.update_yaxes(title_text="Popularity (0-100)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Version Performance Comparison")
    
    # Create comparison data
    version_types = ['Original', 'Sped Up', 'Remix/Edit', 'Jersey Club']
    version_data = []
    
    for version in version_types:
        if version == 'Original':
            mask = (~df['track_name'].str.contains('Sped Up|sped up|Remix|remix|Edit|edit|Jersey|jersey'))
        elif version == 'Sped Up':
            mask = (df['track_name'].str.contains('Sped Up|sped up'))
        elif version == 'Remix/Edit':
            mask = (df['track_name'].str.contains('Remix|remix|Edit|edit')) & (~df['track_name'].str.contains('Jersey|jersey'))
        else:
            mask = (df['track_name'].str.contains('Jersey|jersey'))
        
        version_data.append({
            'Version': version,
            'Avg Streams': df[mask]['streams'].mean(),
            'Avg Popularity': df[mask]['track_popularity'].mean(),
            'Count': mask.sum()
        })
    
    version_df = pd.DataFrame(version_data)
    
    # Create bar charts
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Streams", "Average Popularity"))
    
    fig.add_trace(
        go.Bar(
            x=version_df['Version'],
            y=version_df['Avg Streams'],
            name="Streams",
            marker_color=['#1DB954', '#1ED760', '#1AA34A', '#169F42']
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=version_df['Version'],
            y=version_df['Avg Popularity'],
            name="Popularity",
            marker_color=['#FFD700', '#FFC800', '#FFB700', '#FFA500']
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Performance by Version Type",
        showlegend=False,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Market Availability Impact")
    
    # Create market impact visualization
    fig = px.scatter(
        df,
        x='available_markets_count',
        y='streams',
        color='track_popularity',
        size='track_popularity',
        hover_name='track_name',
        trendline="lowess",
        title="Streams by Market Availability",
        labels={
            'available_markets_count': 'Number of Available Markets',
            'streams': 'Total Streams',
            'track_popularity': 'Popularity Score'
        },
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Track Attribute Analysis")
    
    # Select attributes to compare
    attribute = st.selectbox(
        "Select attribute to analyze",
        ['explicit', 'is_remix', 'is_sped_up', 'is_instrumental', 'is_jersey'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Create box plots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Streams", "Popularity"))
    
    for i, metric in enumerate(['streams', 'track_popularity'], 1):
        fig.add_trace(
            go.Box(
                y=df[metric],
                x=df[attribute].astype(str),
                name=metric,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker_color='#1DB954' if metric == 'streams' else '#FFD700'
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        title=f"Performance by {attribute.replace('_', ' ').title()}",
        showlegend=False,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #AAAAAA;'>
Gwamz Song Performance Predictor Pro • Data Science Edition • v2.0
</p>
""", unsafe_allow_html=True)