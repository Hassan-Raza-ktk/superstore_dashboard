# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from streamlit_toggle import st_toggle_switch
from PIL import Image



def custom_plotly_theme(fig, title=""):
    fig.update_layout(
        title=title,
        paper_bgcolor="#F9F9F9",   # Full background
        plot_bgcolor="#FFFFFF",   # Inside chart area
        font=dict(
            family="Arial",
            color="#333333"
        ),
        title_font=dict(size=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📊 Superstore Dashboard",
    page_icon="🛒",
    layout="wide",
)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------SideBar Beatification----------------

st.markdown("""
<style>
/* Reduce top space in sidebar */
section[data-testid="stSidebar"] .css-1d391kg {
    padding-top: 1rem;
}

/* Shrink width of sidebar */
[data-testid="stSidebar"] {
    width: 230px !important;
    background: linear-gradient(145deg, #1f2c47, #101820);
    color: white;
}

/* Make text and widgets visible */
[data-testid="stSidebar"] .css-1cypcdb, /* title text */
[data-testid="stSidebar"] .css-1v3fvcr, /* radio/select text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stButton>button {
    color: white !important;
}

/* Sidebar button styling */
[data-testid="stSidebar"] .stButton>button {
    background-color: #2a3d5a;
    border-radius: 8px;
    border: none;
    transition: 0.3s;
}

[data-testid="stSidebar"] .stButton>button:hover {
    background-color: #3f5473;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)



# --- CUSTOM STYLING ---
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .css-18e3th9 {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        }
        h1, h2, h3 {
            color: #1f4e79;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---

with st.sidebar:

    logo = Image.open("images\logo.png")
    st.image(logo, width=120, caption="", output_format="PNG")

    st.markdown("""
    <style>
        [data-testid="stSidebar"] img {
            border-radius: 50%;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

    # Dropdown or Navigation
page = st.sidebar.radio("📂 Navigate to:", [
     "Introduction","Overview", "Trends", "Category", "Region", "Segment", "Forecasting", "Clustering"
])

# Report Button
if st.sidebar.button("📄 View Report"):
    page = "Report"



# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/Superstore.csv", encoding='ISO-8859-1', parse_dates=['Order Date'])
    return df

df = load_data()




# ---------------------------- OVERVIEW METRICS -------------------------------
if page == "Overview":
    st.title("📊 Superstore Overview")
    # Basic KPIs
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = df['Order ID'].nunique()
    total_quantity = df['Quantity'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📦 Total Orders", f"{total_orders:,}")
    col3.metric("🛍 Total Quantity", f"{total_quantity:,}")
    col4.metric("📈 Total Profit", f"${total_profit:,.0f}")

    st.markdown("---")

    # Monthly Sales Trend
    monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()

    fig = px.line(
        monthly_sales,
        x='Order Date',
        y='Sales',
        title="📅 Monthly Sales Trend",
        markers=True
    )

    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title_font=dict(size=20, color='lightgray'),
        xaxis=dict(showgrid=True, gridcolor='#444'),
        yaxis=dict(showgrid=True, gridcolor='#444'),
        hoverlabel=dict(bgcolor='black', font_size=14, font_family="Arial")
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------- TRENDS PAGE -------------------------------

elif page == "Trends":
    st.title("📈 Monthly Trends")
    st.markdown("#### Analyze how key metrics change over time")

    # Monthly aggregation
    df['Order Month'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby('Order Month')[['Sales', 'Profit', 'Discount']].sum().reset_index()
    df['Year'] = df['Order Date'].dt.year
    # Melt for multi-line chart
    monthly_melted = monthly.melt(id_vars='Order Month', value_vars=['Sales', 'Profit', 'Discount'],
                            var_name='Metric', value_name='Value')

    fig = px.line(monthly_melted, x='Order Month', y='Value', color='Metric',
                title="📊 Sales, Profit & Discount Trends Over the Years", markers=True)

    # Dark theme styling
    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title_font=dict(size=20, color='lightgray'),
        xaxis=dict(showgrid=True, gridcolor='#444'),
        yaxis=dict(showgrid=True, gridcolor='#444'),
        hoverlabel=dict(bgcolor='black', font_size=14)
    )

    st.plotly_chart(fig, use_container_width=True)


    
    # --- Year Selection Box ---
    unique_years = sorted(df['Year'].unique())
    selected_year = st.selectbox("📅 Select Year to Filter", unique_years)

    # Filter data based on selected year
    filtered_df = df[df['Year'] == selected_year]

    # Monthly aggregation (filtered)
    monthly = filtered_df.groupby('Order Month')[['Sales', 'Profit', 'Discount']].sum().reset_index()

    # Melt for multi-line chart
    monthly_melted = monthly.melt(
        id_vars='Order Month',
        value_vars=['Sales', 'Profit', 'Discount'],
        var_name='Metric', value_name='Value'
    )

    fig = px.line(
        monthly_melted,
        x='Order Month', y='Value',
        color='Metric',
        title=f"📊 Monthly Sales, Profit & Discount Trends — {selected_year}",
        markers=True
    )

    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title_font=dict(size=20, color='lightgray'),
        xaxis=dict(showgrid=True, gridcolor='#444'),
        yaxis=dict(showgrid=True, gridcolor='#444'),
        hoverlabel=dict(bgcolor='black', font_size=14)
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------- Category PAGE -------------------------------

elif page == "Category":
    st.title("📦 Category Analysis")
    st.markdown("### Explore sales and profit across different product categories")

    # Group by Category
    cat_summary = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()

    # Create 2 columns
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(cat_summary, x='Category', y='Sales', color='Category',
                    title='💸 Total Sales by Category', text_auto=True)
        fig1.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            title_font=dict(size=18, color='lightgray'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
        )
        st.plotly_chart(fig1, use_container_width=False, height=200, width=400)
 

    with col2:
        fig2 = px.bar(cat_summary, x='Category', y='Profit', color='Category',
                    title='💰 Total Profit by Category', text_auto=True)
        fig2.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            title_font=dict(size=18, color='lightgray'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
        )
        st.plotly_chart(fig2, use_container_width=False, height=200, width=400)


    # Sub-category Table
    st.markdown("### 📋 Sub-Category Level Summary")
    subcat_summary = df.groupby(['Category', 'Sub-Category'])[['Sales', 'Profit']].sum().reset_index()
    st.dataframe(subcat_summary.sort_values(by='Sales', ascending=False), use_container_width=True)



# ---------------------------- Region PAGE -------------------------------

elif page == "Region":
    st.title("📍 Regional Performance")
    st.markdown("### Evaluate sales and profitability across different regions")

    # Region Summary
    region_summary = df.groupby('Region')[['Sales', 'Profit']].sum().reset_index()

    # Layout: Two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(region_summary, x='Region', y='Sales', color='Region',
                    title='🗺️ Total Sales by Region', text_auto=True)
        fig1.update_layout(
            height=400, width=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            title_font=dict(size=18, color='lightgray'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
        )
        st.plotly_chart(fig1, use_container_width=False)

    with col2:
        fig2 = px.bar(region_summary, x='Region', y='Profit', color='Region',
                    title='💰 Total Profit by Region', text_auto=True)
        fig2.update_layout(
            height=400, width=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            title_font=dict(size=18, color='lightgray'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
        )
        st.plotly_chart(fig2, use_container_width=False)

    # Region Summary Table
    st.markdown("### 📋 Region-wise Summary")
    st.dataframe(region_summary.sort_values(by='Sales', ascending=False), use_container_width=True)


# ---------------------------- Segment PAGE -------------------------------

elif page == "Segment":
    st.title("👥 Segment Analysis")
    st.markdown("### Understand performance across customer segments")

    # Segment Summary
    segment_summary = df.groupby('Segment')[['Sales', 'Profit']].sum().reset_index()

    # Create 2 columns for charts
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(segment_summary, x='Segment', y='Sales', color='Segment',
                    title='📊 Sales by Segment', text_auto=True)
        fig1.update_layout(
            height=400, width=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            title_font=dict(size=18, color='lightgray'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
        )
        st.plotly_chart(fig1, use_container_width=False)

    with col2:
        fig2 = px.bar(segment_summary, x='Segment', y='Profit', color='Segment',
                    title='💰 Profit by Segment', text_auto=True)
        fig2.update_layout(
            height=400, width=500,
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            title_font=dict(size=18, color='lightgray'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#444'),
        )
        st.plotly_chart(fig2, use_container_width=False)

    # Segment Summary Table
    st.markdown("### 📋 Segment-wise Summary")
    st.dataframe(segment_summary.sort_values(by='Sales', ascending=False), use_container_width=True)



# ---------------------------- Forecast PAGE -------------------------------

elif page == "Forecasting":
    st.title("🔮 Sales Forecasting")
    st.markdown("### Prophet-based forecast for future sales")

    # Load forecast data
    forecast_data = pd.read_csv("data/forecast_sales.csv")

    # Plot forecast with confidence interval
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_data['ds'], y=forecast_data['yhat'],
        mode='lines', name='Forecast (yhat)', line=dict(color='cyan')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_data['ds'], y=forecast_data['yhat_upper'],
        mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_data['ds'], y=forecast_data['yhat_lower'],
        mode='lines', name='Lower Bound', fill='tonexty',
        fillcolor='rgba(0, 255, 255, 0.2)', line=dict(width=0), showlegend=True
    ))

    fig.update_layout(
        title='📈 Forecasted Sales with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Sales',
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show forecast table (last 5 future points)
    st.markdown("### 📋 Forecast Table (Last 5 months)")
    st.dataframe(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5).rename(
        columns={
            'ds': 'Date',
            'yhat': 'Forecast',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        }
    ), use_container_width=True)



# ---------------------------- Cluster PAGE -------------------------------

elif page == "Clustering":
    st.title("👥 Clustering Analysis")
    st.markdown("### K-Means clustering based on sales metrics")

    # Load clustering data
    cluster_df = pd.read_csv("data/clustered_data.csv", encoding='ISO-8859-1')

    # Scatter plot: Sales vs Profit (colored by Cluster)
    fig = px.scatter(
        cluster_df,
        x='Sales', y='Profit',
        color='Cluster',
        title='Clustered Sales vs Profit',
        template='plotly_dark',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 1: Create cluster summary
    summary_df = cluster_df.groupby('Cluster').agg({
        'Sales': 'mean',
        'Profit': 'mean',
        'Discount': 'mean',
        'Quantity': 'mean'
    }).round(2).reset_index()

    # Step 2: Add label column manually (based on analysis)
    cluster_labels = {
        0: "🧾 Regular Customers",
        1: "🏆 VIPs — High Value Clients",
        2: "🎯 Medium Segment — Stable",
        3: "⚠️ Risky — Discount Loss Group"
    }

    summary_df["Interpretation"] = summary_df["Cluster"].map(cluster_labels)

    # Step 3: Display table
    st.markdown("### 📋 Cluster Summary Stats")
    st.dataframe(summary_df, use_container_width=True)


# -----------------------------Report-------------------------

elif page == "Report":
    st.title("📄 Analytical Report")

    st.markdown("""
    <style>
        .section-blue {
            color: #1f4e79;
            font-size: 22px;
            font-weight: 600;
            margin: 25px 0 10px;
        }
        .section-green {
            color: #2e7d32;
            font-size: 22px;
            font-weight: 600;
            margin: 25px 0 10px;
        }
        .section-gold {
            color: #e6b800;
            font-size: 22px;
            font-weight: 600;
            margin: 25px 0 10px;
        }
        .section-purple {
            color: #6a1b9a;
            font-size: 22px;
            font-weight: 600;
            margin: 25px 0 10px;
        }
        .section-red {
            color: #c62828;
            font-size: 22px;
            font-weight: 600;
            margin: 25px 0 10px;
        }
        .bullet {
            margin-left: 20px;
            font-size: 16px;
        }
    </style>

    <div class='section-blue'>🧹 Data Cleaning & Preparation</div>
    <div class='bullet'>• Removed missing/inconsistent values</div>
    <div class='bullet'>• Converted 'Order Date' to datetime</div>
    <div class='bullet'>• Extracted Year/Month for trends</div>
    <div class='bullet'>• Normalized features for clustering</div>

    <div class='section-green'>📊 Exploratory Data Analysis (EDA)</div>
    <div class='bullet'>• Sales & Profit highest in <b>West</b> & <b>East</b></div>
    <div class='bullet'>• <b>Technology</b> category leads in revenue</div>
    <div class='bullet'>• <b>Consumer</b> segment is top buyer group</div>
    <div class='bullet'>• Q4 (Oct–Dec) shows seasonal sales boost</div>

    <div class='section-gold'>🔮 Forecasting (Prophet)</div>
    <div class='bullet'>• Model shows <b>upward trend</b> in future sales</div>
    <div class='bullet'>• Confidence bands guide risk management</div>

    <div class='section-purple'>👥 Clustering (K-Means)</div>
    <div class='bullet'>• Created 4 customer clusters:</div>
    <div class='bullet'> 🏆 VIP Clients — high profit</div>
    <div class='bullet'> 🧾 Regulars — average but stable</div>
    <div class='bullet'> 🎯 Mid-Segment — good potential</div>
    <div class='bullet'> ⚠️ Discount Risk — low profit, high discount</div>

    <div class='section-red'>💡 Key Insights</div>
    <div class='bullet'>• Push campaigns in Q4 for max ROI</div>
    <div class='bullet'>• Reduce discounts for Cluster 3</div>
    <div class='bullet'>• Focus on <b>VIPs</b> for loyalty & upselling</div>
    <div class='bullet'>• Expand high-performing categories</div>
    """, unsafe_allow_html=True)

# -----------------------Introduction-----------------------

elif page == "Introduction":
    st.title("👋 Welcome to the Dashboard")

    st.markdown("""
    ### Created by: Hassan Raza  
    **Role:** Data Analyst

    ---
    ### About Dashbaord
    Welcome to the interactive sales dashboard for **Superstore**, built to provide powerful insights into sales, profit, customer segmentation, and future forecasting.

    This dashboard includes:
    - 📊 Exploratory Data Analysis (EDA)
    - 📈 Sales & Profit Trends
    - 📦 Product Category & Segment Performance
    - 🌍 Regional Insights
    - 🔮 Forecasting with Prophet
    - 👥 Customer Clustering using K-Means

    **🛠 Built using:** Python, Pandas, Plotly, Prophet, Scikit-learn, and Streamlit.        
    ---
    ### 🎯 Purpose
    - Provide a comprehensive view of sales & profitability
    - Enable strategic planning through forecasting & clustering
    - Present insights in a clean, interactive, and engaging format

    ---
    ### 🛠 Built With
    - Python (Pandas, Plotly, Prophet, Scikit-learn)
    - Streamlit for interactive front-end
    - A deep focus on data storytelling & clean UI

    ---
    ### 👨‍💻 For Clients
    This dashboard is ideal for:
    - Business Owners
    - Marketing & Sales Teams
    - Analysts & Consultants

    Let the data drive your next business decision.
    """)

    # --- Contact Section ---
    st.markdown("""
                
    ### 📬 Connect with Me

    🔗 [Hassan Raza on GitHub](https://github.com/Hassan-Raza-ktk)  
    💼 [Hassan Raza on LinkedIn](https://www.linkedin.com/in/hassan-raza-9651b6279/)  
    📧 Email: [razakhattak123@gmail.com]
    """)