"""
Nassau Candy Distributor – Factory-to-Customer Shipping Route Efficiency Analysis
Streamlit Dashboard  |  Author: Aditya Gautam  |  Unified Mentor Program
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nassau Candy – Shipping Route Analysis",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CSS THEMING
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-header{font-size:2.2rem;font-weight:700;color:#1a1a2e;margin-bottom:0}
    .sub-header{font-size:1rem;color:#666;margin-top:0}
    .kpi-card{background:#f0f4ff;border-radius:10px;padding:18px 22px;text-align:center}
    .kpi-val{font-size:1.9rem;font-weight:700;color:#2563eb}
    .kpi-lbl{font-size:.82rem;color:#555;margin-top:4px}
    .section-header{font-size:1.25rem;font-weight:600;color:#1a1a2e;
                    border-left:4px solid #2563eb;padding-left:10px;margin:18px 0 10px}
    .note-box{background:#fffbeb;border:1px solid #f59e0b;border-radius:8px;
              padding:10px 14px;font-size:.85rem;color:#92400e}
    div[data-testid="stMetricValue"]{font-size:1.7rem}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_data(path="Nassau_Candy_Distributor.csv"):
    df = pd.read_csv(path)

    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  format='%d-%m-%Y')

    # Fix corrupted ship-date years (spreadsheet TODAY()-formula export bug)
    def fix_year(o, s):
        try:   cand = s.replace(year=o.year)
        except ValueError: cand = s.replace(year=o.year, day=28)
        if cand < o:
            try:   cand = cand.replace(year=o.year+1)
            except ValueError: cand = cand.replace(year=o.year+1, day=28)
        return cand

    df['Ship Date Fixed'] = [fix_year(o, s) for o, s in zip(df['Order Date'], df['Ship Date'])]
    df['Lead Time (days)'] = (df['Ship Date Fixed'] - df['Order Date']).dt.days
    df = df[df['Lead Time (days)'] >= 0].copy()

    # Factory mapping
    factory_map = {
        "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
        "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
        "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
        "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
        "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
        "Laffy Taffy":                        "Sugar Shack",
        "SweeTARTS":                          "Sugar Shack",
        "Nerds":                              "Sugar Shack",
        "Fun Dip":                            "Sugar Shack",
        "Fizzy Lifting Drinks":               "Sugar Shack",
        "Everlasting Gobstopper":             "Secret Factory",
        "Hair Toffee":                        "The Other Factory",
        "Lickable Wallpaper":                 "Secret Factory",
        "Wonka Gum":                          "Secret Factory",
        "Kazookles":                          "The Other Factory",
    }
    factory_coords = {
        "Lot's O' Nuts":     (32.881893, -111.768036),
        "Wicked Choccy's":   (32.076176, -81.088371),
        "Sugar Shack":       (48.11914,  -96.18115),
        "Secret Factory":    (41.446333, -90.565487),
        "The Other Factory": (35.1175,   -89.971107),
    }
    df['Factory'] = df['Product Name'].map(factory_map).fillna("Unknown")
    df['Factory Lat'] = df['Factory'].map(lambda f: factory_coords.get(f,(None,None))[0])
    df['Factory Lon'] = df['Factory'].map(lambda f: factory_coords.get(f,(None,None))[1])
    df['Route'] = df['Factory'] + " → " + df['State/Province']
    df['Route Region'] = df['Factory'] + " → " + df['Region']

    p75 = df['Lead Time (days)'].quantile(0.75)
    df['Is Delayed'] = df['Lead Time (days)'] > p75
    df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
    df['Order Year']  = df['Order Date'].dt.year
    return df, p75

@st.cache_data
def make_route_df(df):
    rdf = df.groupby(['Route','Factory','State/Province','Region']).agg(
        Total_Shipments=('Lead Time (days)', 'count'),
        Avg_Lead_Time  =('Lead Time (days)', 'mean'),
        Lead_Time_Std  =('Lead Time (days)', 'std'),
        Delay_Count    =('Is Delayed', 'sum'),
        Total_Sales    =('Sales', 'sum'),
        Total_Profit   =('Gross Profit', 'sum'),
    ).reset_index()
    rdf['Delay_Frequency'] = (rdf['Delay_Count']/rdf['Total_Shipments']*100).round(2)
    rdf['Lead_Time_Std']   = rdf['Lead_Time_Std'].fillna(0).round(3)
    scaler = MinMaxScaler()
    rdf['LT_norm']    = scaler.fit_transform(rdf[['Avg_Lead_Time']])
    rdf['Delay_norm'] = scaler.fit_transform(rdf[['Delay_Frequency']])
    rdf['Route_Efficiency_Score'] = (
        1 - (0.6*rdf['LT_norm'] + 0.4*rdf['Delay_norm'])
    ) * 100
    rdf['Route_Efficiency_Score'] = rdf['Route_Efficiency_Score'].round(2)
    rdf = rdf.sort_values('Route_Efficiency_Score', ascending=False).reset_index(drop=True)
    rdf['Rank'] = range(1, len(rdf)+1)
    return rdf

df_raw, P75 = load_data()
route_df = make_route_df(df_raw)

# ══════════════════════════════════════════════════════════
# SIDEBAR – FILTERS
# ══════════════════════════════════════════════════════════
st.sidebar.image("https://via.placeholder.com/200x60/1a1a2e/ffffff?text=Nassau+Candy", use_column_width=True)
st.sidebar.markdown("## 🔍 Filters")

regions   = ["All"] + sorted(df_raw['Region'].unique().tolist())
factories = ["All"] + sorted(df_raw['Factory'].unique().tolist())
modes     = ["All"] + sorted(df_raw['Ship Mode'].unique().tolist())

sel_region  = st.sidebar.selectbox("Region",   regions)
sel_factory = st.sidebar.selectbox("Factory",  factories)
sel_mode    = st.sidebar.selectbox("Ship Mode", modes)

min_lt, max_lt = int(df_raw['Lead Time (days)'].min()), int(df_raw['Lead Time (days)'].max())
lt_range = st.sidebar.slider("Lead Time Range (days)", min_lt, max_lt, (min_lt, max_lt))

min_date = df_raw['Order Date'].min().date()
max_date = df_raw['Order Date'].max().date()
date_range = st.sidebar.date_input("Order Date Range", (min_date, max_date),
                                    min_value=min_date, max_value=max_date)

# Apply filters
df = df_raw.copy()
if sel_region  != "All": df = df[df['Region']    == sel_region]
if sel_factory != "All": df = df[df['Factory']   == sel_factory]
if sel_mode    != "All": df = df[df['Ship Mode']  == sel_mode]
df = df[df['Lead Time (days)'].between(lt_range[0], lt_range[1])]
if len(date_range) == 2:
    df = df[(df['Order Date'].dt.date >= date_range[0]) &
            (df['Order Date'].dt.date <= date_range[1])]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records shown:** {len(df):,} / {len(df_raw):,}")
st.sidebar.markdown("""
<small>📌 **Note on Lead Times:** Ship Date column had a spreadsheet 
formula export bug (TODAY()-based dates). Years were corrected; 
relative route comparisons are fully valid.</small>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown('<p class="main-header">🍬 Nassau Candy – Shipping Route Efficiency Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Factory-to-Customer logistics intelligence across US regions · Unified Mentor Program</p>', unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Overview", "🗺️ Geographic Map", "🚚 Ship Mode", "🏭 Route Drill-Down", "📋 Data Table"])

# ────────────────────────────────────────────────
# TAB 1 – OVERVIEW
# ────────────────────────────────────────────────
with tabs[0]:
    # KPIs
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("📦 Total Orders",     f"{len(df):,}")
    col2.metric("⏱️ Avg Lead Time",    f"{df['Lead Time (days)'].mean():.1f} days")
    col3.metric("⚠️ Delay Rate",       f"{df['Is Delayed'].mean()*100:.1f}%")
    col4.metric("💰 Total Sales",      f"${df['Sales'].sum():,.0f}")
    col5.metric("🏭 Active Routes",    f"{df['Route'].nunique()}")

    st.markdown('<p class="section-header">Route Efficiency Leaderboard – Top 10 vs Bottom 10</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        top10 = route_df.head(10)[['Rank','Route','Avg_Lead_Time','Delay_Frequency','Route_Efficiency_Score']]
        top10.columns = ['Rank','Route','Avg LT','Delay%','Eff. Score']
        st.dataframe(top10.style.background_gradient(subset=['Eff. Score'], cmap='Greens'), use_container_width=True, hide_index=True)
    with c2:
        bot10 = route_df.tail(10).sort_values('Rank',ascending=False)[['Rank','Route','Avg_Lead_Time','Delay_Frequency','Route_Efficiency_Score']]
        bot10.columns = ['Rank','Route','Avg LT','Delay%','Eff. Score']
        st.dataframe(bot10.style.background_gradient(subset=['Eff. Score'], cmap='Reds_r'), use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Lead Time Distribution</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='Lead Time (days)', color='Ship Mode', nbins=30,
                           title="Lead Time Distribution by Ship Mode",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig.add_vline(x=df['Lead Time (days)'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['Lead Time (days)'].mean():.1f}d")
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        region_avg = df.groupby('Region')['Lead Time (days)'].mean().reset_index()
        fig = px.bar(region_avg.sort_values('Lead Time (days)'), x='Region', y='Lead Time (days)',
                     color='Lead Time (days)', color_continuous_scale='RdYlGn_r',
                     title="Average Lead Time by Region")
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Monthly Shipment Trend</p>', unsafe_allow_html=True)
    monthly = df.groupby('Order Month').agg(
        Orders=('Lead Time (days)','count'),
        Avg_LT=('Lead Time (days)','mean')
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly['Order Month'], y=monthly['Orders'],
                         name='Shipments', marker_color='#93c5fd', yaxis='y'))
    fig.add_trace(go.Scatter(x=monthly['Order Month'], y=monthly['Avg_LT'],
                             name='Avg Lead Time', line=dict(color='#dc2626', width=2),
                             yaxis='y2', mode='lines+markers'))
    fig.update_layout(
        title="Monthly Shipment Volume & Avg Lead Time",
        yaxis=dict(title='Shipments'),
        yaxis2=dict(title='Avg Lead Time (days)', overlaying='y', side='right'),
        height=380, legend=dict(x=0, y=1.1, orientation='h')
    )
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# TAB 2 – GEOGRAPHIC MAP
# ────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-header">US Shipping Heatmap – Delay Rate by State</p>', unsafe_allow_html=True)

    state_stats = df.groupby('State/Province').agg(
        Orders    =('Lead Time (days)','count'),
        Avg_LT    =('Lead Time (days)','mean'),
        Delay_Pct =('Is Delayed','mean'),
        Total_Sales=('Sales','sum'),
    ).reset_index()
    state_stats['Delay_Pct'] = (state_stats['Delay_Pct']*100).round(2)

    # US state abbreviation map (for choropleth)
    us_abbr = {
        'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
        'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
        'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
        'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
        'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS',
        'Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
        'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC',
        'North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
        'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN',
        'Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
        'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
        'District of Columbia':'DC'
    }
    state_stats['State Code'] = state_stats['State/Province'].map(us_abbr)
    state_us = state_stats[state_stats['State Code'].notna()].copy()

    fig = px.choropleth(state_us, locations='State Code', locationmode='USA-states',
                        color='Delay_Pct', scope='usa',
                        color_continuous_scale='YlOrRd',
                        hover_data={'Avg_LT':':.1f','Orders':True},
                        labels={'Delay_Pct':'Delay Rate (%)','Avg_LT':'Avg Lead Time'},
                        title="State-wise Delay Rate (%)")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Factory Locations</p>', unsafe_allow_html=True)
    factory_locs = pd.DataFrame({
        'Factory': ["Lot's O' Nuts","Wicked Choccy's","Sugar Shack","Secret Factory","The Other Factory"],
        'lat':     [32.881893, 32.076176, 48.11914, 41.446333, 35.1175],
        'lon':     [-111.768036,-81.088371,-96.18115,-90.565487,-89.971107],
    })
    fig2 = px.scatter_geo(factory_locs, lat='lat', lon='lon', text='Factory',
                          scope='usa', title="Factory Locations across USA",
                          size_max=20)
    fig2.update_traces(marker=dict(size=14, color='#2563eb', symbol='star'),
                       textposition='top center')
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-header">Geographic Bottleneck Analysis – Top 15 States by Delay Rate</p>', unsafe_allow_html=True)
    top_delay = state_stats.sort_values('Delay_Pct', ascending=False).head(15)
    fig3 = px.bar(top_delay, x='State/Province', y='Delay_Pct',
                  color='Delay_Pct', color_continuous_scale='Reds',
                  text='Delay_Pct', labels={'Delay_Pct':'Delay Rate (%)'},
                  title="Top 15 Congestion-Prone States")
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# ────────────────────────────────────────────────
# TAB 3 – SHIP MODE
# ────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-header">Ship Mode Performance Comparison</p>', unsafe_allow_html=True)

    sm = df.groupby('Ship Mode').agg(
        Orders   =('Lead Time (days)','count'),
        Avg_LT   =('Lead Time (days)','mean'),
        Std_LT   =('Lead Time (days)','std'),
        Delay_Pct=('Is Delayed','mean'),
        Avg_Cost =('Cost','mean'),
        Total_Sales=('Sales','sum'),
    ).reset_index()
    sm['Delay_Pct']=(sm['Delay_Pct']*100).round(2)
    sm['Avg_LT']   = sm['Avg_LT'].round(2)
    sm['Avg_Cost'] = sm['Avg_Cost'].round(2)

    c1,c2,c3 = st.columns(3)
    with c1:
        fig = px.bar(sm.sort_values('Avg_LT'), x='Ship Mode', y='Avg_LT',
                     color='Ship Mode', text='Avg_LT',
                     title="Avg Lead Time by Ship Mode",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(texttemplate='%{text:.1f}d', textposition='outside')
        fig.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(sm, x='Ship Mode', y='Delay_Pct',
                     color='Ship Mode', text='Delay_Pct',
                     title="Delay Rate (%) by Ship Mode",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.scatter(sm, x='Avg_LT', y='Avg_Cost', size='Orders',
                         color='Ship Mode', text='Ship Mode',
                         title="Cost vs Lead Time Trade-off",
                         color_discrete_sequence=px.colors.qualitative.Dark2)
        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Ship Mode Summary Table</p>', unsafe_allow_html=True)
    sm_disp = sm[['Ship Mode','Orders','Avg_LT','Delay_Pct','Avg_Cost','Total_Sales']].copy()
    sm_disp.columns = ['Ship Mode','Orders','Avg Lead Time (d)','Delay Rate (%)','Avg Cost ($)','Total Sales ($)']
    sm_disp['Total Sales ($)'] = sm_disp['Total Sales ($)'].map('${:,.2f}'.format)
    st.dataframe(sm_disp.style.background_gradient(subset=['Avg Lead Time (d)','Delay Rate (%)'], cmap='YlOrRd'),
                 use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Lead Time Box Plot by Ship Mode</p>', unsafe_allow_html=True)
    fig = px.box(df, x='Ship Mode', y='Lead Time (days)', color='Ship Mode',
                 title="Lead Time Spread by Ship Mode",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# TAB 4 – ROUTE DRILL-DOWN
# ────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-header">Route-Level Performance Explorer</p>', unsafe_allow_html=True)

    factories_list = sorted(df['Factory'].unique().tolist())
    sel_fac2 = st.selectbox("Select Factory", factories_list, key='drill_fac')
    fac_routes = df[df['Factory']==sel_fac2]

    state_perf = fac_routes.groupby('State/Province').agg(
        Orders   =('Lead Time (days)','count'),
        Avg_LT   =('Lead Time (days)','mean'),
        Delay_Pct=('Is Delayed','mean'),
        Sales    =('Sales','sum'),
    ).reset_index()
    state_perf['Delay_Pct']=(state_perf['Delay_Pct']*100).round(2)
    state_perf['Avg_LT']   = state_perf['Avg_LT'].round(2)
    state_perf = state_perf.sort_values('Avg_LT')

    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(state_perf.head(15), x='State/Province', y='Avg_LT',
                     color='Avg_LT', color_continuous_scale='RdYlGn_r',
                     title=f"{sel_fac2} – Avg Lead Time per State (Top 15 Fastest)")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(state_perf, x='Avg_LT', y='Delay_Pct',
                         size='Orders', color='Delay_Pct',
                         hover_name='State/Province',
                         color_continuous_scale='RdYlGn_r',
                         title=f"{sel_fac2} – Lead Time vs Delay Rate per State")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Order-Level Shipment Timeline</p>', unsafe_allow_html=True)
    sample = fac_routes.sample(min(300, len(fac_routes)), random_state=42).sort_values('Order Date')
    fig = px.scatter(sample, x='Order Date', y='Lead Time (days)',
                     color='Ship Mode', hover_data=['State/Province','Product Name'],
                     title=f"Individual Order Lead Times – {sel_fac2}",
                     color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">State Performance Table</p>', unsafe_allow_html=True)
    st.dataframe(state_perf.style.background_gradient(subset=['Avg_LT','Delay_Pct'], cmap='YlOrRd'),
                 use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────
# TAB 5 – DATA TABLE
# ────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-header">Filtered Dataset</p>', unsafe_allow_html=True)
    cols_show = ['Order ID','Order Date','Factory','Ship Mode','State/Province','Region',
                 'Product Name','Lead Time (days)','Sales','Gross Profit','Is Delayed']
    disp = df[cols_show].copy()
    disp['Order Date'] = disp['Order Date'].dt.strftime('%Y-%m-%d')
    disp['Is Delayed'] = disp['Is Delayed'].map({True:'⚠️ Yes', False:'✅ No'})
    st.dataframe(disp, use_container_width=True, height=500)

    csv_out = df[cols_show].to_csv(index=False).encode()
    st.download_button("⬇️ Download Filtered Data", csv_out, "nassau_filtered.csv", "text/csv")
