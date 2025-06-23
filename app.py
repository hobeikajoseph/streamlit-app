import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import streamlit.components.v1 as components
from scipy.stats import chi2_contingency

# Calculate risk factor importance
@st.cache_data
def calculate_risk_factor_importance(df):
    hd_bin = (df['Heart Disease Status'] == 'Yes').astype(int)
    rf = {}
    rf['Age'] = abs(df['Age'].corr(hd_bin))
    rf['Cholesterol Level'] = chi2_contingency(pd.crosstab(df['Cholesterol Level'], df['Heart Disease Status']))[0] / len(df)
    rf['Blood Pressure'] = chi2_contingency(pd.crosstab(df['Blood Pressure'], df['Heart Disease Status']))[0] / len(df)
    rf['Smoking'] = abs((df['Smoking'] == 'Yes').astype(int).corr(hd_bin))
    rf['Family Heart Disease'] = abs((df['Family Heart Disease'] == 'Yes').astype(int).corr(hd_bin))
    rf['Exercise Habits'] = chi2_contingency(pd.crosstab(df['Exercise Habits'], df['Heart Disease Status']))[0] / len(df)
    rf['BMI'] = abs(df['BMI'].corr(hd_bin))
    rf['Diabetes'] = abs((df['Diabetes'] == 'Yes').astype(int).corr(hd_bin))
    return pd.DataFrame(rf.items(), columns=['Risk Factor','Importance'])

# Login Page
def show_login_page():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg,#eef2f3,#8e9eab); }
    .login-container { background:#fff; padding:1.5rem; border-radius:15px; max-width:360px; margin:1rem auto; text-align:center; box-shadow:0 15px 30px rgba(0,0,0,0.05); }
    .medical-icon { font-size:3rem; color:#e74c3c; margin-bottom:0.75rem; }
    .login-header { font-size:2rem; font-weight:700; background:linear-gradient(135deg,#667eea,#764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.25rem; }
    .login-subtitle { color:#555; font-size:0.9rem; margin-bottom:1rem; }
    .stTextInput input { border-radius:6px; border:1px solid #ccc; padding:0.5rem 0.75rem; }
    .stButton button { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; border:none; border-radius:6px; padding:0.5rem; width:100%; }
    #MainMenu, header, footer { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='medical-icon'>🫀</div>
    <h1 class='login-header'>CardioInsight</h1>
    <p class='login-subtitle'>Advanced CVD Analytics Platform</p>
    """, unsafe_allow_html=True)
    with st.form("login_form"):
        pw = st.text_input('🔐 Access Code', type='password', placeholder='Enter code', key='login_pw')
        submitted = st.form_submit_button('🚀 Access Dashboard')
        if submitted:
            if pw == 'streamlit_health2025':
                st.session_state.logged_in = True
            else:
                st.error('❌ Invalid code')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='login-footer'>
        🔒 HIPAA Compliant & Secure • Unauthorized access prohibited
    </div>
    """, unsafe_allow_html=True)
    st.stop()
def show_home():
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<h1 class="dashboard-title"><b>Welcome to the CVD Insights App</b></h1>', unsafe_allow_html=True)
        st.markdown("""
        **Global Impact:** Cardiovascular disease (CVD) claims over 17.9 million lives annually worldwide.

        **Problem Statement:** Heart disease remains the leading cause of mortality, with 80% of cases preventable through lifestyle changes.
        """, unsafe_allow_html=True)
    with col2:
        st.image('heart-intro-photo-1.jpg', width=150)

# Heart Disease Dashboard
def show_hd_dashboard(df_hd):
    st.markdown('<h1 class="dashboard-title">Heart Disease Dashboard</h1>', unsafe_allow_html=True)
    # Filters
    st.sidebar.header('Filters')
    age_min, age_max = int(df_hd['Age'].min()), int(df_hd['Age'].max())
    age_range = st.sidebar.slider('Age Range', age_min, age_max, (age_min, age_max), key='a')
    genders = st.sidebar.multiselect('Gender', df_hd['Gender'].unique(), df_hd['Gender'].unique(), key='g')
    smoking = st.sidebar.multiselect('Smoking', df_hd['Smoking'].unique(), df_hd['Smoking'].unique(), key='s')
    diabetes = st.sidebar.multiselect('Diabetes', df_hd['Diabetes'].unique(), df_hd['Diabetes'].unique(), key='d')
    # Filter data
    df = df_hd[
        df_hd['Age'].between(*age_range) & df_hd['Gender'].isin(genders) & df_hd['Smoking'].isin(smoking) & df_hd['Diabetes'].isin(diabetes)
    ].copy()
    # Key Metrics
    st.subheader('Key Metrics')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('👥 Total Patients', f"{len(df):,}")
    prev = df['Heart Disease Status'].eq('Yes').mean()*100
    c2.metric('❤️ Prevalence', f"{prev:.1f}%")
    hr = df.query("Age>50 & `Cholesterol Level`>240 & `Blood Pressure`>140 & Smoking=='Yes'")
    hr_pct = len(hr)/len(df)*100 if len(df) else 0
    c3.metric('⚠️ High-Risk %', f"{hr_pct:.1f}%")
    avg_age = df.loc[df['Heart Disease Status']=='Yes','Age'].mean()
    c4.metric('📅 Avg Age HD', f"{avg_age:.1f} yrs")
    st.markdown('---')
    # Risk Factors Correlation Heatmap
    corr_df = df.copy()
    corr_df[['Smoking','Exercise Habits']] = corr_df[['Smoking','Exercise Habits']].applymap(lambda x:1 if x=='Yes' else 0)
    corr = corr_df[['Age','Cholesterol Level','Blood Pressure','Smoking','BMI','Exercise Habits']].corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Risk Factors Correlation')
    # Cholesterol Distribution Pie
    hd_only = df[df['Heart Disease Status']=='Yes'].copy()
    hd_only['CholCat'] = hd_only['Cholesterol Level'].apply(lambda x: 'Low' if x<200 else ('Medium' if x<240 else 'High'))
    fig3 = px.pie(hd_only, names='CholCat', title='Cholesterol Distribution (HD Patients)', hole=0.3)
    r1, r2 = st.columns(2)
    with r1: st.plotly_chart(fig2, use_container_width=True)
    with r2: st.plotly_chart(fig3, use_container_width=True)(fig2, use_container_width=True)
    with r2: st.plotly_chart(fig3, use_container_width=True)
    # Secondary Insights: Smoking & BP
    smoke_df = df.groupby(['Smoking','Heart Disease Status']).size().reset_index(name='Count')
    fig5 = px.bar(smoke_df, x='Smoking', y='Count', color='Heart Disease Status', barmode='group', title='Smoking Impact')
    df['BPCat'] = pd.cut(df['Blood Pressure'], bins=[0,80,120,140,200], labels=['Low','Normal','High','Very High'], right=False)
    bp_rate = df.groupby('BPCat')['Heart Disease Status'].apply(lambda s:(s=='Yes').mean()*100).reset_index()
    fig6 = px.bar(bp_rate, x='Heart Disease Status', y='BPCat', orientation='h', title='HD Rate by BP Category')
    r3, r4 = st.columns(2)
    with r3: st.plotly_chart(fig5, use_container_width=True)
    with r4: st.plotly_chart(fig6, use_container_width=True)
    # Top Risk Factors & Patient Risk Categories
    top5 = calculate_risk_factor_importance(df).sort_values('Importance', ascending=False).head(5)
    fig7 = px.bar(top5, x='Importance', y='Risk Factor', orientation='h', title='Top Risk Factors Ranking')
    rc = df.copy()
    rc['RiskCat'] = np.select([
        (rc['Age']>50)&(rc['Cholesterol Level']>240)&(rc['Blood Pressure']>140)&(rc['Smoking']=='Yes'),
        (rc['Heart Disease Status']=='Yes')], ['High Risk','HD Only'], default='Low Risk')
    fig8 = px.pie(rc, names='RiskCat', hole=0.4, title='Patient Risk Categories')
    r5, r6 = st.columns(2)
    with r5: st.plotly_chart(fig7, use_container_width=True)
    with r6: st.plotly_chart(fig8, use_container_width=True)

def show_heatmap(df_demo):
    st.title('Demographics Heatmap')
    st.sidebar.header('Filters')
    y_min, y_max = int(df_demo['Year'].min()), int(df_demo['Year'].max())
    yr = st.sidebar.slider('Year Range', y_min, y_max, (y_min, y_max), key='y')
    df_y = df_demo[df_demo['Year'].between(*yr)]
    vmin, vmax = df_y['Deaths per 100k'].min(), df_y['Deaths per 100k'].max()
    fig_map = px.choropleth(df_y, locations='Country', color='Deaths per 100k', locationmode='country names', range_color=(vmin, vmax), title='Deaths per 100k', height=650)
    st.plotly_chart(fig_map, use_container_width=True)
    top10 = df_y.groupby('Country')['Deaths per 100k'].sum().nlargest(10).reset_index()
    # Number ranks starting at 1
    top10.index = range(1, len(top10) + 1)
    st.table(top10)

def show_predictive():
    st.title('Predictive Insights')
    st.subheader('CVD Risk Calculator')
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider('Age', 20, 80, 45)
        bp = st.selectbox('Blood Pressure', ['Normal', 'High', 'Very High'])
    with c2:
        smoke = st.radio('Smoking', ['No', 'Yes'])
        diab = st.radio('Diabetes', ['No', 'Yes'])
    score = (age - 20) * 0.5 + {'Normal': 0, 'High': 10, 'Very High': 20}[bp]
    score += 10 if smoke == 'Yes' else 0
    score += 10 if diab == 'Yes' else 0
    cat = 'Low' if score < 30 else 'Moderate' if score < 60 else 'High'
    st.markdown(f'**Risk Score:** {int(score)} ({cat})')

# Entry point
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    show_login_page()
else:
    from pathlib import Path
    base_path = Path(__file__).parent
    df_demo = pd.read_csv(base_path / 'demographics_data.csv').dropna()
    long_col = [c for c in df_demo.columns if 'deaths' in c.lower()][0]
    df_demo.rename(columns={long_col: 'Deaths per 100k'}, inplace=True)
    df_hd = pd.read_csv(base_path / 'heart_disease_data.csv').dropna()
    st.set_page_config(page_title='Health Data App', layout='wide')
    page = st.sidebar.selectbox('Navigation', ['Home', 'Heart Disease Dashboard', 'Demographics Heatmap', 'Predictive Insights'], key='nav')
    if page == 'Home':
        show_home()
    elif page == 'Heart Disease Dashboard':
        show_hd_dashboard(df_hd)
    elif page == 'Demographics Heatmap':
        show_heatmap(df_demo)
    else:
        show_predictive()
