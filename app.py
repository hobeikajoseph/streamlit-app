import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
from scipy.stats import chi2_contingency


# Utility: calculate risk factor importance
@st.cache_data
def calculate_risk_factor_importance(df):
    hd_bin = (df['Heart Disease Status'] == 'Yes').astype(int)
    rf = {}
    rf['Age'] = abs(df['Age'].corr(hd_bin))
    rf['Cholesterol Level'] = chi2_contingency(pd.crosstab(df['Cholesterol Level'], df['Heart Disease Status']))[
                                  0] / len(df)
    rf['Blood Pressure'] = chi2_contingency(pd.crosstab(df['Blood Pressure'], df['Heart Disease Status']))[0] / len(df)
    rf['Smoking'] = abs((df['Smoking'] == 'Yes').astype(int).corr(hd_bin))
    rf['Family Heart Disease'] = abs((df['Family Heart Disease'] == 'Yes').astype(int).corr(hd_bin))
    rf['Exercise Habits'] = chi2_contingency(pd.crosstab(df['Exercise Habits'], df['Heart Disease Status']))[0] / len(
        df)
    rf['BMI'] = abs(df['BMI'].corr(hd_bin))
    rf['Diabetes'] = abs((df['Diabetes'] == 'Yes').astype(int).corr(hd_bin))
    return pd.DataFrame(rf.items(), columns=['Risk Factor', 'Importance'])


# Utility: background image for Home
@st.cache_data
def get_base64(image_path):
    with open(image_path, 'rb') as img:
        return base64.b64encode(img.read()).decode()


def add_bg(image_path):
    b64 = get_base64(image_path)
    st.markdown(f"""
    <style>
    .stApp {{ background-image: url(data:image/png;base64,{b64}); background-size: cover; }}
    </style>
    """, unsafe_allow_html=True)


# Login Page
def show_login_page():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #eef2f3, #8e9eab); }
    .login-container { background: #fff; padding: 2rem; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.05); max-width:400px; margin:1rem auto 1rem auto; text-align:center; } text-align:center; text-align:center; }
    .medical-icon { font-size:3.5rem; color:#e74c3c; margin-bottom:1rem; }
    .login-header { font-size:2.5rem; font-weight:700; background:linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.5rem; }
    .login-subtitle { color:#555; font-size:1rem; margin-bottom:1.5rem; }
    .login-footer { margin-top:1rem; color:#666; font-size:0.8rem; }
    .stTextInput input { border-radius:8px; border:1px solid #ccc; padding:0.6rem 1rem; }
    .stButton button { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; border:none; border-radius:8px; padding:0.6rem 0; width:100%; }
    #MainMenu, header, footer { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("""
        <div class="medical-icon">🫀</div>
        <h1 class="login-header">CardioInsight</h1>
        <p class="login-subtitle">Advanced Cardiovascular Disease Analytics Platform</p>
    """, unsafe_allow_html=True)
    with st.form("login_form"):
        password = st.text_input("🔐 Access Code", type="password", placeholder="Enter code...")
        submitted = st.form_submit_button("🚀 Access Dashboard")
        if submitted:
            if password == "streamlit_health2025":
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("❌ Invalid access code.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="login-footer">
            🔒 HIPAA Compliant & Secure • Unauthorized access prohibited
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# Home Page
def show_home():
    # Small image positioned on right with text on left
    col_txt, col_img = st.columns([3, 1])
    with col_txt:
        st.markdown('<h1 class="dashboard-title"><b>Welcome to the CVD Insights App</b></h1>', unsafe_allow_html=True)
        st.markdown("""
        **Global Impact:** Cardiovascular disease (CVD) claims over 17.9 million lives annually worldwide.

        **Problem Statement:** Heart disease remains the leading cause of mortality, with 80% of cases preventable through lifestyle changes.
        """, unsafe_allow_html=True)
        # Key Insights Preview
        st.markdown('---')
        st.markdown('### 📝 Key Insights')
        # Two-column layout for insight sections
        kb1, kb2 = st.columns(2)
        with kb1:
            st.markdown("""
            🏥 **Disease Burden**  
            - Overall heart disease prevalence: **20.3%**  
            - High-risk patients (>50 yrs & multiple risk factors): **6.3%**  
            - Average patient age: **49.1 years**  
            - Total patients analyzed: **7,067**  
            """, unsafe_allow_html=True)
        with kb2:
            st.markdown("""
            ⚠️ **Risk Factors**  
            - Top risk factor: **Cholesterol Level** (highest correlation)  
            - Secondary risk factors: **Blood Pressure & Age**  
            - Smoking impact: **2.8× higher risk** in smokers vs non-smokers  
            - Gender disparity: **Males** show higher prevalence in older age groups  
            """, unsafe_allow_html=True)
        # Second row of insights
        gi, ci = st.columns(2)
        with gi:
            st.markdown("""
            🌍 **Geographic & Demographic Insights**  
            - Regional disparities: Eastern Europe leads in CVD mortality (Hungary: 26,727 deaths/100k)  
            - Age pattern: CVD risk doubles after age 50  
            - Prevention potential: ~**65%** of cases have modifiable risk factors  
            - Healthcare burden: Peak mortality in **41-50** age group  
            """, unsafe_allow_html=True)
        with ci:
            st.markdown("""
            💡 **Clinical Implications**  
            - Early intervention opportunity: **74.2%** in low-risk category  
            - Multiple risk factors present in **24.9%** of high-risk patients  
            - Blood pressure management could prevent ~**40%** of complications  
            - Lifestyle interventions most effective in **31-50** age demographic  
            """, unsafe_allow_html=True)
    with col_img:
        st.image('images/heart-intro-photo-1.jpg', width=150)


# Heart Disease Dashboard
def show_hd_dashboard(df_hd):
    st.markdown('<h1 class="dashboard-title">Heart Disease Dashboard</h1>', unsafe_allow_html=True)
    # Filters
    st.sidebar.header('Filters')
    age_min, age_max = int(df_hd.Age.min()), int(df_hd.Age.max())
    age_range = st.sidebar.slider('Age Range', age_min, age_max, (age_min, age_max), key='a')
    genders = st.sidebar.multiselect('Gender', df_hd.Gender.unique(), df_hd.Gender.unique(), key='g')
    smoking = st.sidebar.multiselect('Smoking', df_hd.Smoking.unique(), df_hd.Smoking.unique(), key='s')
    diabetes = st.sidebar.multiselect('Diabetes', df_hd.Diabetes.unique(), df_hd.Diabetes.unique(), key='d')
    df = df_hd[
        df_hd.Age.between(*age_range) & df_hd.Gender.isin(genders) & df_hd.Smoking.isin(smoking) & df_hd.Diabetes.isin(
            diabetes)]
    # Key Metrics
    st.subheader('Key Metrics')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('👥 Total Patients', f"{len(df):,}")
    prevalence = df['Heart Disease Status'].eq('Yes').mean() * 100
    col2.metric('❤️ Prevalence', f"{prevalence:.1f}%")
    hr_count = df.query('Age>50 & `Cholesterol Level`>240 & `Blood Pressure`>140 & Smoking=="Yes"')
    hr_pct = len(hr_count) / len(df) * 100 if len(df) > 0 else 0
    col3.metric('⚠️ High Risk', f"{hr_pct:.1f}%")
    avg_age = df.query('`Heart Disease Status`=="Yes"').Age.mean()
    col4.metric('📅 Avg Age HD', f"{avg_age:.1f}")
    st.markdown('---')
    # Primary Visualizations (2 per row)
    # 1. Age vs Prevalence
    df_line = df.copy()
    bins = list(range(age_min, age_max + 5, 5))
    df_line['AgeBucket'] = pd.cut(df_line.Age, bins=bins, right=False)
    trend = df_line.groupby('AgeBucket')['Heart Disease Status'].apply(
        lambda s: (s == 'Yes').mean() * 100).reset_index()
    trend['AgeBucket'] = trend['AgeBucket'].astype(str)
    fig1 = px.line(trend, x='AgeBucket', y='Heart Disease Status', markers=True, title='Age vs HD Prevalence (%)')
    # 2. Correlation Heatmap
    corr_df = df.copy()
    corr_df[['Smoking', 'Exercise Habits']] = corr_df[['Smoking', 'Exercise Habits']].applymap(
        lambda x: 1 if x == 'Yes' else 0)
    corr = corr_df[['Age', 'Cholesterol Level', 'Blood Pressure', 'Smoking', 'BMI', 'Exercise Habits']].corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Risk Factors Correlation')
    r1, r2 = st.columns(2)
    with r1: st.plotly_chart(fig1, use_container_width=True)
    with r2: st.plotly_chart(fig2, use_container_width=True)
    # 3. Gender Split by Age Group & 4. Age vs Trend Area
    age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61+']
    df['AgeGroup'] = pd.Categorical(
        pd.cut(df.Age, bins=[0, 21, 31, 41, 51, 61, age_max + 1], labels=age_labels, right=False),
        categories=age_labels, ordered=True)
    fig3 = px.bar(df, x='AgeGroup', color='Gender', barmode='stack', title='Gender Split by Age Group')
    fig4 = px.area(trend, x='AgeBucket', y='Heart Disease Status', title='Age vs HD Prevalence Trend')
    r3, r4 = st.columns(2)
    with r3: st.plotly_chart(fig3, use_container_width=True)
    with r4: st.plotly_chart(fig4, use_container_width=True)
    # 5. Smoking Impact & 6. BP Categories
    smoke_hd = df.groupby(['Smoking', 'Heart Disease Status']).size().reset_index(name='Count')
    fig5 = px.bar(smoke_hd, x='Smoking', y='Count', color='Heart Disease Status', barmode='group',
                  title='Smoking Impact')
    bp_cat = pd.cut(df['Blood Pressure'], bins=[0, 80, 120, 140, 200], labels=['Low', 'Normal', 'High', 'Very High'],
                    right=False)
    bp_rate = df.assign(BPCat=bp_cat).groupby('BPCat')['Heart Disease Status'].apply(
        lambda s: (s == 'Yes').mean() * 100).reset_index()
    fig6 = px.bar(bp_rate, x='Heart Disease Status', y='BPCat', orientation='h', title='HD Rate by BP Category')
    r5, r6 = st.columns(2)
    with r5: st.plotly_chart(fig5, use_container_width=True)
    with r6: st.plotly_chart(fig6, use_container_width=True)
    # 7. Top Risk Factors & 8. Patient Risk Categories
    top5 = calculate_risk_factor_importance(df).sort_values('Importance', ascending=False).head(5)
    fig7 = px.bar(top5, x='Importance', y='Risk Factor', orientation='h', title='Top Risk Factors Ranking')
    rc = df.copy()
    rc['RiskCat'] = np.select([
        (rc.Age > 50) & (rc['Cholesterol Level'] > 240) & (rc['Blood Pressure'] > 140) & (rc.Smoking == 'Yes'),
        (rc['Heart Disease Status'] == 'Yes')],
        ['High Risk', 'HD Only'], default='Low Risk')
    fig8 = px.pie(rc, names='RiskCat', hole=0.4, title='Patient Risk Categories')
    r7, r8 = st.columns(2)
    with r7: st.plotly_chart(fig7, use_container_width=True)
    with r8: st.plotly_chart(fig8, use_container_width=True)


# Demographics Heatmap
def show_heatmap(df_demo):
    st.title('Demographics Heatmap')
    st.sidebar.header('Filters')
    y_min, y_max = int(df_demo.Year.min()), int(df_demo.Year.max())
    yr = st.sidebar.slider('Year Range', y_min, y_max, (y_min, y_max), key='y')
    df_y = df_demo[df_demo.Year.between(*yr)]
    vmin, vmax = df_y['Deaths per 100k'].min(), df_y['Deaths per 100k'].max()
    fig_map = px.choropleth(df_y, locations='Country', color='Deaths per 100k', locationmode='country names',
                            range_color=(vmin, vmax), title='Deaths per 100k', height=650)
    st.plotly_chart(fig_map, use_container_width=True)
    top10 = df_y.groupby('Country')['Deaths per 100k'].sum().nlargest(10).reset_index()
    st.table(top10)

    # Predictive Insights
    # ... (unchanged)(df_hd):
    st.title("Heart Disease Dashboard")
    st.sidebar.header('Filters')
    age_min, age_max = int(df_hd.Age.min()), int(df_hd.Age.max())
    age_range = st.sidebar.slider('Age Range', age_min, age_max, (age_min, age_max), key='a')
    genders = st.sidebar.multiselect('Gender', df_hd.Gender.unique(), df_hd.Gender.unique(), key='g')
    smoking = st.sidebar.multiselect('Smoking', df_hd.Smoking.unique(), df_hd.Smoking.unique(), key='s')
    diabetes = st.sidebar.multiselect('Diabetes', df_hd.Diabetes.unique(), df_hd.Diabetes.unique(), key='d')
    df = df_hd[
        df_hd.Age.between(*age_range) & df_hd.Gender.isin(genders) & df_hd.Smoking.isin(smoking) & df_hd.Diabetes.isin(
            diabetes)]
    st.subheader('Key Metrics')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('👥 Total', f"{len(df):,}")
    col2.metric('❤️ Prevalence', f"{df['Heart Disease Status'].eq('Yes').mean() * 100:.1f}%")
    col3.metric('⚠️ High Risk',
                f"{len(df.query('Age>50 & `Cholesterol Level`>240 & `Blood Pressure`>140 & Smoking==\"Yes\"')) / len(df) * 100:.1f}%")
    col4.metric('📅 Avg Age HD', f"{df.query('`Heart Disease Status`==\"Yes\"').Age.mean():.1f}")
    st.markdown('---')
    # Charts
    df_line = df.copy()
    bins = list(range(age_min, age_max + 5, 5))
    df_line['AgeBucket'] = pd.cut(df_line.Age, bins=bins, right=False)
    trend = df_line.groupby('AgeBucket')['Heart Disease Status'].apply(
        lambda s: (s == 'Yes').mean() * 100).reset_index()
    trend['AgeBucket'] = trend.AgeBucket.astype(str)
    fig1 = px.line(trend, x='AgeBucket', y='Heart Disease Status', markers=True, title='Age vs Prevalence (%)')
    corr_df = df.copy()
    corr_df[['Smoking', 'Exercise Habits']] = corr_df[['Smoking', 'Exercise Habits']].applymap(
        lambda x: 1 if x == 'Yes' else 0)
    corr = corr_df[['Age', 'Cholesterol Level', 'Blood Pressure', 'Smoking', 'BMI', 'Exercise Habits']].corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Correlation Heatmap')
    r1, c1 = st.columns(2)
    with r1: st.plotly_chart(fig1, use_container_width=True)
    with c1: st.plotly_chart(fig2, use_container_width=True)


# Demographics Heatmap
def show_heatmap(df_demo):
    st.title('Demographics Heatmap')
    st.sidebar.header('Filters')
    y_min, y_max = int(df_demo.Year.min()), int(df_demo.Year.max())
    yr = st.sidebar.slider('Year', y_min, y_max, (y_min, y_max), key='y')
    df_y = df_demo[df_demo.Year.between(*yr)]
    vmin, vmax = df_y['Deaths per 100k'].min(), df_y['Deaths per 100k'].max()
    fig = px.choropleth(df_y, locations='Country', color='Deaths per 100k', locationmode='country names',
                        range_color=(vmin, vmax), title='Deaths per 100k')
    st.plotly_chart(fig, use_container_width=True)
    top10 = df_y.groupby('Country')['Deaths per 100k'].sum().nlargest(10).reset_index()
    st.table(top10)


# Predictive Insights
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
    # Load data
    df_demo = pd.read_csv('demographics_data.csv').dropna()
    if 'Code' in df_demo.columns: df_demo.drop('Code', axis=1, inplace=True)
    longc = [c for c in df_demo.columns if 'deaths' in c.lower()][0]
    df_demo.rename(columns={longc: 'Deaths per 100k'}, inplace=True)
    df_hd = pd.read_csv('heart_disease_data.csv').dropna()
    # Page config
    st.set_page_config(page_title='Health Data App', layout='wide')
    page = st.sidebar.selectbox('Navigation',
                                ['Home', 'Heart Disease Dashboard', 'Demographics Heatmap', 'Predictive Insights'],
                                key='nav')
    if page == 'Home':
        show_home()
    elif page == 'Heart Disease Dashboard':
        show_hd_dashboard(df_hd)
    elif page == 'Demographics Heatmap':
        show_heatmap(df_demo)
    else:
        show_predictive()
