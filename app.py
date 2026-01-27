import streamlit as st
import pandas as pd
import yaml
import torch
import time
from models.t5_generator import T5JDGenerator
from models.bert_ranker import BertRanker
from utils.metrics import calculate_cvm
from models.attention_utils import compare_candidates
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Import Novel Features
try:
    from utils.hallucination_detector import HallucinationDetector
    from models.drift_detector import RoleDriftDetector
    from utils.sensitivity_analysis import SensitivityAnalyzer
    from utils.skill_rarity import SkillRarityCalculator
    NOVEL_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some novel features unavailable: {e}")
    NOVEL_FEATURES_AVAILABLE = False

# --- Configuration & Setup ---
st.set_page_config(
    page_title="TalentAI | Enterprise Edition",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Fail silently if CSS is missing

local_css("assets/style.css")

# Load Config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Config file not found. Please ensure 'config.yaml' exists.")
    st.stop()

# Extract feature flags
features = config.get("features", {})
HRI_ENABLED = features.get("hri_enabled", False)
DRIFT_ENABLED = features.get("drift_enabled", False)
SENSITIVITY_ENABLED = features.get("sensitivity_enabled", False)
SRW_ENABLED = features.get("srw_enabled", False)
HRI_THRESHOLD = features.get("hri_threshold", 0.3)
DRIFT_THRESHOLD = features.get("drift_threshold", 0.4)
SENSITIVITY_SIMS = features.get("sensitivity_simulations", 50)

# --- Model Loading ---
@st.cache_resource
def load_models():
    t5 = T5JDGenerator(model_name=config["models"]["t5_name"])
    ranker = BertRanker(model_name=config["models"]["bert_name"])
    return t5, ranker

@st.cache_resource
def load_novel_features():
    """Load novel feature modules."""
    modules = {}
    if NOVEL_FEATURES_AVAILABLE:
        skill_corpus = config.get("data", {}).get("skill_corpus", "data/job_skills.csv")
        if HRI_ENABLED:
            try:
                modules["hri"] = HallucinationDetector(skill_corpus)
            except Exception as e:
                print(f"HRI init failed: {e}")
        if DRIFT_ENABLED:
            try:
                modules["drift"] = RoleDriftDetector()
            except Exception as e:
                print(f"Drift detector init failed: {e}")
        if SENSITIVITY_ENABLED:
            modules["sensitivity"] = SensitivityAnalyzer()
        if SRW_ENABLED:
            try:
                modules["srw"] = SkillRarityCalculator(skill_corpus)
            except Exception as e:
                print(f"SRW init failed: {e}")
    return modules

with st.spinner("Initializing Research Engine..."):
    t5_model, bert_model = load_models()
    novel_modules = load_novel_features()

# --- Sidebar: Controls & Physics ---
# --- Sidebar: Controls & Physics ---
with st.sidebar:
    st.markdown("### Critical Competency Assessment")
    st.caption("Candidate scoring parameters")

    # Navigation
    st.markdown("### Ranking Physics")
    st.caption("Candidate scoring parameters")
    
    with st.expander("Parameter Details", expanded=False):
        st.markdown("""
        **Alpha (Semantic Match)**:  
        Relevance of resume content to JD.
        
        **Beta (Skill Penalty)**:  
        Strictness on mandatory skills.
        
        **Gamma (Experience Gap)**:  
        Penalty for experience shortfall.
        """)

    # Sliders
    alpha = st.slider("Alpha (Semantic Match)", 0.0, 1.0, config["weights"]["alpha"])
    beta = st.slider("Beta (Skill Penalty)", 0.0, 1.0, config["weights"]["beta"])
    gamma = st.slider("Gamma (Exp. Gap)", 0.0, 1.0, config["weights"]["gamma"])

    # Total Weight Indicator
    total = alpha + beta + gamma
    st.metric("Total Weight", f"{total:.1f}", delta="Normalized" if abs(total-1.0)<0.1 else "Unbalanced", delta_color="normal")
    
    st.markdown("---")
    
    # Radar Chart
    categories = ['Python', 'SQL', 'AWS', 'System Design', 'Commun.']
    # Dynamic logic for demo purposes based on sliders
    values = [70 + (alpha*20), 60 + (beta*30), 50 + (gamma*40), 80, 90] 
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Ideal Profile',
        line_color='#334155',
        fillcolor='rgba(51, 65, 85, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, tickfont=dict(size=8)),
            bgcolor='white'
        ),
        showlegend=False,
        height=250,
        margin=dict(l=30, r=30, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=10, color="#64748B")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.caption("Required skill levels for current open positions")
    
    st.markdown("---")
    
    # Novel Feature Toggles
    st.markdown("### 🔬 Research Features")
    show_hri = st.checkbox("Show HRI Score", value=HRI_ENABLED and "hri" in novel_modules)
    show_drift = st.checkbox("Show Drift Detection", value=DRIFT_ENABLED and "drift" in novel_modules)
    show_stability = st.checkbox("Show Rank Stability", value=SENSITIVITY_ENABLED and "sensitivity" in novel_modules)
    use_srw = st.checkbox("Use Skill Rarity Weighting", value=SRW_ENABLED and "srw" in novel_modules)


# --- Main Interface ---
# State Management
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "generated_role" not in st.session_state:
    st.session_state["generated_role"] = ""

# Layout: Breadcrumbs
# Custom Header
# Enterprise Header
st.markdown("""
<div class="enterprise-header">
    <div class="header-title-group">
        <h1>TalentAI</h1>
        <div class="header-subtitle">Professional Talent Analytics Platform</div>
    </div>
    <div>
        <button class="enterprise-badge">Enterprise Edition</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Position Builder", "Candidate Evaluation", "Analytics"])

# --- TAB 1: GENERATOR (Split 50/50) ---
with tab1:
    col_input, col_preview = st.columns([1, 1], gap="medium")
    
    with col_input:
        st.subheader("Job Description Generator")
        with st.container(border=True):
            role = st.text_input("Target Role", "Senior Backend Engineer")
            exp = st.slider("Experience Required", 0, 15, 8, format="%d years")
            
            st.write("")
            mandatory = st.text_area("Mandatory Skills", "Python, Django, AWS", height=100)
            optional = st.text_input("Optional Skills", "Kubernetes, PostgreSQL")
            
            st.write("")
            if st.button("Generate Job Description", type="primary", use_container_width=True):
                with st.spinner("Synthesizing..."):
                    time.sleep(0.5)
                    jd_text = t5_model.generate(role, mandatory, optional, exp)
                    st.session_state["jd_text"] = jd_text
                    st.session_state["generated_role"] = role
                    st.session_state["mandatory"] = mandatory
                    st.session_state["exp"] = exp
                    
                    # Run novel feature analysis
                    if show_hri and "hri" in novel_modules:
                        hri_score, hri_details = novel_modules["hri"].calculate_hri(jd_text, mandatory.split(","))
                        st.session_state["hri_score"] = hri_score
                        st.session_state["hri_details"] = hri_details
                    
                    if show_drift and "drift" in novel_modules:
                        drift_score, drift_details = novel_modules["drift"].detect_drift(role, jd_text, mandatory)
                        st.session_state["drift_score"] = drift_score
                        st.session_state["drift_details"] = drift_details

    with col_preview:
        st.subheader("Generated Preview")
        with st.container(border=True):
            if st.session_state["jd_text"]:
                st.markdown(st.session_state["jd_text"])
                st.divider()
                
                # Footer Metrics including novel features
                cvm = calculate_cvm(st.session_state["jd_text"], st.session_state.get("mandatory", ""))
                
                # Metrics row 1: Original metrics
                c1, c2 = st.columns(2)
                c1.metric("Constraint Adherence", f"{(1-cvm):.0%}")
                c2.metric("Missing Skills", f"{cvm:.0%}", delta_color="inverse")
                
                # Metrics row 2: Novel features (HRI + Drift)
                if st.session_state.get("hri_score") is not None or st.session_state.get("drift_score") is not None:
                    c3, c4 = st.columns(2)
                    
                    if st.session_state.get("hri_score") is not None:
                        hri = st.session_state["hri_score"]
                        hri_level = st.session_state.get("hri_details", {}).get("risk_level", "")
                        c3.metric(
                            "Hallucination Risk", 
                            f"{hri:.0%}",
                            delta=hri_level.split()[-1] if hri_level else None,
                            delta_color="inverse" if hri > HRI_THRESHOLD else "normal"
                        )
                    
                    if st.session_state.get("drift_score") is not None:
                        drift = st.session_state["drift_score"]
                        drift_level = st.session_state.get("drift_details", {}).get("drift_level", "")
                        c4.metric(
                            "Role Alignment", 
                            f"{(1-drift):.0%}",
                            delta=drift_level.split()[-1] if drift_level else None,
                            delta_color="normal" if drift < DRIFT_THRESHOLD else "inverse"
                        )
                        
                        # Show drift warning if significant
                        if drift > DRIFT_THRESHOLD:
                            contaminated = st.session_state.get("drift_details", {}).get("contaminated_role")
                            if contaminated:
                                st.warning(f"⚠️ JD may drift towards: {contaminated.title()}")
                
            else:
                st.info("Fill in the form and click Generate to see the job description preview")
                st.caption("Waiting for input...")
                for _ in range(8): st.write("") # Spacer to match height

# --- TAB 2: RANKER (Table View with Stability) ---
# --- TAB 2: CANDIDATE EVALUATION MATRIX ---
with tab2:
    st.markdown("### Candidate Evaluation Matrix")
    st.caption("Rank candidates based on position requirements using advanced skill matching")
    
    # State check
    if not st.session_state["jd_text"]:
        st.info("ℹ️ To begin candidate evaluation, generate a position description first using the Position Builder tab.")
    
    if st.button("Generate Initial Rankings", type="primary", use_container_width=True):
        if not st.session_state["jd_text"]:
            st.error("Please generate a JD first.")
        else:
            with st.spinner("Analyzing candidates via BERT..."):
                try:
                    resumes_df = pd.read_csv(config.get("data", {}).get("resumes", "data/resumes.csv"))
                    # Mocking file processing for demo speed, using pre-loaded CSV as source
                    # In real app, we parse `uploaded_files`
                    
                    results = []
                    # Reuse existing logic
                    skill_weights = None
                    if use_srw and "srw" in novel_modules:
                        mandatory_skills = [s.strip().lower() for s in st.session_state.get("mandatory", "").split(",")]
                        skill_weights = {}
                        for skill in mandatory_skills:
                            weight, _ = novel_modules["srw"].get_rarity_weight(skill)
                            skill_weights[skill] = weight

                    for idx, row in resumes_df.iterrows():
                        resume_text = f"Skills: {row['Skills']}. Experience: {row['Experience (Years)']} years."
                        gap = max(0, st.session_state.get("exp", 5) - row['Experience (Years)'])
                        
                        score, info = bert_model.get_research_score(
                            resume_text, st.session_state["jd_text"], 
                            st.session_state.get("mandatory", ""), gap, alpha, beta, gamma,
                            skill_rarity_weights=skill_weights
                        )
                        results.append({
                            "Candidate": row.get("Name", f"Candidate {idx}"),
                            "Global Score": score,
                            "Semantic Match": info["sim"],
                            "Risk": info["pen_miss"]
                        })
                    
                    results_df = pd.DataFrame(results).sort_values(by="Global Score", ascending=False)
                    st.session_state["results_df"] = results_df
                    
                    # Display Table
                    st.dataframe(
                        results_df,
                        column_config={
                            "Global Score": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=0.5),
                        },
                        use_container_width=True, hide_index=True
                    )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# --- TAB 3: ANALYTICS ---
with tab3:
    # KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <h3>Total Applicants</h3>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">1,256</div>
            <div style="color: #10B981; font-weight: 600;">+12% from last period</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <h3>Qualified Candidates</h3>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">847</div>
            <div style="color: #64748B;">67.4% qualification rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <h3>Positions Filled</h3>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">124</div>
            <div style="color: #64748B;">14.6% hire rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Recruitment Pipeline Trend")
        st.caption("Monthly applicant and qualification metrics")
        # Dummy Data for visual match
        df_trend = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Applicants': [120, 140, 160, 190, 215, 240],
            'Qualified': [80, 95, 110, 125, 142, 155]
        })
        fig_trend = px.line(df_trend, x='Month', y=['Applicants', 'Qualified'], 
                            color_discrete_sequence=['#334155', '#10B981'], markers=True)
        fig_trend.update_layout(plot_bgcolor='white', paper_bgcolor='white', 
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with chart_col2:
        st.subheader("Position Outcomes Distribution")
        st.caption("Conversion metrics across pipeline stages")
        df_bar = pd.DataFrame({
            'Stage': ['Applied', 'Screened', 'Interviewed', 'Offer', 'Hired'],
            'Count': [1256, 847, 420, 150, 124]
        })
        fig_bar = px.bar(df_bar, x='Stage', y='Count',
                         color_discrete_sequence=['#0F766E']) # Teal
        fig_bar.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_bar, use_container_width=True)
