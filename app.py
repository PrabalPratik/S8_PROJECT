import streamlit as st
import pandas as pd
import yaml
import torch
import time
from models.t5_generator import T5JDGenerator
from models.bert_ranker import BertRanker
from utils.metrics import calculate_cvm
from models.attention_utils import compare_candidates

# --- Configuration & Setup ---
st.set_page_config(
    page_title="TalentAI | Research Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Config file not found. Please ensure 'config.yaml' exists.")
    st.stop()

# --- Model Loading ---
@st.cache_resource
def load_models():
    t5 = T5JDGenerator(model_name=config["models"]["t5_name"])
    ranker = BertRanker(model_name=config["models"]["bert_name"])
    return t5, ranker

with st.spinner("Initializing Research Engine..."):
    t5_model, bert_model = load_models()

# --- Sidebar: Controls & Physics ---
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=50)
    st.title("TalentAI")
    st.caption("v2.0 Enterprise Edition")
    st.markdown("---")

    # Navigation (Implicit via Tabs in Main, but visual cue here)
    st.markdown("### ⚙️ Ranking Physics")
    st.caption("Control how candidates are scored.")
    
    with st.expander("📚 What do these mean?", expanded=False):
        st.markdown("""
        **Alpha (Semantic Match)**:  
        How well the resume *meaning* matches the JD.  
        *High Alpha* = "Hire for potential/vibe."

        **Beta (Skill Penalty)**:  
        How strictly to punish missing *mandatory* skills.  
        *High Beta* = "Must check every box."

        **Gamma (Experience Gap)**:  
        Penalty for lacking years of experience.  
        """)

    # Sliders
    alpha = st.slider("Alpha (Semantic Match)", 0.0, 1.0, config["weights"]["alpha"])
    beta = st.slider("Beta (Skill Penalty)", 0.0, 1.0, config["weights"]["beta"])
    gamma = st.slider("Gamma (Exp. Gap)", 0.0, 1.0, config["weights"]["gamma"])

    # Total Weight Indicator (Like the screenshot)
    total = alpha + beta + gamma
    st.metric("Total Weight", f"{total:.1f}", delta="Normalized" if abs(total-1.0)<0.1 else "Unbalanced", delta_color="normal")
    
    st.markdown("---")
    st.caption("Backend: T5-Small + BERT-Uncased")

# --- Main Interface ---
# No huge title, just tabs like the screenshot
# st.title("TalentAI") 

# State Management
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "generated_role" not in st.session_state:
    st.session_state["generated_role"] = ""

# Layout: Breadcrumbs
st.caption(f"TalentAI > {st.session_state.get('generated_role', 'Dashboard')}")

# Tabs - The core navigation
tab1, tab2, tab3 = st.tabs(["✨ Generator", "📊 Ranker", "🔬 X-Ray Analysis"])

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
            if st.button("🚀 Generate Job Description", type="primary", use_container_width=True):
                with st.spinner("Synthesizing..."):
                    time.sleep(0.5)
                    jd_text = t5_model.generate(role, mandatory, optional, exp)
                    st.session_state["jd_text"] = jd_text
                    st.session_state["generated_role"] = role
                    st.session_state["mandatory"] = mandatory
                    st.session_state["exp"] = exp

    with col_preview:
        st.subheader("Generated Preview")
        with st.container(border=True):
            if st.session_state["jd_text"]:
                st.markdown(st.session_state["jd_text"])
                st.divider()
                # Footer Metrics
                cvm = calculate_cvm(st.session_state["jd_text"], st.session_state.get("mandatory", ""))
                c1, c2 = st.columns(2)
                c1.metric("Constraint Adherence", f"{(1-cvm):.0%}")
                c2.metric("Hallucination Rate", f"{cvm:.0%}", delta_color="inverse")
            else:
                st.info("Fill in the form and click Generate to see the job description preview")
                st.caption("Waiting for input...")
                for _ in range(8): st.write("") # Spacer to match height

# --- TAB 2: RANKER (Table View) ---
with tab2:
    # Sub-nav style header
    c_head, c_btn = st.columns([3, 1])
    c_head.subheader(f"Ranker")
    
    if st.session_state["jd_text"]:
        try:
            resumes_df = pd.read_csv("data/resumes.csv")
            
            results = []
            for idx, row in resumes_df.iterrows():
                resume_text = f"Skills: {row['Skills']}. Experience: {row['Experience (Years)']} years. Education: {row['Education']}."
                gap = max(0, st.session_state["exp"] - row['Experience (Years)'])
                score, info = bert_model.get_research_score(
                    resume_text, st.session_state["jd_text"], 
                    st.session_state.get("mandatory", ""), gap, alpha, beta, gamma
                )
                results.append({
                    "ID": idx, "Candidate": row.get("Name", f"Unknown"),
                    "Global Score": score, "Semantic Match": info["sim"], 
                    "Risk Factor": info["pen_miss"], "Raw Resume": resume_text
                })
            
            results_df = pd.DataFrame(results).sort_values(by="Global Score", ascending=False)
            st.session_state["results_df"] = results_df

             # High-Fidelity Table
            st.dataframe(
                results_df[["Candidate", "Global Score", "Semantic Match", "Risk Factor"]],
                column_config={
                    "Global Score": st.column_config.ProgressColumn(
                        "Global Score", format="%.0f%%", min_value=0, max_value=0.5
                    ),
                    "Semantic Match": st.column_config.ProgressColumn(
                        "Semantic", format="%.0f%%", min_value=0, max_value=1
                    ),
                    "Risk Factor": st.column_config.ProgressColumn(
                        "Risk (Skill Gap)", format="%.0f%%", min_value=0, max_value=1
                    )
                },
                use_container_width=True, hide_index=True, height=400
            )
        except Exception as e:
            st.error(f"Data Error: {e}")
    else:
        st.warning("Generate a JD first.")

# --- TAB 3: ANALYSIS (Comparison) ---
with tab3:
    st.subheader("X-Ray Analysis")
    if "results_df" in st.session_state:
        df = st.session_state["results_df"]
        
        # Selectors next to chart
        c_sel, c_chart = st.columns([1, 2])
        
        with c_sel:
            st.markdown("##### Candidate Comparison")
            cand_a_name = st.selectbox("Candidate A (Blue)", df["Candidate"], index=0)
            cand_b_name = st.selectbox("Candidate B (Red)", df["Candidate"], index=1 if len(df)>1 else 0)
            
            if st.button("Run Token Analysis"):
                row_a = df[df["Candidate"] == cand_a_name].iloc[0]
                row_b = df[df["Candidate"] == cand_b_name].iloc[0]
                
                features = compare_candidates(bert_model, bert_model.tokenizer, st.session_state["jd_text"], row_a["Raw Resume"], row_b["Raw Resume"])
                st.session_state["features"] = features
                st.session_state["comparison"] = (cand_a_name, cand_b_name)

        with c_chart:
            if "features" in st.session_state:
                # Bar Chart
                features = st.session_state["features"]
                plot_df = pd.DataFrame(features, columns=["Token", "Advantage"])
                
                # Make it look like the Red/Blue chart
                # Actually compare_candidates returns A's advantage. 
                
                st.caption(f"Token Advantages: {st.session_state['comparison'][0]} vs {st.session_state['comparison'][1]}")
                st.bar_chart(plot_df.set_index("Token"), color=["#2563EB"])
                
    else:
        st.info("Rank candidates to unlock X-Ray.")
