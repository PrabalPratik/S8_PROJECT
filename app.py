import streamlit as st
import pandas as pd
import yaml
import torch
import time
from models.t5_generator import T5JDGenerator
from models.bert_ranker import BertRanker
from utils.metrics import calculate_cvm
from models.attention_utils import compare_candidates

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
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=50)
    st.title("TalentAI")
    st.caption("v3.0 Research Edition")
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
    
    # Novel Feature Toggles
    st.markdown("### 🔬 Research Features")
    show_hri = st.checkbox("Show HRI Score", value=HRI_ENABLED and "hri" in novel_modules)
    show_drift = st.checkbox("Show Drift Detection", value=DRIFT_ENABLED and "drift" in novel_modules)
    show_stability = st.checkbox("Show Rank Stability", value=SENSITIVITY_ENABLED and "sensitivity" in novel_modules)
    use_srw = st.checkbox("Use Skill Rarity Weighting", value=SRW_ENABLED and "srw" in novel_modules)
    
    st.markdown("---")
    st.caption("Backend: T5-Small + BERT-Uncased")
    if novel_modules:
        st.caption(f"Active Features: {len(novel_modules)}")

# --- Main Interface ---
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
with tab2:
    c_head, c_btn = st.columns([3, 1])
    c_head.subheader(f"Ranker")
    
    if st.session_state["jd_text"]:
        try:
            resumes_df = pd.read_csv(config.get("data", {}).get("resumes", "data/resumes.csv"))
            
            # Get skill rarity weights if SRW enabled
            skill_weights = None
            if use_srw and "srw" in novel_modules:
                mandatory_skills = [s.strip().lower() for s in st.session_state.get("mandatory", "").split(",")]
                skill_weights = {}
                for skill in mandatory_skills:
                    weight, _ = novel_modules["srw"].get_rarity_weight(skill)
                    skill_weights[skill] = weight
            
            results = []
            candidates_for_sensitivity = []
            
            for idx, row in resumes_df.iterrows():
                resume_text = f"Skills: {row['Skills']}. Experience: {row['Experience (Years)']} years. Education: {row['Education']}."
                gap = max(0, st.session_state["exp"] - row['Experience (Years)'])
                
                score, info = bert_model.get_research_score(
                    resume_text, st.session_state["jd_text"], 
                    st.session_state.get("mandatory", ""), gap, alpha, beta, gamma,
                    skill_rarity_weights=skill_weights
                )
                
                result = {
                    "ID": idx, 
                    "Candidate": row.get("Name", f"Candidate {idx}"),
                    "Global Score": score, 
                    "Semantic Match": info["sim"], 
                    "Risk Factor": info["pen_miss"],
                    "Raw Resume": resume_text
                }
                results.append(result)
                
                candidates_for_sensitivity.append({
                    "name": row.get("Name", f"Candidate {idx}"),
                    "resume_text": resume_text,
                    "experience_gap": gap
                })
            
            results_df = pd.DataFrame(results).sort_values(by="Global Score", ascending=False)
            
            # Calculate rank stability if enabled
            if show_stability and "sensitivity" in novel_modules and len(candidates_for_sensitivity) > 1:
                with st.spinner("Calculating rank stability..."):
                    analyzer = novel_modules["sensitivity"]
                    stability_results = analyzer.quick_stability_check(
                        bert_model, candidates_for_sensitivity[:20],  # Limit to top 20 for speed
                        st.session_state["jd_text"],
                        st.session_state.get("mandatory", ""),
                        n_quick=SENSITIVITY_SIMS
                    )
                    results_df["Stability"] = results_df["Candidate"].map(stability_results).fillna(50.0)
            
            st.session_state["results_df"] = results_df

            # Display columns
            display_cols = ["Candidate", "Global Score", "Semantic Match", "Risk Factor"]
            if "Stability" in results_df.columns:
                display_cols.append("Stability")
            
            # Column config
            col_config = {
                "Global Score": st.column_config.ProgressColumn(
                    "Global Score", format="%.0f%%", min_value=0, max_value=0.5
                ),
                "Semantic Match": st.column_config.ProgressColumn(
                    "Semantic", format="%.0f%%", min_value=0, max_value=1
                ),
                "Risk Factor": st.column_config.ProgressColumn(
                    "Risk (Skill Gap)", format="%.0f%%", min_value=0, max_value=1
                )
            }
            
            if "Stability" in results_df.columns:
                col_config["Stability"] = st.column_config.ProgressColumn(
                    "Rank Stability", format="%.0f%%", min_value=0, max_value=100
                )
            
            st.dataframe(
                results_df[display_cols],
                column_config=col_config,
                use_container_width=True, hide_index=True, height=400
            )
            
            # SRW info
            if use_srw and skill_weights:
                with st.expander("💎 Skill Rarity Weights"):
                    for skill, weight in sorted(skill_weights.items(), key=lambda x: x[1], reverse=True):
                        if weight >= 0.7:
                            st.markdown(f"🔴 **{skill}**: {weight:.2f} (Very Rare)")
                        elif weight >= 0.5:
                            st.markdown(f"🟠 **{skill}**: {weight:.2f} (Rare)")
                        elif weight >= 0.3:
                            st.markdown(f"🟡 **{skill}**: {weight:.2f} (Uncommon)")
                        else:
                            st.markdown(f"🟢 **{skill}**: {weight:.2f} (Common)")
                            
        except Exception as e:
            st.error(f"Data Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("Generate a JD first.")

# --- TAB 3: ANALYSIS (Comparison + Features) ---
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
                
                features_result = compare_candidates(
                    bert_model, bert_model.tokenizer, 
                    st.session_state["jd_text"], 
                    row_a["Raw Resume"], row_b["Raw Resume"]
                )
                st.session_state["features"] = features_result
                st.session_state["comparison"] = (cand_a_name, cand_b_name)

        with c_chart:
            if "features" in st.session_state:
                features_data = st.session_state["features"]
                plot_df = pd.DataFrame(features_data, columns=["Token", "Advantage"])
                
                st.caption(f"Token Advantages: {st.session_state['comparison'][0]} vs {st.session_state['comparison'][1]}")
                st.bar_chart(plot_df.set_index("Token"), color=["#2563EB"])
                
    else:
        st.info("Rank candidates to unlock X-Ray.")
