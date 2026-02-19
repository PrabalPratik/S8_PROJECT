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

# Security imports
from utils.security import (
    InputSanitizer,
    get_rate_limiter,
    escape_html
)

# Import Novel Features
try:
    from utils.hallucination_detector import HallucinationDetector
    from models.drift_detector import RoleDriftDetector
    from utils.sensitivity_analysis import SensitivityAnalyzer
    from utils.skill_rarity import SkillRarityCalculator
    from utils.fairness_auditor import FairnessAuditor, generate_synthetic_demographics
    from models.constraint_graph import ConstraintCoverageGraph
    from utils.feedback_logger import get_feedback_logger
    from utils.data_persistence import get_data_persistence
    # New modules
    from utils.screening import ScreeningEngine
    from models.question_generator import QuestionGenerator
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
FAIRNESS_ENABLED = features.get("fairness_enabled", False)
FAIRNESS_THRESHOLD = features.get("fairness_threshold", 0.8)

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
                # Link to CCG for correction suggestions
                try:
                    ccg = ConstraintCoverageGraph(skill_corpus)
                    modules["hri"].set_correction_graph(ccg)
                    modules["ccg"] = ccg
                except Exception as ccg_e:
                    print(f"CCG linking failed: {ccg_e}")
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
        if FAIRNESS_ENABLED:
            try:
                modules["fairness"] = FairnessAuditor(dir_threshold=FAIRNESS_THRESHOLD)
            except Exception as e:
                print(f"Fairness auditor init failed: {e}")
            except Exception as e:
                print(f"Fairness auditor init failed: {e}")
        
        # Initialize Screening & Questions (Always enabled if available)
        try:
            modules["screening"] = ScreeningEngine()
            modules["questions"] = QuestionGenerator()
        except Exception as e:
            print(f"Screening/Questions init failed: {e}")

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
    

    # Novel Feature Toggles
    st.markdown("### 🔬 Research Features")
    show_hri = st.checkbox("Show HRI Score", value=HRI_ENABLED and "hri" in novel_modules)
    show_drift = st.checkbox("Show Drift Detection", value=DRIFT_ENABLED and "drift" in novel_modules)
    show_stability = st.checkbox("Show Rank Stability", value=SENSITIVITY_ENABLED and "sensitivity" in novel_modules)
    use_srw = st.checkbox("Use Skill Rarity Weighting", value=SRW_ENABLED and "srw" in novel_modules)
    use_banding = st.checkbox("Enable Smart Screening Bands", value=True)


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
            role_input = st.text_input("Target Role", "Senior Backend Engineer", max_chars=100)
            exp = st.slider("Experience Required", 0, 15, 8, format="%d years")
            
            st.write("")
            mandatory_input = st.text_area("Mandatory Skills", "Python, Django, AWS", height=100, max_chars=500)
            optional_input = st.text_input("Optional Skills", "Kubernetes, PostgreSQL", max_chars=300)
            
            # Sanitize inputs
            role, role_valid = InputSanitizer.sanitize_role(role_input)
            mandatory, mandatory_valid = InputSanitizer.sanitize_skills(mandatory_input)
            optional, _ = InputSanitizer.sanitize_skills(optional_input)
            
            # Show validation warnings
            if role_input and not role_valid:
                st.warning("⚠️ Role name contains invalid characters")
            if mandatory_input and not mandatory_valid:
                st.warning("⚠️ Some skills contain invalid characters")
            
            st.write("")
            if st.button("Generate Job Description", type="primary", use_container_width=True):
                # Rate limiting check
                rate_limiter = get_rate_limiter()
                session_key = f"jd_gen_{st.session_state.get('session_id', 'default')}"
                allowed, remaining = rate_limiter.check_rate_limit(session_key, max_requests=10, window_seconds=60)
                
                if not allowed:
                    wait_time = rate_limiter.get_wait_time(session_key)
                    st.error(f"⏳ Rate limit reached. Please wait {wait_time} seconds.")
                elif not role_valid:
                    st.error("Please enter a valid role name.")
                elif not mandatory:
                    st.error("Please enter at least one mandatory skill.")
                else:
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
                    
                    # Save JD to persistent storage
                    try:
                        dp = get_data_persistence()
                        dp.save_jd(
                            role=role,
                            experience=exp,
                            mandatory_skills=mandatory,
                            optional_skills=optional,
                            jd_text=jd_text,
                            hri_score=st.session_state.get("hri_score"),
                            drift_score=st.session_state.get("drift_score")
                        )
                    except Exception as e:
                        print(f"JD persistence failed: {e}")
                    
                    # Rerun to refresh UI
                    st.rerun()


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
                
                # HRI Correction Suggestions
                hri_details = st.session_state.get("hri_details", {})
                corrections = hri_details.get("corrections", {})
                if corrections:
                    st.divider()
                    st.markdown("**🛠️ Suggested Corrections:**")
                    for unknown, suggestions in corrections.items():
                        suggestion_text = ", ".join([f"{s[0]} ({s[1]:.0%})" for s in suggestions])
                        st.caption(f"• Replace *{unknown}* with: {suggestion_text}")
                    
                    if st.button("✅ Apply All Corrections", key="apply_corrections"):
                        if "hri" in novel_modules:
                            corrected_jd = novel_modules["hri"].apply_corrections(
                                st.session_state["jd_text"], corrections
                            )
                            st.session_state["jd_text"] = corrected_jd
                            # Re-run HRI on corrected text
                            hri_score, hri_details = novel_modules["hri"].calculate_hri(
                                corrected_jd, st.session_state.get("mandatory", "").split(",")
                            )
                            st.session_state["hri_score"] = hri_score
                            st.session_state["hri_details"] = hri_details
                            st.rerun()
                
                # Human Feedback Section
                st.divider()
                st.markdown("**📝 Rate this JD:**")
                fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])
                
                with fb_col1:
                    if st.button("👍 Good", key="jd_thumbs_up", use_container_width=True):
                        try:
                            logger = get_feedback_logger()
                            logger.log_feedback(
                                "jd_quality", 1,
                                {
                                    "role": st.session_state.get("generated_role", ""),
                                    "skills": st.session_state.get("mandatory", ""),
                                    "hri_score": st.session_state.get("hri_score", ""),
                                    "drift_score": st.session_state.get("drift_score", "")
                                }
                            )
                            st.success("Thanks for feedback!")
                        except Exception:
                            st.success("Feedback noted!")
                
                with fb_col2:
                    if st.button("👎 Needs Work", key="jd_thumbs_down", use_container_width=True):
                        try:
                            logger = get_feedback_logger()
                            logger.log_feedback(
                                "jd_quality", -1,
                                {
                                    "role": st.session_state.get("generated_role", ""),
                                    "skills": st.session_state.get("mandatory", ""),
                                    "hri_score": st.session_state.get("hri_score", ""),
                                    "drift_score": st.session_state.get("drift_score", "")
                                }
                            )
                            st.warning("Thanks! We'll improve.")
                        except Exception:
                            st.warning("Feedback noted!")
                
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
                        # Use Resume_str if available (new processed dataset), else fallback to old format
                        if 'Resume_str' in resumes_df.columns:
                            resume_text = row['Resume_str'][:2000]  # Truncate for BERT max length
                            # Extract experience heuristically (look for "X years" pattern)
                            import re
                            exp_match = re.search(r'(\d+)\s*(?:years?|yrs?)', str(resume_text).lower())
                            candidate_exp = int(exp_match.group(1)) if exp_match else 3  # Default 3 years
                        else:
                            resume_text = f"Skills: {row.get('Skills', '')}. Experience: {row.get('Experience (Years)', 3)} years."
                            candidate_exp = row.get('Experience (Years)', 3)
                        
                        gap = max(0, st.session_state.get("exp", 5) - candidate_exp)
                        
                        score, info = bert_model.get_research_score(
                            resume_text, st.session_state["jd_text"], 
                            st.session_state.get("mandatory", ""), gap, alpha, beta, gamma,
                            skill_rarity_weights=skill_weights
                        )
                        results.append({
                            "Candidate": row.get("Candidate", row.get("Name", f"Candidate {idx}")),
                            "Global Score": score,
                            "Semantic Match": info["sim"],
                            "Risk": info["pen_miss"],
                            "Missing Skills": info["missing_skills"],
                            "Experience Gap": info["pen_exp"],
                            "Resume Text": resume_text,
                            "Exp Years": candidate_exp
                        })
                    
                    results_df = pd.DataFrame(results).sort_values(by="Global Score", ascending=False)
                    st.session_state["results_df"] = results_df
                    
                    # Save rankings to persistent storage
                    try:
                        dp = get_data_persistence()
                        dp.save_rankings(
                            role=st.session_state.get("generated_role", ""),
                            mandatory_skills=st.session_state.get("mandatory", ""),
                            rankings_df=results_df
                        )
                    except Exception as e:
                        print(f"Rankings persistence failed: {e}")
                    
                    
                    # --- Smart Screening: Selected / Rejected View ---
                    if use_banding and "screening" in novel_modules:
                        sorted_results = results_df.to_dict('records')
                        grouped = novel_modules["screening"].group_candidates(sorted_results)
                        
                        # Split into Selected (Prime + Contender) and Rejected (Potential + Mismatch)
                        selected = grouped.get("Prime", []) + grouped.get("Contender", [])
                        rejected = grouped.get("Potential", []) + grouped.get("Mismatch", [])
                        
                        # Sort each group high → low
                        selected.sort(key=lambda x: x["Global Score"], reverse=True)
                        rejected.sort(key=lambda x: x["Global Score"], reverse=True)
                        
                        # Store for View Report (outside spinner)
                        st.session_state["screening_selected"] = selected
                        st.session_state["screening_rejected"] = rejected

                    else:
                        # Fallback to original Table View
                        st.dataframe(
                            results_df,
                            column_config={
                                "Global Score": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=0.5),
                            },
                            use_container_width=True, hide_index=True
                        )

                    
                    # Feedback for rankings
                    st.divider()
                    st.markdown("**📝 Rate these rankings:**")
                    rank_fb1, rank_fb2, _ = st.columns([1, 1, 2])
                    
                    with rank_fb1:
                        if st.button("👍 Accurate", key="rank_thumbs_up", use_container_width=True):
                            try:
                                logger = get_feedback_logger()
                                logger.log_feedback(
                                    "ranking_accuracy", 1,
                                    {"role": st.session_state.get("generated_role", "")}
                                )
                                st.success("Thanks!")
                            except Exception:
                                st.success("Noted!")
                    
                    with rank_fb2:
                        if st.button("👎 Inaccurate", key="rank_thumbs_down", use_container_width=True):
                            try:
                                logger = get_feedback_logger()
                                logger.log_feedback(
                                    "ranking_accuracy", -1,
                                    {"role": st.session_state.get("generated_role", "")}
                                )
                                st.warning("We'll improve!")
                            except Exception:
                                st.warning("Noted!")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # --- Persistent Screening Display (renders even without re-clicking Generate) ---
    if use_banding and "screening_selected" in st.session_state:
        selected = st.session_state["screening_selected"]
        rejected = st.session_state["screening_rejected"]
        
        st.markdown("---")
        st.markdown("### 🎯 Smart Screening Results")
        st.caption(f"**{len(selected)}** shortlisted  •  **{len(rejected)}** not shortlisted  •  Sorted highest → lowest")
        
        # --- ✅ SELECTED CANDIDATES ---
        st.markdown("#### ✅ Shortlisted for Interview")
        if selected:
            for cand in selected:
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    band = cand.get("band_info", {}).get("band", "")
                    st.markdown(f"🟢 **{cand['Candidate']}** — *{band}*")
                    st.caption(f"Score: {cand['Global Score']:.2f} | Semantic: {cand['Semantic Match']:.2f}")
                with c2:
                    st.markdown(f"**{cand['Global Score']:.0%}**")
                with c3:
                    if st.button("📄 View Report", key=f"sel_{cand['Candidate']}"):
                        st.session_state["selected_candidate"] = cand
                        st.rerun()
                st.divider()
        else:
            st.info("No candidates met the shortlist threshold for this role.")
        
        # --- ❌ REJECTED CANDIDATES ---
        st.markdown("#### ❌ Not Shortlisted")
        if rejected:
            for cand in rejected:
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    band = cand.get("band_info", {}).get("band", "")
                    st.markdown(f"🔴 **{cand['Candidate']}** — *{band}*")
                    st.caption(f"Score: {cand['Global Score']:.2f} | Semantic: {cand['Semantic Match']:.2f}")
                with c2:
                    st.markdown(f"**{cand['Global Score']:.0%}**")
                with c3:
                    if st.button("📄 View Report", key=f"rej_{cand['Candidate']}"):
                        st.session_state["selected_candidate"] = cand
                        st.rerun()
                st.divider()

    # --- VIEW REPORT PANEL ---
    if "selected_candidate" in st.session_state:
        sel = st.session_state["selected_candidate"]
        st.markdown("---")
        
        # Header with status
        band = sel.get("band_info", {}).get("band", "Unknown")
        if band in ["Prime", "Contender"]:
            st.success(f"### ✅ Candidate Report: {sel['Candidate']}  —  SHORTLISTED ({band})")
        else:
            st.error(f"### ❌ Candidate Report: {sel['Candidate']}  —  NOT SHORTLISTED ({band})")
        
        # Score Breakdown
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Semantic Match", f"{sel['Semantic Match']:.0%}")
        col_s2.metric("Skill Penalty", f"{sel['Risk']:.0%}", delta_color="inverse")
        col_s3.metric("Experience Gap", f"{sel['Experience Gap']:.0%}", delta_color="inverse")
        
        # Feedback
        if "screening" in novel_modules:
            feedback = novel_modules["screening"].generate_feedback(
                {
                    "semantic_score": sel["Semantic Match"],
                    "missing_skills": sel.get("Missing Skills", []),
                    "exp_years": sel.get("Exp Years", 0),
                    "Global Score": sel["Global Score"]
                },
                {
                    "mandatory_skills": st.session_state.get("mandatory", "").split(","),
                    "min_exp": st.session_state.get("exp", 0)
                }
            )
            st.info(feedback)
        
        # Missing Skills
        missing = sel.get("Missing Skills", [])
        if missing:
            st.warning(f"**Missing Mandatory Skills:** {', '.join(missing)}")
        else:
            st.success("**All mandatory skills matched.**")
        
        # Interview Questions
        if "questions" in novel_modules:
            st.markdown("**🎤 Suggested Interview Questions:**")
            import re
            projects = []
            proj_matches = re.findall(r'(?:Project|System):\s*([A-Za-z0-9 ]+)', sel.get("Resume Text", ""), re.IGNORECASE)
            if proj_matches:
                for p in proj_matches[:2]:
                    projects.append({"name": p.strip(), "tech": []})
            else:
                projects.append({"name": "Recent Work", "tech": []})

            questions = novel_modules["questions"].generate_questions({
                "projects": projects,
                "experience_years": sel.get("Exp Years", 0),
                "skills": st.session_state.get("mandatory", "").split(",")
            })
            for i, q in enumerate(questions, 1):
                st.markdown(f"**Q{i}.** {q}")
        
        # Resume Preview
        with st.expander("📃 Resume Preview", expanded=False):
            st.text(sel.get("Resume Text", "No resume text available.")[:1500])
        
        if st.button("✖ Close Report", type="primary"):
            del st.session_state["selected_candidate"]
            st.rerun()

# --- TAB 3: ANALYTICS ---
with tab3:
    st.markdown("### 📊 Recruitment Analytics Dashboard")
    st.caption("Real-time metrics based on your candidate evaluations")
    
    # Load actual resume data for base metrics
    try:
        resumes_df = pd.read_csv(config.get("data", {}).get("resumes", "data/resumes.csv"))
        total_applicants = len(resumes_df)
    except:
        resumes_df = None
        total_applicants = 0
    
    # Get ranking results from session state
    results_df = st.session_state.get("results_df", None)
    
    # Calculate dynamic metrics
    if results_df is not None and len(results_df) > 0:
        qualified_count = len(results_df[results_df["Global Score"] >= 0.2])
        qualification_rate = (qualified_count / len(results_df) * 100) if len(results_df) > 0 else 0
        avg_score = results_df["Global Score"].mean()
        top_score = results_df["Global Score"].max()
    else:
        qualified_count = 0
        qualification_rate = 0
        avg_score = 0
        top_score = 0
    
    # Track session metrics
    if "analytics_history" not in st.session_state:
        st.session_state["analytics_history"] = {
            "evaluations": [],
            "jds_generated": 0,
            "rankings_run": 0
        }
    
    # KPI Cards - Dynamic Data (values are computed, not user-input, so safe for display)
    col1, col2, col3 = st.columns(3)
    
    # Note: These values are derived from internal computations, not raw user input
    # Safe to display in HTML as they are numeric/computed values
    safe_total = escape_html(str(total_applicants))
    safe_qualified = escape_html(str(qualified_count))
    safe_rate = escape_html(f"{qualification_rate:.1f}")
    safe_avg = escape_html(f"{avg_score:.2f}")
    safe_top = escape_html(f"{top_score:.2f}")
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Total Applicants</h3>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">{safe_total}</div>
            <div style="color: #64748B;">From resumes.csv</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        rate_color = '#10B981' if qualification_rate >= 50 else '#64748B'
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Qualified Candidates</h3>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">{safe_qualified}</div>
            <div style="color: {rate_color};">{safe_rate}% qualification rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Average Score</h3>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">{safe_avg}</div>
            <div style="color: #64748B;">Top: {safe_top}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts based on actual data
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Candidate Score Distribution")
        st.caption("Score distribution from current ranking")
        
        if results_df is not None and len(results_df) > 0:
            fig_hist = px.histogram(
                results_df, x="Global Score", nbins=10,
                color_discrete_sequence=['#334155']
            )
            fig_hist.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis_title="Score", yaxis_title="Count"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Run candidate rankings to see score distribution")
        
    with chart_col2:
        st.subheader("Semantic vs Risk Analysis")
        st.caption("Candidate performance breakdown")
        
        if results_df is not None and len(results_df) > 0:
            fig_scatter = px.scatter(
                results_df, x="Semantic Match", y="Risk",
                hover_data=["Candidate", "Global Score"],
                color="Global Score",
                color_continuous_scale="RdYlGn"
            )
            fig_scatter.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis_title="Semantic Match Score", yaxis_title="Risk Score"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Run candidate rankings to see analysis")
    
    # Experience Distribution from actual resume data
    if resumes_df is not None and "Experience (Years)" in resumes_df.columns:
        st.markdown("---")
        st.subheader("Applicant Pool Analysis")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.caption("Experience Distribution")
            fig_exp = px.histogram(
                resumes_df, x="Experience (Years)", nbins=8,
                color_discrete_sequence=['#0F766E']
            )
            fig_exp.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_exp, use_container_width=True)
        
        with exp_col2:
            st.caption("Top Skills in Pool")
            if "Skills" in resumes_df.columns:
                # Parse and count skills
                all_skills = []
                for skills in resumes_df["Skills"].dropna():
                    all_skills.extend([s.strip() for s in str(skills).split(",")])
                
                skill_counts = pd.Series(all_skills).value_counts().head(8)
                df_skills = pd.DataFrame({"Skill": skill_counts.index, "Count": skill_counts.values})
                
                fig_skills = px.bar(
                    df_skills, x="Count", y="Skill", orientation='h',
                    color_discrete_sequence=['#334155']
                )
                fig_skills.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig_skills, use_container_width=True)
    
    # --- Fairness Metrics Section ---
    if "fairness" in novel_modules and st.session_state.get("results_df") is not None:
        st.markdown("---")
        st.subheader("⚖️ Fairness Audit")
        st.caption("Demographic parity and exposure analysis for current rankings")
        
        results_df = st.session_state["results_df"]
        
        # Generate synthetic demographics for demo
        demographics_df = generate_synthetic_demographics(results_df["Candidate"].tolist())
        
        # Run audit
        auditor = novel_modules["fairness"]
        report = auditor.audit(results_df, demographics_df, ["Gender", "Age_Group"], top_k=3)
        
        # Display metrics
        fair_col1, fair_col2, fair_col3 = st.columns(3)
        
        with fair_col1:
            overall_status = report["summary"]["overall_status"]
            st.metric("Overall Status", overall_status)
        
        with fair_col2:
            if "Gender" in report["by_attribute"]:
                gender_dir = report["by_attribute"]["Gender"]["dir"]["dir_score"]
                st.metric("Gender DIR", f"{gender_dir:.1%}", 
                         delta="Compliant" if gender_dir >= FAIRNESS_THRESHOLD else "Below 80%",
                         delta_color="normal" if gender_dir >= FAIRNESS_THRESHOLD else "inverse")
        
        with fair_col3:
            if "Age_Group" in report["by_attribute"]:
                age_dir = report["by_attribute"]["Age_Group"]["dir"]["dir_score"]
                st.metric("Age Group DIR", f"{age_dir:.1%}",
                         delta="Compliant" if age_dir >= FAIRNESS_THRESHOLD else "Below 80%",
                         delta_color="normal" if age_dir >= FAIRNESS_THRESHOLD else "inverse")
        
        # Recommendations
        if report["recommendations"]:
            with st.expander("📌 Recommendations", expanded=False):
                for rec in report["recommendations"]:
                    st.warning(rec)

