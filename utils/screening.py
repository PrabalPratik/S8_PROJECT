from typing import List, Dict, Any
import numpy as np

class ScreeningEngine:
    """
    Handles candidate screening logic including:
    1. Score Banding (grouping similar scores)
    2. Feedback Generation (explaining the score)
    """
    
    def __init__(self, band_threshold: float = 0.02):
        self.band_threshold = band_threshold

    def categorize_candidate(self, score: float, missing_skills: int) -> Dict[str, Any]:
        """
        Determines the screening category/band for a candidate.
        """
        if score >= 0.85 and missing_skills == 0:
            return {"band": "Prime", "color": "green", "icon": "🟢"}
        elif (score >= 0.70) or (score >= 0.85 and missing_skills <= 1):
            return {"band": "Contender", "color": "yellow", "icon": "🟡"}
        elif score >= 0.40:
            return {"band": "Potential", "color": "gray", "icon": "⚪"}
        else:
            return {"band": "Mismatch", "color": "red", "icon": "🔴"}

    def generate_feedback(self, candidate_data: Dict[str, Any], jd_requirements: Dict[str, Any]) -> str:
        """
        Generates natural language feedback explaining the candidate's standing.
        
        expected candidate_data keys:
        - semantic_score (float)
        - missing_skills (list of str)
        - exp_years (int)
        
        expected jd_requirements keys:
        - mandatory_skills (list of str)
        - min_exp (int)
        """
        feedback = []
        
        # 1. Semantic Feedback
        semantic = candidate_data.get("semantic_score", 0)
        if semantic > 0.8:
            feedback.append("Strong semantic alignment with the role description.")
        elif semantic > 0.6:
            feedback.append("Good understanding of the core concepts.")
        else:
            feedback.append("Profile content has low overlap with role requirements.")

        # 2. Skills Feedback
        missing = candidate_data.get("missing_skills", [])
        if not missing:
            feedback.append("Matches all mandatory skill requirements.")
        else:
            skills_str = ", ".join(missing[:3])
            feedback.append(f"Missing specific mandatory skills: {skills_str}.")

        # 3. Experience Feedback
        cand_exp = candidate_data.get("exp_years", 0)
        req_exp = jd_requirements.get("min_exp", 0)
        
        if cand_exp >= req_exp:
            feedback.append(f"Meets experience requirement ({cand_exp} years).")
        else:
            gap = req_exp - cand_exp
            feedback.append(f"Experience gap of {gap} years.")

        # Final Decision Prefix
        decision = ""
        category_band = self.categorize_candidate(candidate_data.get("Global Score", 0), len(missing))["band"]
        
        if category_band in ["Prime", "Contender"]:
            decision = "**✅ SELECTION REASONING:** Candidate is qualified based on high semantic match and skill coverage. "
        elif category_band == "Potential":
            decision = "**⚠️ REVIEW REASONING:** Candidate has potential but falls short on specific requirements. "
        else:
            decision = "**⛔ REJECTION REASONING:** Candidate does not meet the core mandatory criteria for this role. "

        return decision + " ".join(feedback)

    def group_candidates(self, ranked_candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups a list of ranked candidates into bands.
        """
        grouped = {
            "Prime": [],
            "Contender": [],
            "Potential": [],
            "Mismatch": []
        }
        
        for cand in ranked_candidates:
            # logic to determine missing skills count from candidate data if not pre-calculated
            # assuring 'missing_skills_count' exists in cand dict
            missing_count = len(cand.get("missing_skills", []))
            
            category = self.categorize_candidate(cand["Global Score"], missing_count)
            cand["band_info"] = category
            grouped[category["band"]].append(cand)
            
        return grouped
