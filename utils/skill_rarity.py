"""
Skill Rarity Weighting (SRW) - Novel Feature F5

Computes inverse document frequency for skills and applies weighted penalties.
Rare, high-value skills get boosted penalties, preventing generic candidates
from ranking high on specialized roles.
"""

import re
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class SkillRarityCalculator:
    """
    Calculates skill rarity weights based on corpus frequency.
    
    Rare skills get higher weights, making missing them more penalizing.
    This prevents candidates with only common skills from ranking high.
    """
    
    def __init__(self, skill_corpus_path: Optional[str] = None):
        """
        Initialize the Skill Rarity Calculator.
        
        Args:
            skill_corpus_path: Path to CSV file containing job skills data
        """
        self.skill_frequency: Dict[str, int] = defaultdict(int)
        self.skill_idf: Dict[str, float] = {}
        self.total_documents = 0
        self.max_idf = 1.0
        
        if skill_corpus_path:
            self._compute_skill_rarity(skill_corpus_path)
    
    def _parse_skill_list(self, skill_string: str) -> List[str]:
        """Parse skill string into list of normalized skills."""
        if pd.isna(skill_string) or not skill_string:
            return []
        
        skill_string = str(skill_string)
        if skill_string.startswith('[') and skill_string.endswith(']'):
            skill_string = skill_string[1:-1]
            skills = re.split(r"',\s*'|',\s*\"|\", '|\"', '|\", \"", skill_string)
            skills = [s.strip().strip("'\"").lower() for s in skills]
        else:
            skills = [s.strip().lower() for s in skill_string.split(',')]
        
        return [s for s in skills if s and len(s) > 1]
    
    def _compute_skill_rarity(self, corpus_path: str) -> None:
        """Compute IDF-based rarity weights for all skills."""
        print(f"Computing skill rarity weights from {corpus_path}...")
        
        try:
            df = pd.read_csv(corpus_path)
        except Exception as e:
            print(f"Warning: Could not load skill corpus: {e}")
            return
        
        # Find skill column
        skill_col = None
        for col in df.columns:
            if 'skill' in col.lower():
                skill_col = col
                break
        
        if not skill_col:
            print("Warning: No skill column found in corpus")
            return
        
        # Count document frequency for each skill
        doc_frequency: Dict[str, int] = defaultdict(int)
        
        for idx, row in df.iterrows():
            skills = self._parse_skill_list(row.get(skill_col, ''))
            if not skills:
                continue
            
            self.total_documents += 1
            seen = set()
            
            for skill in skills:
                self.skill_frequency[skill] += 1
                if skill not in seen:
                    doc_frequency[skill] += 1
                    seen.add(skill)
        
        # Calculate IDF (Inverse Document Frequency)
        # IDF = log(total_docs / (doc_freq + 1))
        for skill, freq in doc_frequency.items():
            idf = math.log(self.total_documents / (freq + 1)) + 1
            self.skill_idf[skill] = idf
        
        if self.skill_idf:
            self.max_idf = max(self.skill_idf.values())
        
        print(f"Rarity computed: {len(self.skill_idf)} skills, max IDF: {self.max_idf:.2f}")
    
    def get_rarity_weight(self, skill: str) -> Tuple[float, str]:
        """
        Get rarity weight for a single skill.
        
        Returns:
            Tuple of (weight, rarity_category)
            weight: 0.0-1.0, higher = rarer
        """
        skill = skill.lower().strip()
        
        if skill not in self.skill_idf:
            # Unknown skill - could be very rare or a typo
            return 0.8, "Unknown (potentially rare)"
        
        # Normalize IDF to 0-1 range
        normalized_weight = self.skill_idf[skill] / self.max_idf if self.max_idf > 0 else 0.5
        
        # Determine category
        if normalized_weight >= 0.7:
            category = "🔴 Very Rare"
        elif normalized_weight >= 0.5:
            category = "🟠 Rare"
        elif normalized_weight >= 0.3:
            category = "🟡 Uncommon"
        else:
            category = "🟢 Common"
        
        return normalized_weight, category
    
    def get_weighted_penalties(
        self,
        missing_skills: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted penalty for missing skills.
        
        Args:
            missing_skills: List of skills the candidate is missing
            
        Returns:
            Tuple of (total_weighted_penalty, skill_weights_dict)
        """
        if not missing_skills:
            return 0.0, {}
        
        skill_weights = {}
        total_weight = 0.0
        
        for skill in missing_skills:
            weight, _ = self.get_rarity_weight(skill)
            skill_weights[skill] = weight
            total_weight += weight
        
        # Normalize to 0-1 range
        max_possible = len(missing_skills)  # If all skills had weight 1.0
        normalized_penalty = total_weight / max_possible if max_possible > 0 else 0
        
        return min(1.0, normalized_penalty), skill_weights
    
    def analyze_skill_set(self, skills: List[str]) -> Dict:
        """
        Analyze a set of skills for rarity distribution.
        
        Args:
            skills: List of skills to analyze
            
        Returns:
            Analysis dictionary
        """
        if not skills:
            return {"error": "No skills provided"}
        
        categories = {"very_rare": [], "rare": [], "uncommon": [], "common": [], "unknown": []}
        weights = {}
        
        for skill in skills:
            weight, category = self.get_rarity_weight(skill)
            weights[skill] = weight
            
            if "Very Rare" in category:
                categories["very_rare"].append(skill)
            elif "Rare" in category:
                categories["rare"].append(skill)
            elif "Uncommon" in category:
                categories["uncommon"].append(skill)
            elif "Unknown" in category:
                categories["unknown"].append(skill)
            else:
                categories["common"].append(skill)
        
        avg_rarity = np.mean(list(weights.values())) if weights else 0.5
        
        return {
            "skill_weights": weights,
            "categories": categories,
            "average_rarity": avg_rarity,
            "has_rare_skills": len(categories["very_rare"]) + len(categories["rare"]) > 0
        }
    
    def get_top_rare_skills(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get the top N rarest skills in the corpus."""
        sorted_skills = sorted(
            self.skill_idf.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_skills[:n]
    
    def get_top_common_skills(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get the top N most common skills in the corpus."""
        sorted_skills = sorted(
            self.skill_idf.items(),
            key=lambda x: x[1]
        )
        return sorted_skills[:n]


def compute_weighted_skill_penalty(
    resume_text: str,
    mandatory_skills: List[str],
    rarity_calculator: SkillRarityCalculator
) -> Tuple[float, Dict]:
    """
    Compute weighted skill penalty with rarity boosting.
    
    Args:
        resume_text: Candidate's resume text
        mandatory_skills: List of required skills
        rarity_calculator: Initialized SkillRarityCalculator
        
    Returns:
        Tuple of (weighted_penalty, details)
    """
    resume_lower = resume_text.lower()
    
    missing_skills = []
    present_skills = []
    
    for skill in mandatory_skills:
        skill_lower = skill.lower().strip()
        if skill_lower in resume_lower:
            present_skills.append(skill_lower)
        else:
            missing_skills.append(skill_lower)
    
    if not missing_skills:
        return 0.0, {
            "missing_skills": [],
            "present_skills": present_skills,
            "weighted_penalty": 0.0,
            "message": "All required skills present"
        }
    
    weighted_penalty, skill_weights = rarity_calculator.get_weighted_penalties(missing_skills)
    
    details = {
        "missing_skills": missing_skills,
        "present_skills": present_skills,
        "skill_weights": skill_weights,
        "weighted_penalty": weighted_penalty,
        "unweighted_penalty": len(missing_skills) / len(mandatory_skills)
    }
    
    return weighted_penalty, details


def format_rarity_report(analysis: Dict) -> str:
    """Generate a formatted skill rarity report."""
    report = "\n" + "=" * 50 + "\n"
    report += "💎 Skill Rarity Analysis\n"
    report += "=" * 50 + "\n\n"
    
    categories = analysis.get('categories', {})
    
    if categories.get('very_rare'):
        report += "🔴 Very Rare Skills:\n"
        for skill in categories['very_rare']:
            weight = analysis['skill_weights'].get(skill, 0)
            report += f"   • {skill} (weight: {weight:.2f})\n"
    
    if categories.get('rare'):
        report += "\n🟠 Rare Skills:\n"
        for skill in categories['rare']:
            weight = analysis['skill_weights'].get(skill, 0)
            report += f"   • {skill} (weight: {weight:.2f})\n"
    
    if categories.get('uncommon'):
        report += "\n🟡 Uncommon Skills:\n"
        for skill in categories['uncommon'][:5]:
            weight = analysis['skill_weights'].get(skill, 0)
            report += f"   • {skill} (weight: {weight:.2f})\n"
        if len(categories['uncommon']) > 5:
            report += f"   ... and {len(categories['uncommon']) - 5} more\n"
    
    if categories.get('common'):
        report += "\n🟢 Common Skills:\n"
        for skill in categories['common'][:5]:
            weight = analysis['skill_weights'].get(skill, 0)
            report += f"   • {skill} (weight: {weight:.2f})\n"
        if len(categories['common']) > 5:
            report += f"   ... and {len(categories['common']) - 5} more\n"
    
    report += f"\n📊 Average Rarity Score: {analysis.get('average_rarity', 0):.2f}\n"
    
    return report


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Skill Rarity Weighting (SRW) - Test")
    print("=" * 50)
    
    # Initialize calculator
    calculator = SkillRarityCalculator("data/job_skills.csv")
    
    # Test skill rarity
    test_skills = ["python", "machine learning", "quantum computing", "kubernetes", "excel"]
    print("\n📊 Individual Skill Rarity:")
    for skill in test_skills:
        weight, category = calculator.get_rarity_weight(skill)
        print(f"   {skill}: {category} (weight: {weight:.2f})")
    
    # Test weighted penalty
    missing = ["kubernetes", "terraform", "python"]
    penalty, details = calculator.get_weighted_penalties(missing)
    print(f"\n📉 Missing Skills Penalty:")
    print(f"   Skills: {missing}")
    print(f"   Weighted Penalty: {penalty:.2f}")
    print(f"   Weights: {details}")
    
    # Show top rare skills
    print("\n🔝 Top 10 Rarest Skills:")
    for skill, idf in calculator.get_top_rare_skills(10):
        print(f"   • {skill}: {idf:.2f}")
