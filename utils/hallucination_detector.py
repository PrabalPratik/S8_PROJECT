"""
Hallucination Risk Index (HRI) - Novel Feature F2

Estimates how likely a generated JD contains hallucinated requirements
by comparing T5 output tokens against skill taxonomy and training corpus.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import math


class HallucinationDetector:
    """
    Detects hallucinated skills/requirements in generated job descriptions.
    
    HRI = 0.0 → Low hallucination risk (all skills are well-known)
    HRI = 1.0 → High hallucination risk (many unknown/rare skills)
    """
    
    def __init__(self, skill_corpus_path: Optional[str] = None):
        """
        Initialize the Hallucination Detector.
        
        Args:
            skill_corpus_path: Path to CSV file containing job skills data
        """
        self.skill_taxonomy: Set[str] = set()
        self.skill_frequency: Dict[str, int] = defaultdict(int)
        self.skill_tfidf: Dict[str, float] = {}
        self.total_documents = 0
        
        if skill_corpus_path:
            self._build_taxonomy(skill_corpus_path)
    
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
    
    def _build_taxonomy(self, corpus_path: str) -> None:
        """Build skill taxonomy and TF-IDF scores from corpus."""
        print(f"Building skill taxonomy from {corpus_path}...")
        
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
        
        # Document frequency for IDF calculation
        doc_frequency: Dict[str, int] = defaultdict(int)
        
        for idx, row in df.iterrows():
            skills = self._parse_skill_list(row.get(skill_col, ''))
            if not skills:
                continue
            
            self.total_documents += 1
            seen_in_doc = set()
            
            for skill in skills:
                self.skill_taxonomy.add(skill)
                self.skill_frequency[skill] += 1
                
                # Document frequency (each skill counted once per document)
                if skill not in seen_in_doc:
                    doc_frequency[skill] += 1
                    seen_in_doc.add(skill)
        
        # Calculate TF-IDF scores
        for skill, freq in self.skill_frequency.items():
            tf = freq / sum(self.skill_frequency.values())
            idf = math.log(self.total_documents / (doc_frequency[skill] + 1))
            self.skill_tfidf[skill] = tf * idf
        
        print(f"Taxonomy built: {len(self.skill_taxonomy)} unique skills from {self.total_documents} documents")
    
    def _extract_potential_skills(self, text: str) -> List[str]:
        """
        Extract potential skill-like phrases from text.
        Uses heuristics to identify technical terms and skills.
        """
        text_lower = text.lower()
        
        # Common skill patterns
        patterns = [
            # Technical skills (programming languages, frameworks, etc.)
            r'\b(?:python|java|javascript|c\+\+|ruby|go|rust|scala|kotlin|swift)\b',
            r'\b(?:react|angular|vue|django|flask|spring|node\.?js|express)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|terraform|jenkins)\b',
            r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
            r'\b(?:machine learning|deep learning|nlp|computer vision|ai)\b',
            # Soft skills
            r'\b(?:communication|leadership|teamwork|problem.?solving|analytical)\b',
            # Domain skills - generic pattern for multi-word skills
            r'\b[a-z]+(?:\s+[a-z]+){0,2}\s+(?:experience|skills?|knowledge|expertise)\b',
        ]
        
        found = set()
        
        # First, check against known taxonomy
        for skill in self.skill_taxonomy:
            if skill in text_lower:
                found.add(skill)
        
        # Then, extract using patterns
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            found.update(matches)
        
        return list(found)
    
    def calculate_hri(
        self,
        generated_jd: str,
        required_skills: Optional[List[str]] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate Hallucination Risk Index for a generated JD.
        
        Args:
            generated_jd: The generated job description text
            required_skills: Optional list of skills that SHOULD be in the JD
            
        Returns:
            Tuple of (hri_score, details_dict)
        """
        if not generated_jd:
            return 1.0, {"error": "Empty JD provided"}
        
        # Extract skills mentioned in the generated JD
        extracted_skills = self._extract_potential_skills(generated_jd)
        
        if not extracted_skills:
            return 0.5, {
                "message": "No skills detected in JD",
                "extracted_skills": [],
                "known_skills": [],
                "unknown_skills": [],
                "rare_skills": []
            }
        
        # Categorize skills
        known_skills = []
        unknown_skills = []
        rare_skills = []
        
        for skill in extracted_skills:
            skill = skill.lower().strip()
            if skill in self.skill_taxonomy:
                known_skills.append(skill)
                # Check if it's rare (bottom 10%)
                if self.skill_frequency.get(skill, 0) < (self.total_documents * 0.01):
                    rare_skills.append(skill)
            else:
                unknown_skills.append(skill)
        
        # Calculate HRI components
        total_skills = len(extracted_skills)
        
        # Unknown skill ratio (highest weight - these are potential hallucinations)
        unknown_ratio = len(unknown_skills) / total_skills if total_skills > 0 else 0
        
        # Rare skill ratio (medium weight - unusual but not necessarily wrong)
        rare_ratio = len(rare_skills) / total_skills if total_skills > 0 else 0
        
        # Check required skill coverage (if provided)
        missing_required_penalty = 0.0
        missing_required = []
        if required_skills:
            required_set = set(s.lower().strip() for s in required_skills)
            jd_lower = generated_jd.lower()
            for req_skill in required_set:
                if req_skill not in jd_lower:
                    missing_required.append(req_skill)
            
            missing_required_penalty = len(missing_required) / len(required_set) if required_set else 0
        
        # Calculate final HRI
        hri = (
            unknown_ratio * 0.5 +           # Unknown skills are most concerning
            rare_ratio * 0.2 +              # Rare skills might be legitimate niche requirements
            missing_required_penalty * 0.3  # Missing required skills suggests generation issues
        )
        
        hri = min(1.0, max(0.0, hri))  # Clamp to [0, 1]
        
        # Determine risk level
        if hri < 0.2:
            risk_level = "🟢 Low"
        elif hri < 0.4:
            risk_level = "🟡 Moderate"
        elif hri < 0.6:
            risk_level = "🟠 Elevated"
        else:
            risk_level = "🔴 High"
        
        details = {
            "hri_score": hri,
            "risk_level": risk_level,
            "extracted_skills": extracted_skills,
            "known_skills": known_skills,
            "unknown_skills": unknown_skills,
            "rare_skills": rare_skills,
            "missing_required": missing_required,
            "unknown_ratio": unknown_ratio,
            "rare_ratio": rare_ratio,
            "missing_required_penalty": missing_required_penalty,
            "corrections": {}  # Will be populated if correction graph is set
        }
        
        # Generate corrections if graph is available
        if hasattr(self, 'correction_graph') and self.correction_graph is not None:
            corrections = self.suggest_corrections(unknown_skills)
            details["corrections"] = corrections
        
        return hri, details
    
    def set_correction_graph(self, ccg) -> None:
        """
        Link a ConstraintCoverageGraph for correction suggestions.
        
        Args:
            ccg: Initialized ConstraintCoverageGraph instance
        """
        self.correction_graph = ccg
    
    def suggest_corrections(self, unknown_skills: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Suggest valid skill replacements for unknown/hallucinated skills.
        
        Uses the ConstraintCoverageGraph to find semantically similar valid skills.
        
        Args:
            unknown_skills: List of skills not found in taxonomy
            
        Returns:
            Dict mapping unknown skill to list of (suggested_skill, similarity_score) tuples
        """
        if not hasattr(self, 'correction_graph') or self.correction_graph is None:
            return {}
        
        corrections = {}
        
        for unknown in unknown_skills:
            unknown_lower = unknown.lower().strip()
            suggestions = []
            
            # Strategy 1: Find similar skills by substring matching
            for known_skill in self.skill_taxonomy:
                # Check if unknown is a substring or vice versa
                if unknown_lower in known_skill or known_skill in unknown_lower:
                    suggestions.append((known_skill, 0.8))
                # Check for word overlap
                elif set(unknown_lower.split()) & set(known_skill.split()):
                    suggestions.append((known_skill, 0.6))
            
            # Strategy 2: Use graph neighbors if we have partial matches
            if not suggestions and self.correction_graph:
                # Try to find skills that commonly co-occur with the context
                for known_skill in list(self.skill_taxonomy)[:100]:  # Sample for efficiency
                    neighbors = self.correction_graph.get_neighbors(known_skill, top_k=5)
                    for neighbor, weight in neighbors:
                        if neighbor in self.skill_taxonomy:
                            suggestions.append((neighbor, weight / 100))  # Normalize
            
            # Deduplicate and sort by score
            seen = set()
            unique_suggestions = []
            for skill, score in sorted(suggestions, key=lambda x: -x[1]):
                if skill not in seen:
                    seen.add(skill)
                    unique_suggestions.append((skill, round(score, 2)))
                    if len(unique_suggestions) >= 3:
                        break
            
            if unique_suggestions:
                corrections[unknown] = unique_suggestions
        
        return corrections
    
    def apply_corrections(self, text: str, corrections: Dict[str, List[Tuple[str, float]]]) -> str:
        """
        Apply suggested corrections to text.
        
        Replaces unknown skills with the top suggestion.
        
        Args:
            text: Original text containing hallucinated skills
            corrections: Dict of corrections from suggest_corrections()
            
        Returns:
            Corrected text
        """
        corrected = text
        for unknown, suggestions in corrections.items():
            if suggestions:
                top_suggestion = suggestions[0][0]
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(unknown), re.IGNORECASE)
                corrected = pattern.sub(top_suggestion, corrected)
        
        return corrected
    
    def get_skill_confidence(self, skill: str) -> Tuple[float, str]:
        """
        Get confidence score for a single skill.
        
        Returns:
            Tuple of (confidence, explanation)
        """
        skill = skill.lower().strip()
        
        if skill not in self.skill_taxonomy:
            return 0.0, f"'{skill}' not found in skill taxonomy - potential hallucination"
        
        freq = self.skill_frequency.get(skill, 0)
        
        if freq < self.total_documents * 0.01:
            return 0.3, f"'{skill}' is rare (appears in <1% of JDs)"
        elif freq < self.total_documents * 0.05:
            return 0.6, f"'{skill}' is uncommon (appears in 1-5% of JDs)"
        else:
            return 1.0, f"'{skill}' is well-established (appears in >5% of JDs)"
    
    def format_hri_report(self, hri_score: float, details: Dict) -> str:
        """Generate a formatted HRI report."""
        report = f"\n{'='*50}\n"
        report += f"🔬 Hallucination Risk Index (HRI) Report\n"
        report += f"{'='*50}\n\n"
        
        report += f"📊 HRI Score: {hri_score:.2f} - {details.get('risk_level', 'Unknown')}\n\n"
        
        report += f"📋 Skills Detected: {len(details.get('extracted_skills', []))}\n"
        report += f"   ✅ Known: {len(details.get('known_skills', []))}\n"
        report += f"   ⚠️ Unknown: {len(details.get('unknown_skills', []))}\n"
        report += f"   📌 Rare: {len(details.get('rare_skills', []))}\n"
        
        if details.get('unknown_skills'):
            report += f"\n⚠️ Potentially Hallucinated Skills:\n"
            for skill in details['unknown_skills'][:5]:
                report += f"   • {skill}\n"
                # Show corrections if available
                if details.get('corrections', {}).get(skill):
                    suggestions = details['corrections'][skill]
                    report += f"     → Suggested: {', '.join([s[0] for s in suggestions])}\n"
        
        if details.get('missing_required'):
            report += f"\n❌ Missing Required Skills:\n"
            for skill in details['missing_required'][:5]:
                report += f"   • {skill}\n"
        
        return report


def calculate_hri(generated_jd: str, skill_taxonomy: Set[str], corpus_tfidf: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Simplified function interface for HRI calculation.
    
    Args:
        generated_jd: Generated job description text
        skill_taxonomy: Set of known skills
        corpus_tfidf: TF-IDF scores for skills
        
    Returns:
        Tuple of (hri_score, flagged_terms)
    """
    detector = HallucinationDetector()
    detector.skill_taxonomy = skill_taxonomy
    detector.skill_tfidf = corpus_tfidf
    detector.total_documents = 1000  # Default assumption
    
    hri, details = detector.calculate_hri(generated_jd)
    flagged_terms = details.get('unknown_skills', []) + details.get('rare_skills', [])
    
    return hri, flagged_terms


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Hallucination Risk Index (HRI) - Test")
    print("=" * 50)
    
    # Initialize detector
    detector = HallucinationDetector("data/job_skills.csv")
    
    # Test with sample JD
    sample_jd = """
    We are looking for a Senior Data Scientist with expertise in Python, 
    Machine Learning, and Deep Learning. Experience with TensorFlow and 
    PyTorch required. Knowledge of QuantumML and HyperDataSync frameworks 
    is a plus. Must have strong communication skills.
    """
    
    required_skills = ["Python", "Machine Learning", "SQL"]
    
    hri, details = detector.calculate_hri(sample_jd, required_skills)
    print(detector.format_hri_report(hri, details))
