"""
Constraint Coverage Graph (CCG) - Novel Feature F1

Converts skill lists into a graph structure and scores structural coverage
instead of flat token matching. Enables semantic relationships between skills.
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional


class ConstraintCoverageGraph:
    """
    Graph-based skill coverage scoring system.
    
    Nodes = Skills
    Edges = Co-occurrence in JD corpus
    Score = Coverage + Closeness to role-specific clusters
    """
    
    def __init__(self, skill_corpus_path: Optional[str] = None):
        """
        Initialize the Constraint Coverage Graph.
        
        Args:
            skill_corpus_path: Path to CSV file containing job skills data
        """
        self.graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.skill_frequency: Dict[str, int] = defaultdict(int)
        self.role_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.total_jobs = 0
        
        if skill_corpus_path:
            self._build_skill_graph(skill_corpus_path)
    
    def _parse_skill_list(self, skill_string: str) -> List[str]:
        """Parse skill string into list of normalized skills."""
        if pd.isna(skill_string) or not skill_string:
            return []
        
        # Handle Python list representation in CSV
        skill_string = str(skill_string)
        if skill_string.startswith('[') and skill_string.endswith(']'):
            # Remove brackets and quotes
            skill_string = skill_string[1:-1]
            skills = re.split(r"',\s*'|',\s*\"|\"', '|\", \"|', \"|\", '", skill_string)
            skills = [s.strip().strip("'\"").lower() for s in skills]
        else:
            # Simple comma-separated
            skills = [s.strip().lower() for s in skill_string.split(',')]
        
        return [s for s in skills if s and len(s) > 1]
    
    def _build_skill_graph(self, corpus_path: str) -> None:
        """
        Build skill co-occurrence graph from JD corpus.
        
        Creates edges between skills that appear together in job descriptions.
        Edge weight = number of co-occurrences.
        """
        print(f"Building skill graph from {corpus_path}...")
        
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
        
        # Find category/role column for clustering
        role_col = None
        for col in df.columns:
            if 'category' in col.lower() or 'role' in col.lower() or 'title' in col.lower():
                role_col = col
                break
        
        # Build graph from co-occurrences
        for idx, row in df.iterrows():
            skills = self._parse_skill_list(row.get(skill_col, ''))
            if not skills:
                continue
            
            self.total_jobs += 1
            
            # Update skill frequencies
            for skill in skills:
                self.skill_frequency[skill] += 1
            
            # Create edges between co-occurring skills
            for i, skill1 in enumerate(skills):
                for skill2 in skills[i+1:]:
                    self.graph[skill1][skill2] += 1
                    self.graph[skill2][skill1] += 1
            
            # Build role clusters
            if role_col and not pd.isna(row.get(role_col)):
                role = str(row[role_col]).lower().strip()
                self.role_clusters[role].update(skills)
        
        print(f"Graph built: {len(self.skill_frequency)} skills, {self.total_jobs} jobs")
    
    def get_neighbors(self, skill: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k co-occurring skills for a given skill."""
        skill = skill.lower().strip()
        if skill not in self.graph:
            return []
        
        neighbors = sorted(
            self.graph[skill].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return neighbors
    
    def get_skill_centrality(self, skill: str) -> float:
        """
        Calculate centrality score for a skill (0-1).
        Higher = more connected/important skill.
        """
        skill = skill.lower().strip()
        if skill not in self.graph:
            return 0.0
        
        # Degree centrality: number of connections normalized
        degree = len(self.graph[skill])
        max_degree = max(len(v) for v in self.graph.values()) if self.graph else 1
        
        return degree / max_degree if max_degree > 0 else 0.0
    
    def get_coverage_score(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        role: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate structural coverage score.
        
        Args:
            candidate_skills: Skills from resume/candidate
            required_skills: Required skills from JD
            role: Optional role for cluster bonus
            
        Returns:
            Tuple of (score, details_dict)
        """
        # Normalize inputs
        candidate_skills = [s.lower().strip() for s in candidate_skills if s]
        required_skills = [s.lower().strip() for s in required_skills if s]
        candidate_set = set(candidate_skills)
        required_set = set(required_skills)
        
        if not required_skills:
            return 1.0, {"message": "No required skills specified"}
        
        # 1. Direct Coverage (traditional matching)
        direct_matches = candidate_set.intersection(required_set)
        direct_coverage = len(direct_matches) / len(required_set)
        
        # 2. Indirect Coverage (graph-based)
        # Check if candidate has skills closely related to missing requirements
        missing_skills = required_set - candidate_set
        indirect_coverage = 0.0
        related_skills_found = {}
        
        for missing in missing_skills:
            if missing in self.graph:
                # Find if candidate has any related skills
                for cand_skill in candidate_set:
                    if cand_skill in self.graph[missing]:
                        # Weight by co-occurrence strength
                        strength = self.graph[missing][cand_skill]
                        max_strength = max(self.graph[missing].values()) if self.graph[missing] else 1
                        relatedness = strength / max_strength
                        
                        if relatedness > 0.3:  # Threshold for meaningful relationship
                            related_skills_found[missing] = (cand_skill, relatedness)
                            indirect_coverage += relatedness / len(missing_skills)
                            break
        
        # 3. Role Cluster Bonus
        cluster_bonus = 0.0
        if role:
            role = role.lower().strip()
            if role in self.role_clusters:
                cluster_skills = self.role_clusters[role]
                cluster_overlap = len(candidate_set.intersection(cluster_skills))
                cluster_bonus = min(0.1, cluster_overlap / max(len(cluster_skills), 1) * 0.1)
        
        # 4. Skill Centrality Bonus
        # Reward candidates who have high-centrality (important) skills
        centrality_bonus = 0.0
        for skill in direct_matches:
            centrality_bonus += self.get_skill_centrality(skill)
        centrality_bonus = min(0.1, centrality_bonus / max(len(required_set), 1) * 0.1)
        
        # Final score
        final_score = min(1.0, (
            direct_coverage * 0.6 +      # Primary weight on direct matches
            indirect_coverage * 0.2 +     # Bonus for related skills
            cluster_bonus +               # Role cluster alignment
            centrality_bonus              # Important skills bonus
        ))
        
        details = {
            "direct_coverage": direct_coverage,
            "direct_matches": list(direct_matches),
            "missing_skills": list(missing_skills),
            "indirect_coverage": indirect_coverage,
            "related_skills_found": related_skills_found,
            "cluster_bonus": cluster_bonus,
            "centrality_bonus": centrality_bonus,
            "final_score": final_score
        }
        
        return final_score, details
    
    def visualize_skill_neighborhood(self, skill: str, top_k: int = 8) -> str:
        """
        Generate a text-based visualization of skill neighborhood.
        
        Returns a formatted string showing the skill's connections.
        """
        skill = skill.lower().strip()
        neighbors = self.get_neighbors(skill, top_k)
        
        if not neighbors:
            return f"No connections found for '{skill}'"
        
        result = f"📊 Skill Graph: {skill}\n"
        result += "═" * 40 + "\n"
        
        max_weight = max(w for _, w in neighbors)
        for neighbor, weight in neighbors:
            bar_length = int((weight / max_weight) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            result += f"  ├─ {neighbor:20} {bar} ({int(weight)})\n"
        
        return result


def extract_skills_from_text(text: str, skill_graph: ConstraintCoverageGraph) -> List[str]:
    """
    Extract skills from text by matching against known skills in the graph.
    
    Args:
        text: Text to extract skills from (resume or JD)
        skill_graph: Initialized ConstraintCoverageGraph
        
    Returns:
        List of detected skills
    """
    text_lower = text.lower()
    found_skills = []
    
    for skill in skill_graph.skill_frequency.keys():
        # Check for whole word match
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    return found_skills


# Example usage and testing
if __name__ == "__main__":
    print("=" * 50)
    print("Constraint Coverage Graph (CCG) - Test")
    print("=" * 50)
    
    # Initialize with corpus
    ccg = ConstraintCoverageGraph("data/job_skills.csv")
    
    # Test skill neighborhood
    test_skill = "python"
    print(f"\n{ccg.visualize_skill_neighborhood(test_skill)}")
    
    # Test coverage scoring
    candidate_skills = ["python", "sql", "machine learning", "pandas"]
    required_skills = ["python", "sql", "tensorflow", "deep learning"]
    
    score, details = ccg.get_coverage_score(
        candidate_skills,
        required_skills,
        role="data scientist"
    )
    
    print(f"\n📈 Coverage Analysis:")
    print(f"   Candidate: {candidate_skills}")
    print(f"   Required: {required_skills}")
    print(f"   Final Score: {score:.2%}")
    print(f"   Direct Coverage: {details['direct_coverage']:.2%}")
    print(f"   Indirect Coverage: {details['indirect_coverage']:.2%}")
    print(f"   Related Skills Found: {details['related_skills_found']}")
