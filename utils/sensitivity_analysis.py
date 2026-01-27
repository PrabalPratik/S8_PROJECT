"""
Candidate Sensitivity Simulation - Novel Feature F3

Monte-Carlo perturbation analysis on α, β, γ weights to show rank stability.
Enables defensible AI decisions in HR by quantifying ranking confidence.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class CandidateResult:
    """Stores candidate information and their score."""
    name: str
    resume_text: str
    mandatory_skills: str
    experience_gap: float
    base_score: float
    rank: int


class SensitivityAnalyzer:
    """
    Performs Monte-Carlo sensitivity analysis on ranking parameters.
    
    Quantifies how stable each candidate's rank is under parameter perturbations.
    """
    
    def __init__(
        self,
        base_alpha: float = 0.7,
        base_beta: float = 0.2,
        base_gamma: float = 0.1,
        perturbation_std: float = 0.15
    ):
        """
        Initialize the Sensitivity Analyzer.
        
        Args:
            base_alpha: Base weight for semantic similarity
            base_beta: Base weight for skill penalty
            base_gamma: Base weight for experience gap penalty
            perturbation_std: Standard deviation for parameter perturbations
        """
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_gamma = base_gamma
        self.perturbation_std = perturbation_std
    
    def _generate_perturbed_weights(self) -> Tuple[float, float, float]:
        """Generate perturbed α, β, γ weights."""
        alpha = self.base_alpha + random.gauss(0, self.perturbation_std)
        beta = self.base_beta + random.gauss(0, self.perturbation_std)
        gamma = self.base_gamma + random.gauss(0, self.perturbation_std)
        
        # Ensure non-negative weights
        alpha = max(0.1, min(1.0, alpha))
        beta = max(0.0, min(0.5, beta))
        gamma = max(0.0, min(0.5, gamma))
        
        # Normalize to sum to 1 (optional, can be toggled)
        total = alpha + beta + gamma
        if total > 0:
            alpha /= total
            beta /= total
            gamma /= total
        
        return alpha, beta, gamma
    
    def monte_carlo_rank_stability(
        self,
        ranker,  # BertRanker instance
        candidates: List[Dict],
        jd_text: str,
        mandatory_skills: str,
        n_simulations: int = 100
    ) -> Dict[str, Dict]:
        """
        Perform Monte-Carlo simulation to assess rank stability.
        
        Args:
            ranker: BertRanker instance with get_research_score method
            candidates: List of candidate dicts with 'name', 'resume_text', 'experience_gap'
            jd_text: Job description text
            mandatory_skills: Required skills string
            n_simulations: Number of Monte-Carlo iterations
            
        Returns:
            Dict mapping candidate names to stability metrics
        """
        if not candidates:
            return {}
        
        # Track rank history for each candidate
        rank_history: Dict[str, List[int]] = {c['name']: [] for c in candidates}
        score_history: Dict[str, List[float]] = {c['name']: [] for c in candidates}
        
        for sim in range(n_simulations):
            # Generate perturbed weights
            alpha, beta, gamma = self._generate_perturbed_weights()
            
            # Score all candidates with perturbed weights
            sim_scores = []
            for candidate in candidates:
                score, _ = ranker.get_research_score(
                    candidate['resume_text'],
                    jd_text,
                    mandatory_skills=mandatory_skills,
                    experience_gap=candidate.get('experience_gap', 0),
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )
                sim_scores.append((candidate['name'], score))
                score_history[candidate['name']].append(score)
            
            # Sort and assign ranks
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (name, _) in enumerate(sim_scores, 1):
                rank_history[name].append(rank)
        
        # Calculate stability metrics
        results = {}
        for candidate in candidates:
            name = candidate['name']
            ranks = rank_history[name]
            scores = score_history[name]
            
            # Base rank (from original weights)
            base_score, _ = ranker.get_research_score(
                candidate['resume_text'],
                jd_text,
                mandatory_skills=mandatory_skills,
                experience_gap=candidate.get('experience_gap', 0),
                alpha=self.base_alpha,
                beta=self.base_beta,
                gamma=self.base_gamma
            )
            
            # Calculate stability as percentage of simulations where rank stayed same
            base_rank = self._get_base_rank(candidates, ranker, jd_text, mandatory_skills, name)
            rank_matches = sum(1 for r in ranks if r == base_rank)
            rank_stability = (rank_matches / n_simulations) * 100
            
            # Calculate score variance
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            score_cv = (score_std / score_mean * 100) if score_mean > 0 else 0
            
            # Determine stability category
            if rank_stability >= 80:
                stability_level = "🟢 Highly Stable"
            elif rank_stability >= 50:
                stability_level = "🟡 Moderately Stable"
            else:
                stability_level = "🔴 Unstable"
            
            results[name] = {
                "rank_stability_pct": round(rank_stability, 1),
                "stability_level": stability_level,
                "base_rank": base_rank,
                "base_score": round(base_score, 4),
                "rank_range": (min(ranks), max(ranks)),
                "score_mean": round(score_mean, 4),
                "score_std": round(score_std, 4),
                "score_cv_pct": round(score_cv, 1)
            }
        
        return results
    
    def _get_base_rank(
        self,
        candidates: List[Dict],
        ranker,
        jd_text: str,
        mandatory_skills: str,
        target_name: str
    ) -> int:
        """Get the base rank for a candidate using original weights."""
        base_scores = []
        for candidate in candidates:
            score, _ = ranker.get_research_score(
                candidate['resume_text'],
                jd_text,
                mandatory_skills=mandatory_skills,
                experience_gap=candidate.get('experience_gap', 0),
                alpha=self.base_alpha,
                beta=self.base_beta,
                gamma=self.base_gamma
            )
            base_scores.append((candidate['name'], score))
        
        base_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, _) in enumerate(base_scores, 1):
            if name == target_name:
                return rank
        return len(candidates)
    
    def quick_stability_check(
        self,
        ranker,
        candidates: List[Dict],
        jd_text: str,
        mandatory_skills: str,
        n_quick: int = 30
    ) -> Dict[str, float]:
        """
        Quick stability check with fewer simulations.
        Returns just the stability percentage for each candidate.
        """
        results = self.monte_carlo_rank_stability(
            ranker, candidates, jd_text, mandatory_skills, n_simulations=n_quick
        )
        return {name: data['rank_stability_pct'] for name, data in results.items()}


def format_stability_report(results: Dict[str, Dict]) -> str:
    """Generate a formatted stability analysis report."""
    report = "\n" + "=" * 60 + "\n"
    report += "📊 Candidate Rank Stability Analysis (Monte-Carlo)\n"
    report += "=" * 60 + "\n\n"
    
    # Sort by stability (descending)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['rank_stability_pct'],
        reverse=True
    )
    
    for name, data in sorted_results:
        stability = data['rank_stability_pct']
        level = data['stability_level']
        base_rank = data['base_rank']
        rank_range = data['rank_range']
        
        # Create stability bar
        bar_length = int(stability / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        report += f"👤 {name}\n"
        report += f"   Rank: #{base_rank} | Stability: {bar} {stability:.0f}%\n"
        report += f"   {level}\n"
        report += f"   Rank Range: #{rank_range[0]} - #{rank_range[1]}\n"
        report += f"   Score: {data['score_mean']:.3f} ± {data['score_std']:.3f}\n\n"
    
    # Add interpretation
    report += "─" * 60 + "\n"
    report += "📌 Interpretation:\n"
    report += "   • High stability (>80%): Rank is robust to parameter changes\n"
    report += "   • Low stability (<50%): Rank may change significantly with\n"
    report += "     different weight configurations - review manually\n"
    
    return report


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Candidate Sensitivity Simulation - Test")
    print("=" * 60)
    
    # Mock ranker for testing
    class MockRanker:
        def get_research_score(self, resume_text, jd_text, mandatory_skills=None, 
                               experience_gap=0, alpha=0.7, beta=0.2, gamma=0.1):
            # Simple mock scoring
            base = 0.5 + hash(resume_text) % 1000 / 2000
            penalty = beta * 0.1 + gamma * experience_gap / 10
            score = alpha * base - penalty
            return max(0, min(1, score)), {}
    
    # Test candidates
    candidates = [
        {"name": "Alice", "resume_text": "Python SQL ML expert", "experience_gap": 0},
        {"name": "Bob", "resume_text": "Java Spring developer", "experience_gap": 2},
        {"name": "Carol", "resume_text": "Data analyst Excel", "experience_gap": 1},
    ]
    
    analyzer = SensitivityAnalyzer()
    ranker = MockRanker()
    
    results = analyzer.monte_carlo_rank_stability(
        ranker=ranker,
        candidates=candidates,
        jd_text="Data Scientist with Python and ML skills",
        mandatory_skills="Python, SQL, ML",
        n_simulations=50
    )
    
    print(format_stability_report(results))
