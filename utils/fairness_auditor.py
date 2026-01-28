"""
Fairness Auditor - Novel Feature F6

Calculates demographic parity and exposure-aware fairness metrics
for the candidate ranking pipeline.

Metrics implemented:
- Demographic Parity: Selection rate per demographic group
- Disparate Impact Ratio (DIR): 80% rule compliance
- Exposure Parity: Position-weighted visibility across groups
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import random


class FairnessAuditor:
    """
    Audits ranking results for demographic fairness.
    
    Usage:
        auditor = FairnessAuditor()
        report = auditor.audit(rankings_df, demographics_df)
    """
    
    def __init__(self, dir_threshold: float = 0.8):
        """
        Initialize the Fairness Auditor.
        
        Args:
            dir_threshold: Disparate Impact Ratio threshold (default 0.8 = 80% rule)
        """
        self.dir_threshold = dir_threshold
    
    def calculate_selection_rates(
        self,
        rankings: pd.DataFrame,
        demographics: pd.DataFrame,
        protected_attr: str,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Calculate selection rate for each demographic group.
        
        Selection Rate = (# selected from group) / (# total in group)
        
        Args:
            rankings: DataFrame with 'Candidate' and 'Global Score' columns
            demographics: DataFrame with 'Candidate' and protected attribute columns
            protected_attr: Column name of protected attribute (e.g., 'Gender', 'Age_Group')
            top_k: Number of top candidates considered "selected"
            
        Returns:
            Dict mapping group name to selection rate
        """
        # Merge rankings with demographics
        merged = rankings.merge(demographics, on='Candidate', how='left')
        
        if protected_attr not in merged.columns:
            return {"error": f"Protected attribute '{protected_attr}' not found"}
        
        # Get top-k selected candidates
        top_candidates = merged.nlargest(top_k, 'Global Score')['Candidate'].tolist()
        
        # Calculate selection rate per group
        selection_rates = {}
        for group in merged[protected_attr].unique():
            group_members = merged[merged[protected_attr] == group]
            group_total = len(group_members)
            group_selected = len([c for c in top_candidates if c in group_members['Candidate'].values])
            
            selection_rates[str(group)] = group_selected / group_total if group_total > 0 else 0.0
        
        return selection_rates
    
    def calculate_disparate_impact_ratio(
        self,
        selection_rates: Dict[str, float],
        reference_group: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate Disparate Impact Ratio (DIR).
        
        DIR = min_group_rate / max_group_rate
        
        A DIR < 0.8 indicates potential adverse impact (80% rule).
        
        Args:
            selection_rates: Dict of group selection rates
            reference_group: Optional reference group (default: highest rate group)
            
        Returns:
            Tuple of (DIR score, details dict)
        """
        if not selection_rates or "error" in selection_rates:
            return 0.0, {"error": "Invalid selection rates"}
        
        rates = list(selection_rates.values())
        max_rate = max(rates) if rates else 0
        min_rate = min(rates) if rates else 0
        
        if max_rate == 0:
            dir_score = 1.0  # No selections made
        else:
            dir_score = min_rate / max_rate
        
        # Identify advantaged and disadvantaged groups
        max_group = max(selection_rates, key=selection_rates.get)
        min_group = min(selection_rates, key=selection_rates.get)
        
        # Determine compliance
        is_compliant = dir_score >= self.dir_threshold
        
        details = {
            "dir_score": round(dir_score, 3),
            "threshold": self.dir_threshold,
            "is_compliant": is_compliant,
            "status": "✅ Compliant" if is_compliant else "⚠️ Adverse Impact Detected",
            "advantaged_group": max_group,
            "disadvantaged_group": min_group,
            "selection_rates": selection_rates
        }
        
        return dir_score, details
    
    def calculate_exposure_parity(
        self,
        rankings: pd.DataFrame,
        demographics: pd.DataFrame,
        protected_attr: str
    ) -> Tuple[float, Dict]:
        """
        Calculate exposure-aware fairness using position-weighted visibility.
        
        Exposure = sum(1 / log2(rank + 1)) for each group member
        
        This metric accounts for the fact that higher-ranked candidates
        get exponentially more visibility than lower-ranked ones.
        
        Args:
            rankings: DataFrame with 'Candidate' and 'Global Score' columns
            demographics: DataFrame with 'Candidate' and protected attribute columns
            protected_attr: Column name of protected attribute
            
        Returns:
            Tuple of (exposure disparity score, details dict)
        """
        # Merge and rank
        merged = rankings.merge(demographics, on='Candidate', how='left')
        merged = merged.sort_values('Global Score', ascending=False).reset_index(drop=True)
        merged['Rank'] = merged.index + 1
        
        if protected_attr not in merged.columns:
            return 0.0, {"error": f"Protected attribute '{protected_attr}' not found"}
        
        # Calculate exposure per group
        group_exposure = defaultdict(float)
        group_count = defaultdict(int)
        
        for idx, row in merged.iterrows():
            group = str(row[protected_attr])
            rank = row['Rank']
            exposure = 1.0 / np.log2(rank + 1)  # DCG-style discounting
            group_exposure[group] += exposure
            group_count[group] += 1
        
        # Normalize by group size
        normalized_exposure = {
            g: group_exposure[g] / group_count[g] if group_count[g] > 0 else 0
            for g in group_exposure
        }
        
        # Calculate disparity (max - min) / max
        exposures = list(normalized_exposure.values())
        if not exposures or max(exposures) == 0:
            disparity = 0.0
        else:
            disparity = (max(exposures) - min(exposures)) / max(exposures)
        
        details = {
            "exposure_disparity": round(disparity, 3),
            "normalized_exposure": {k: round(v, 3) for k, v in normalized_exposure.items()},
            "raw_exposure": {k: round(v, 3) for k, v in group_exposure.items()},
            "group_counts": dict(group_count),
            "is_fair": disparity < 0.2  # 20% disparity threshold
        }
        
        return disparity, details
    
    def audit(
        self,
        rankings: pd.DataFrame,
        demographics: pd.DataFrame,
        protected_attrs: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run full fairness audit across multiple protected attributes.
        
        Args:
            rankings: DataFrame with 'Candidate' and 'Global Score' columns
            demographics: DataFrame with candidate demographics
            protected_attrs: List of protected attribute columns to audit
            top_k: Number of top candidates for selection rate calculation
            
        Returns:
            Comprehensive audit report
        """
        if protected_attrs is None:
            # Auto-detect demographic columns
            protected_attrs = [col for col in demographics.columns 
                              if col not in ['Candidate', 'Name', 'ID']]
        
        report = {
            "summary": {},
            "by_attribute": {},
            "overall_compliance": True,
            "recommendations": []
        }
        
        for attr in protected_attrs:
            # Selection rates
            selection_rates = self.calculate_selection_rates(
                rankings, demographics, attr, top_k
            )
            
            # DIR
            dir_score, dir_details = self.calculate_disparate_impact_ratio(selection_rates)
            
            # Exposure
            exp_disparity, exp_details = self.calculate_exposure_parity(
                rankings, demographics, attr
            )
            
            report["by_attribute"][attr] = {
                "selection_rates": selection_rates,
                "dir": dir_details,
                "exposure": exp_details
            }
            
            # Update overall compliance
            if not dir_details.get("is_compliant", True):
                report["overall_compliance"] = False
                report["recommendations"].append(
                    f"Review ranking criteria for '{attr}' - DIR below 80%"
                )
            
            if not exp_details.get("is_fair", True):
                report["recommendations"].append(
                    f"Consider exposure re-ranking for '{attr}' - high visibility disparity"
                )
        
        # Summary
        report["summary"] = {
            "attributes_audited": len(protected_attrs),
            "compliant_count": sum(
                1 for attr in report["by_attribute"].values() 
                if attr["dir"].get("is_compliant", False)
            ),
            "overall_status": "✅ Fair" if report["overall_compliance"] else "⚠️ Review Needed"
        }
        
        return report
    
    def format_report(self, report: Dict) -> str:
        """Generate human-readable fairness report."""
        output = "\n" + "=" * 50 + "\n"
        output += "⚖️ FAIRNESS AUDIT REPORT\n"
        output += "=" * 50 + "\n\n"
        
        output += f"📊 Status: {report['summary']['overall_status']}\n"
        output += f"📋 Attributes Audited: {report['summary']['attributes_audited']}\n"
        output += f"✓ Compliant: {report['summary']['compliant_count']}\n\n"
        
        for attr, details in report["by_attribute"].items():
            output += f"--- {attr.upper()} ---\n"
            dir_info = details["dir"]
            output += f"  DIR Score: {dir_info['dir_score']:.1%} "
            output += f"({dir_info['status']})\n"
            
            output += "  Selection Rates:\n"
            for group, rate in details["selection_rates"].items():
                output += f"    • {group}: {rate:.1%}\n"
            
            exp_info = details["exposure"]
            output += f"  Exposure Disparity: {exp_info['exposure_disparity']:.1%}\n\n"
        
        if report["recommendations"]:
            output += "📌 RECOMMENDATIONS:\n"
            for rec in report["recommendations"]:
                output += f"  ⚠️ {rec}\n"
        
        return output


def generate_synthetic_demographics(
    candidates: List[str],
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic demographic data for testing.
    
    Creates realistic but random demographic attributes for candidates.
    
    Args:
        candidates: List of candidate names/IDs
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with Candidate and demographic columns
    """
    random.seed(seed)
    np.random.seed(seed)
    
    demographics = []
    for candidate in candidates:
        demographics.append({
            "Candidate": candidate,
            "Gender": random.choice(["Male", "Female", "Non-Binary"]),
            "Age_Group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
            "Ethnicity": random.choice(["Group_A", "Group_B", "Group_C", "Group_D"]),
            "Disability": random.choice(["No", "Yes"]),
            "Veteran": random.choice(["No", "Yes"])
        })
    
    return pd.DataFrame(demographics)


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Fairness Auditor - Test")
    print("=" * 50)
    
    # Create sample rankings
    rankings = pd.DataFrame({
        "Candidate": ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"],
        "Global Score": [0.85, 0.82, 0.78, 0.75, 0.72, 0.68, 0.65, 0.60]
    })
    
    # Generate demographics
    demographics = generate_synthetic_demographics(rankings["Candidate"].tolist())
    
    # Run audit
    auditor = FairnessAuditor()
    report = auditor.audit(rankings, demographics, ["Gender", "Age_Group"], top_k=3)
    
    print(auditor.format_report(report))
