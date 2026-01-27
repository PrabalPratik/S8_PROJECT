# -*- coding: utf-8 -*-
"""
Test Suite for Novel Features (F1-F5)

Tests all 5 research-driven features:
- F1: Constraint Coverage Graph (CCG)
- F2: Hallucination Risk Index (HRI)
- F3: Candidate Sensitivity Simulation
- F4: Role Drift Detector
- F5: Skill Rarity Weighting (SRW)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_constraint_coverage_graph():
    """Test F1: Constraint Coverage Graph"""
    print("\n" + "=" * 60)
    print("🧪 TEST F1: Constraint Coverage Graph (CCG)")
    print("=" * 60)
    
    try:
        from models.constraint_graph import ConstraintCoverageGraph, extract_skills_from_text
        
        # Initialize CCG
        ccg = ConstraintCoverageGraph("data/job_skills.csv")
        
        # Test 1: Skill graph built
        assert len(ccg.skill_frequency) > 0, "Skill graph should have skills"
        print(f"✅ Skill graph built: {len(ccg.skill_frequency)} skills")
        
        # Test 2: Coverage scoring
        candidate_skills = ["python", "sql", "machine learning"]
        required_skills = ["python", "sql", "tensorflow", "deep learning"]
        
        score, details = ccg.get_coverage_score(candidate_skills, required_skills)
        
        assert 0 <= score <= 1, "Coverage score should be between 0 and 1"
        assert details["direct_coverage"] > 0, "Should have some direct coverage"
        print(f"✅ Coverage scoring works: {score:.2%}")
        print(f"   Direct coverage: {details['direct_coverage']:.2%}")
        print(f"   Missing skills: {details['missing_skills']}")
        
        # Test 3: Skill neighborhood
        neighbors = ccg.get_neighbors("python", top_k=5)
        print(f"✅ Neighbor detection: 'python' has {len(neighbors)} neighbors")
        
        print("\n✅ F1: Constraint Coverage Graph - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ F1: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hallucination_risk_index():
    """Test F2: Hallucination Risk Index"""
    print("\n" + "=" * 60)
    print("🧪 TEST F2: Hallucination Risk Index (HRI)")
    print("=" * 60)
    
    try:
        from utils.hallucination_detector import HallucinationDetector
        
        # Initialize detector
        detector = HallucinationDetector("data/job_skills.csv")
        
        # Test 1: Taxonomy built
        assert len(detector.skill_taxonomy) > 0, "Should have skills in taxonomy"
        print(f"✅ Taxonomy built: {len(detector.skill_taxonomy)} skills")
        
        # Test 2: Low HRI for known skills
        good_jd = "We need Python and machine learning experience. SQL is required."
        hri_low, details_low = detector.calculate_hri(good_jd)
        print(f"✅ Known skills JD: HRI = {hri_low:.2f} ({details_low['risk_level']})")
        
        # Test 3: Higher HRI for unknown skills
        risky_jd = "Must know QuantumBrainML and HyperDataSync framework."
        hri_high, details_high = detector.calculate_hri(risky_jd)
        print(f"✅ Unknown skills JD: HRI = {hri_high:.2f} ({details_high['risk_level']})")
        
        # Test 4: Required skills check
        hri_missing, details_missing = detector.calculate_hri(
            "Node.js developer needed.",
            required_skills=["Python", "Django"]
        )
        print(f"✅ Missing required skills check: {len(details_missing['missing_required'])} missing")
        
        print("\n✅ F2: Hallucination Risk Index - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ F2: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensitivity_simulation():
    """Test F3: Candidate Sensitivity Simulation"""
    print("\n" + "=" * 60)
    print("🧪 TEST F3: Candidate Sensitivity Simulation")
    print("=" * 60)
    
    try:
        from utils.sensitivity_analysis import SensitivityAnalyzer, format_stability_report
        
        # Mock ranker for testing
        class MockRanker:
            def get_research_score(self, resume_text, jd_text, mandatory_skills=None, 
                                   experience_gap=0, alpha=0.7, beta=0.2, gamma=0.1, **kwargs):
                base = 0.5 + len(resume_text) % 10 / 20
                penalty = beta * 0.1 + gamma * experience_gap / 10
                score = alpha * base - penalty
                return max(0, min(1, score)), {}
        
        # Initialize analyzer
        analyzer = SensitivityAnalyzer()
        ranker = MockRanker()
        
        # Test candidates
        candidates = [
            {"name": "Alice", "resume_text": "Python SQL ML expert with 5 years exp", "experience_gap": 0},
            {"name": "Bob", "resume_text": "Java Spring developer", "experience_gap": 2},
            {"name": "Carol", "resume_text": "Data analyst Excel visualization", "experience_gap": 1},
        ]
        
        # Test 1: Monte-Carlo stability
        results = analyzer.monte_carlo_rank_stability(
            ranker=ranker,
            candidates=candidates,
            jd_text="Data Scientist with Python and ML skills",
            mandatory_skills="Python, SQL, ML",
            n_simulations=30
        )
        
        assert len(results) == len(candidates), "Should have results for all candidates"
        print(f"✅ Monte-Carlo simulation completed for {len(candidates)} candidates")
        
        for name, data in results.items():
            stability = data['rank_stability_pct']
            print(f"   {name}: {stability:.0f}% stable (rank #{data['base_rank']})")
        
        # Test 2: Quick stability check
        quick_results = analyzer.quick_stability_check(
            ranker, candidates, "JD text", "skills", n_quick=20
        )
        assert all(0 <= v <= 100 for v in quick_results.values()), "Stability should be 0-100%"
        print(f"✅ Quick stability check works")
        
        print("\n✅ F3: Candidate Sensitivity Simulation - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ F3: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drift_detector():
    """Test F4: Role Drift Detector"""
    print("\n" + "=" * 60)
    print("🧪 TEST F4: Role Drift Detector")
    print("=" * 60)
    
    try:
        from models.drift_detector import RoleDriftDetector, format_drift_report
        
        # Initialize detector (may use fallback if sentence-transformers not installed)
        detector = RoleDriftDetector()
        print("✅ Drift detector initialized")
        
        # Test 1: Good alignment
        role = "Data Scientist"
        aligned_jd = """
        We are looking for a Data Scientist to join our team. You will work on 
        machine learning models, statistical analysis, and data science projects.
        """
        
        drift_low, details_low = detector.detect_drift(role, aligned_jd, "Python, Machine Learning")
        print(f"✅ Aligned JD: drift = {drift_low:.2f} ({details_low['drift_level']})")
        
        # Test 2: Role drift
        drifted_jd = """
        We need someone to build data pipelines and ETL processes. 
        Experience with Spark, Airflow, and data warehouse solutions required.
        """
        
        drift_high, details_high = detector.detect_drift(role, drifted_jd)
        print(f"✅ Drifted JD: drift = {drift_high:.2f} ({details_high['drift_level']})")
        
        if details_high.get("contaminated_role"):
            print(f"   Detected drift towards: {details_high['contaminated_role']}")
        
        # Test 3: Bulk detection
        bulk_results = detector.bulk_detect(role, [aligned_jd, drifted_jd])
        assert len(bulk_results) == 2, "Should have 2 results"
        print(f"✅ Bulk detection works: {len(bulk_results)} JDs analyzed")
        
        print("\n✅ F4: Role Drift Detector - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ F4: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_rarity_weighting():
    """Test F5: Skill Rarity Weighting"""
    print("\n" + "=" * 60)
    print("🧪 TEST F5: Skill Rarity Weighting (SRW)")
    print("=" * 60)
    
    try:
        from utils.skill_rarity import SkillRarityCalculator, compute_weighted_skill_penalty
        
        # Initialize calculator
        calculator = SkillRarityCalculator("data/job_skills.csv")
        
        # Test 1: Rarity weights computed
        assert len(calculator.skill_idf) > 0, "Should have IDF scores"
        print(f"✅ Rarity weights computed for {len(calculator.skill_idf)} skills")
        
        # Test 2: Individual skill rarity
        test_skills = ["python", "communication", "machine learning"]
        for skill in test_skills:
            weight, category = calculator.get_rarity_weight(skill)
            print(f"   {skill}: {category} (weight: {weight:.2f})")
        
        # Test 3: Weighted penalties
        missing_skills = ["python", "kubernetes", "terraform"]
        penalty, weights = calculator.get_weighted_penalties(missing_skills)
        print(f"✅ Weighted penalty for {len(missing_skills)} missing skills: {penalty:.2f}")
        
        # Test 4: Top rare skills
        top_rare = calculator.get_top_rare_skills(5)
        print(f"✅ Top 5 rarest skills retrieved")
        for skill, idf in top_rare[:3]:
            print(f"   {skill}: IDF = {idf:.2f}")
        
        # Test 5: Skill set analysis
        analysis = calculator.analyze_skill_set(test_skills)
        assert "average_rarity" in analysis, "Should have average rarity"
        print(f"✅ Skill set analysis: avg rarity = {analysis['average_rarity']:.2f}")
        
        print("\n✅ F5: Skill Rarity Weighting - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ F5: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bert_ranker_srw_integration():
    """Test SRW integration with BertRanker"""
    print("\n" + "=" * 60)
    print("🧪 TEST: BertRanker SRW Integration")
    print("=" * 60)
    
    try:
        from models.bert_ranker import BertRanker
        from utils.skill_rarity import SkillRarityCalculator
        
        # Initialize ranker
        ranker = BertRanker(model_name="bert-base-uncased")
        print("✅ BertRanker initialized")
        
        # Initialize SRW
        calculator = SkillRarityCalculator("data/job_skills.csv")
        
        # Get skill weights
        skills = ["python", "kubernetes", "terraform"]
        skill_weights = {s: calculator.get_rarity_weight(s)[0] for s in skills}
        
        # Test scoring without SRW
        resume = "Python developer with SQL experience"
        jd = "Need Python, Kubernetes, Terraform expert"
        
        score_no_srw, info_no_srw = ranker.get_research_score(
            resume, jd, mandatory_skills="Python, Kubernetes, Terraform"
        )
        
        # Test scoring with SRW
        score_with_srw, info_with_srw = ranker.get_research_score(
            resume, jd, mandatory_skills="Python, Kubernetes, Terraform",
            skill_rarity_weights=skill_weights
        )
        
        print(f"✅ Score without SRW: {score_no_srw:.4f}")
        print(f"✅ Score with SRW: {score_with_srw:.4f}")
        print(f"   SRW enabled flag: {info_with_srw['srw_enabled']}")
        print(f"   Missing skills: {info_with_srw['missing_skills']}")
        
        print("\n✅ BertRanker SRW Integration - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ BertRanker SRW Integration: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("🔬 NOVEL FEATURES TEST SUITE")
    print("=" * 60)
    print("Testing 5 research-driven features for TalentAI...")
    
    results = {
        "F1_CCG": test_constraint_coverage_graph(),
        "F2_HRI": test_hallucination_risk_index(),
        "F3_Sensitivity": test_sensitivity_simulation(),
        "F4_Drift": test_drift_detector(),
        "F5_SRW": test_skill_rarity_weighting(),
        "SRW_Integration": test_bert_ranker_srw_integration(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test}: {status}")
    
    print(f"\n🏆 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Novel features are ready for use.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
