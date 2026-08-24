import sys
sys.path.append('d:/GIT/Devlens')
import numpy as np
import pandas as pd
from devlens.models.project_fit import (
    ProjectRequirement,
    DeveloperProfile,
    compute_language_similarity,
    compute_fit_score,
    load_developer_pool,
    rank_developers,
    rule_based_match
)

# 1. Zero-byte developer edge case test
print("=== 1. ZERO-BYTE DEVELOPER EDGE CASE TEST ===")
zero_dev = DeveloperProfile(
    username="test_zero_activity_user",
    assigned_archetype="Unclassified / Low Signal",
    primary_language="Unknown",
    language_bytes={},
    doc_score=0.0,
    eng_maturity_score=0.0,
)

req_test = ProjectRequirement(
    name="Test Frontend Requirement",
    required_archetypes=["Frontend Developer"],
    min_doc_score=0.40,
    min_eng_maturity_score=0.30,
    required_languages=["JavaScript", "TypeScript"],
    weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20}
)

lang_sim = compute_language_similarity(zero_dev.language_bytes, req_test.required_languages)
fit_res = compute_fit_score(zero_dev, req_test)

print(f"Zero Developer language similarity: {lang_sim} (type: {type(lang_sim)})")
print(f"Zero Developer total fit score:     {fit_res['fit_score']} (NaN check: {np.isnan(fit_res['fit_score'])})")
print(f"Zero Developer explanation:         {fit_res['explanation']}")

# 2. Check full 188 pool for any NaNs
print("\n" + "="*80)
print("=== 2. FULL 188 POOL SANITY CHECK (ZERO NANS/INFS) ===")
print("="*80)
pool = load_developer_pool()
print(f"Total pool size: {len(pool)}")

all_ranked = rank_developers(req_test, pool)
nan_count = sum(1 for r in all_ranked if np.isnan(r["fit_score"]) or np.isinf(r["fit_score"]))
print(f"Total NaN/Inf scores across all 188 developers: {nan_count}")

# 3. Check Backend and Full-Stack heuristic students
print("\n" + "="*80)
print("=== 3. BACKEND & FULL-STACK HEURISTIC STUDENTS VERIFICATION ===")
print("="*80)

backend_req = ProjectRequirement(
    name="Enterprise Core Services (Backend Focus)",
    required_archetypes=["Backend Developer"],
    min_doc_score=0.30,
    min_eng_maturity_score=0.30,
    required_languages=["C", "C++", "Java", "Python"],
    weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20}
)

fullstack_req = ProjectRequirement(
    name="End-to-End Product Platform (Full-Stack Focus)",
    required_archetypes=["Full-Stack Developer"],
    min_doc_score=0.30,
    min_eng_maturity_score=0.30,
    required_languages=["JavaScript", "Python", "HTML", "CSS"],
    weights={"archetype_match": 0.40, "language_similarity": 0.40, "doc_maturity_fit": 0.20}
)

# Backend ranking
backend_ranked = rank_developers(backend_req, pool)
backend_rule = rule_based_match(backend_req, pool)

print("\n--- Backend Requirement: Top 5 Ranked Candidates ---")
print(f"{'Rank':<5} {'Username':<26} {'Fit Score':<10} {'Arch':<6} {'Lang':<6} {'Doc/Eng':<8} {'Assigned Archetype':<22}")
print("-" * 85)
for r in backend_ranked[:5]:
    print(f"#{r['rank']:<4} {r['username']:<26} {r['fit_score']:<10.4f} {r['archetype_score']:<6.2f} {r['language_score']:<6.2f} {r['doc_eng_score']:<8.2f} {r['assigned_archetype']:<22}")

print("\nSpecific Check for Backend Students:")
for uname in ["Niranjan-017", "priyadharshinidhandapani"]:
    b_dev = next((r for r in backend_ranked if r["username"] == uname), None)
    b_rule = next((m for m in backend_rule if m["username"] == uname), None)
    if b_dev:
        print(f"• {uname:<26}: Rank #{b_dev['rank']:<3} | Fit Score: {b_dev['fit_score']:.4f} (Arch={b_dev['archetype_score']:.2f}, Lang={b_dev['language_score']:.2f}, Doc/Eng={b_dev['doc_eng_score']:.2f}) | Rule Match: {b_rule['match']}")
        print(f"  Explanation: {b_dev['explanation']}")

# Full-Stack ranking
fs_ranked = rank_developers(fullstack_req, pool)
fs_rule = rule_based_match(fullstack_req, pool)

print("\n--- Full-Stack Requirement: Top 5 Ranked Candidates ---")
print(f"{'Rank':<5} {'Username':<26} {'Fit Score':<10} {'Arch':<6} {'Lang':<6} {'Doc/Eng':<8} {'Assigned Archetype':<22}")
print("-" * 85)
for r in fs_ranked[:5]:
    print(f"#{r['rank']:<4} {r['username']:<26} {r['fit_score']:<10.4f} {r['archetype_score']:<6.2f} {r['language_score']:<6.2f} {r['doc_eng_score']:<8.2f} {r['assigned_archetype']:<22}")

print("\nSpecific Check for Full-Stack Student:")
fs_dev = next((r for r in fs_ranked if r["username"] == "ggomathi2007"), None)
fs_rule_match = next((m for m in fs_rule if m["username"] == "ggomathi2007"), None)
if fs_dev:
    print(f"• ggomathi2007: Rank #{fs_dev['rank']:<3} | Fit Score: {fs_dev['fit_score']:.4f} (Arch={fs_dev['archetype_score']:.2f}, Lang={fs_dev['language_score']:.2f}, Doc/Eng={fs_dev['doc_eng_score']:.2f}) | Rule Match: {fs_rule_match['match']}")
    print(f"  Explanation: {fs_dev['explanation']}")
