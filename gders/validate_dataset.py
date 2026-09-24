import json
from pathlib import Path
import numpy as np
from gders.data.repository_assessment import is_technical_comment_heuristic

def validate():
    raw_dir = Path("data/gders/raw_comments")
    manifest_file = Path("data/gders/processed/dataset_manifest.json")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    jsonl_files = sorted(list(raw_dir.glob("*.jsonl")))

    all_comment_ids = set()
    duplicates_found = 0
    malformed_lines = 0
    repo_stats = {}
    all_comment_lengths = []
    all_technical_flags = []
    total_comments = 0
    total_reviewers = set()
    all_prs_with_comments = set()

    for f_path in jsonl_files:
        repo_name = f_path.stem.replace("__", "/")
        repo_comment_ids = set()
        repo_comments = 0
        repo_prs = set()
        repo_reviewers = set()
        repo_lengths = []
        repo_tech_count = 0
        
        with open(f_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception as e:
                    malformed_lines += 1
                    continue
                
                # Check repo field matches
                if rec.get("repository") != repo_name:
                    print(f"Mismatch repo field in {f_path}: {rec.get('repository')}")
                
                cid = rec.get("comment_id")
                if cid in repo_comment_ids:
                    duplicates_found += 1
                repo_comment_ids.add(cid)
                all_comment_ids.add(cid)
                
                pr_num = rec.get("pull_request_number")
                if isinstance(pr_num, int):
                    repo_prs.add(pr_num)
                    all_prs_with_comments.add(f"{repo_name}#{pr_num}")
                
                body = rec.get("comment_body", "")
                repo_lengths.append(len(body))
                all_comment_lengths.append(len(body))
                
                is_tech, _ = is_technical_comment_heuristic(body)
                all_technical_flags.append(is_tech)
                if is_tech:
                    repo_tech_count += 1
                    
                login = rec.get("commenter_login")
                if login and login != "unknown":
                    repo_reviewers.add(login)
                    total_reviewers.add(login)
                    
                repo_comments += 1
                total_comments += 1
                
        repo_stats[repo_name] = {
            "comments": repo_comments,
            "prs_collected": manifest["repositories"].get(repo_name, {}).get("prs_collected", 0),
            "unique_prs_with_comments": len(repo_prs),
            "unique_reviewers": len(repo_reviewers),
            "avg_comment_length": round(float(np.mean(repo_lengths)), 2) if repo_lengths else 0.0,
            "median_comment_length": round(float(np.median(repo_lengths)), 2) if repo_lengths else 0.0,
            "technical_comments": repo_tech_count,
            "technical_ratio": round(repo_tech_count / repo_comments, 4) if repo_comments else 0.0,
        }

    total_prs = manifest["summary_statistics"]["total_prs_collected"]
    prs_with_comments_count = len(all_prs_with_comments)

    print("=== DATASET VALIDATION SUMMARY ===")
    print(f"Total repositories: {len(jsonl_files)}")
    print(f"Total PRs collected across repos: {total_prs}")
    print(f"Total PRs with review comments: {prs_with_comments_count} ({prs_with_comments_count/total_prs:.1%})")
    print(f"Total review comments collected: {total_comments}")
    print(f"Total unique comment IDs: {len(all_comment_ids)}")
    print(f"Duplicates within repos: {duplicates_found}")
    print(f"Malformed lines: {malformed_lines}")
    print(f"Total unique reviewers: {len(total_reviewers)}")
    print(f"Average comments per PR (all PRs): {round(total_comments / total_prs, 2)}")
    print(f"Average comments per active PR: {round(total_comments / prs_with_comments_count, 2) if prs_with_comments_count else 0}")
    print(f"Overall Avg Comment Length: {round(float(np.mean(all_comment_lengths)), 2)} chars")
    print(f"Overall Median Comment Length: {round(float(np.median(all_comment_lengths)), 2)} chars")
    print(f"Overall Technical Comments (Heuristic): {sum(all_technical_flags)}")
    print(f"Overall Technical Ratio: {round(sum(all_technical_flags)/len(all_technical_flags), 4) if all_technical_flags else 0.0:.2%}")

    print("\n=== PER-REPOSITORY BREAKDOWN ===")
    for r, s in repo_stats.items():
        print(f"{r:<32} | PRs: {s['prs_collected']:2d} | Active PRs: {s['unique_prs_with_comments']:2d} | Comments: {s['comments']:3d} | Reviewers: {s['unique_reviewers']:2d} | TechRatio: {s['technical_ratio']:6.1%} | AvgLen: {s['avg_comment_length']:5.1f}")

if __name__ == "__main__":
    validate()
