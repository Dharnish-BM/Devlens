import pandas as pd

def normalize(series):
    """Min-max scale a column to 0-100. Handles case where all values are equal."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return series.apply(lambda x: 50.0)  # neutral score if no variation
    return ((series - min_val) / (max_val - min_val) * 100).round(2)

def engineer_features(df):
    df = df.copy()
    
    # --- 1. Consistency Score (already computed, just carry it over) ---
    df["consistency_score"] = df["consistency_score"].fillna(0)
    
    # --- 2. Tech Breadth Score ---
    df["tech_breadth_score"] = normalize(df["num_unique_languages"])
    
    # --- 3. Project Activity Score ---
    # Combine non-fork repos (original work) + stars + forks received (community validation)
    # Weighted: repos matter most, stars/forks are a bonus signal
    df["_activity_raw"] = (
        df["non_fork_repos"] * 3 +
        df["total_stars"] * 1 +
        df["total_forks"] * 1
    )
    df["project_activity_score"] = normalize(df["_activity_raw"])
    
    # --- 4. Collaboration Score ---
    # PRs + issues = active collaboration; follower/following ratio = community engagement
    df["_collab_raw"] = (
        df["pr_contributions_past_year"] * 2 +
        df["issue_contributions_past_year"] * 1
    )
    df["collaboration_score"] = normalize(df["_collab_raw"])
    
    # --- 5. Overall Composite Score (simple average for now) ---
    df["overall_skill_score"] = (
        df["consistency_score"] +
        df["tech_breadth_score"] +
        df["project_activity_score"] +
        df["collaboration_score"]
    ) / 4
    df["overall_skill_score"] = df["overall_skill_score"].round(2)
    
    # Drop intermediate helper columns
    df = df.drop(columns=["_activity_raw", "_collab_raw"])
    
    return df

if __name__ == "__main__":
    raw_df = pd.read_csv("college_profiles_raw.csv")
    
    print(f"Loaded {len(raw_df)} profiles")
    
    featured_df = engineer_features(raw_df)
    
    # Show key columns
    display_cols = ["username", "consistency_score", "tech_breadth_score", 
                     "project_activity_score", "collaboration_score", "overall_skill_score"]
    print("\n", featured_df[display_cols].sort_values("overall_skill_score", ascending=False))
    
    featured_df.to_csv("college_profiles_features.csv", index=False)
    print(f"\n✅ Saved to college_profiles_features.csv")