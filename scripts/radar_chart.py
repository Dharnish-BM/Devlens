import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_radar(row, save_path=None):
    """Create a radar chart for one student's skill profile."""
    
    categories = ["Consistency", "Tech Breadth", "Project Activity", "Collaboration"]
    values = [
        row["consistency_score"],
        row["tech_breadth_score"],
        row["project_activity_score"],
        row["collaboration_score"]
    ]
    
    # Radar charts need to "close the loop" - repeat first value at the end
    values += values[:1]
    
    # Calculate angle for each category
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#2563eb')
    ax.fill(angles, values, alpha=0.25, color='#2563eb')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
    
    username = row["username"]
    overall = row["overall_skill_score"]
    ax.set_title(f"{username}\nOverall Skill Score: {overall}", fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()

def plot_all_students(csv_path="college_profiles_features.csv", output_folder="radar_charts"):
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        filename = f"{output_folder}/{row['username']}_radar.png"
        plot_radar(row, save_path=filename)
    
    print(f"\n✅ Generated {len(df)} radar charts in '{output_folder}/' folder")

if __name__ == "__main__":
    plot_all_students()