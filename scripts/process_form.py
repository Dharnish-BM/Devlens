import pandas as pd
from build_profile import build_single_profile
import time

# Load form responses
form_df = pd.read_csv("form_responses.csv")

# Print columns so you can confirm exact header names
print("Columns found in form:", list(form_df.columns))

# --- IMPORTANT: update this to match your actual GitHub username column name ---
USERNAME_COLUMN = "GitHub Username "  

usernames = form_df[USERNAME_COLUMN].dropna().str.strip().tolist()
print(f"\nFound {len(usernames)} usernames to process\n")

all_rows = []
failed_usernames = []

for i, username in enumerate(usernames):
    print(f"[{i+1}/{len(usernames)}] Processing {username}...")
    row = build_single_profile(username)
    
    if row:
        all_rows.append(row)
    else:
        failed_usernames.append(username)
    
    time.sleep(1)  # stay polite to GitHub's API

# Save successful profiles
result_df = pd.DataFrame(all_rows)
result_df.to_csv("college_profiles_raw.csv", index=False)
print(f"\n✅ Saved {len(result_df)} profiles to college_profiles_raw.csv")

# Save failed usernames separately so you can investigate (typos, private profiles, etc.)
if failed_usernames:
    pd.DataFrame({"failed_username": failed_usernames}).to_csv("failed_usernames.csv", index=False)
    print(f"⚠️ {len(failed_usernames)} usernames failed — saved to failed_usernames.csv for review")