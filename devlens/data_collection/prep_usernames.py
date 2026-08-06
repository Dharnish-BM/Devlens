import os
import re
import argparse
from typing import List, Dict, Tuple, Set

GITHUB_USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$")

RESERVED_WORDS = {
    "topics", "features", "settings", "pricing", "about", "marketplace",
    "search", "orgs", "organizations", "login", "join", "explore",
    "trending", "security", "enterprise", "customer-stories", "readme",
    "sponsors", "collections", "events", "issues", "pulls", "notifications"
}


def parse_and_clean_line(raw_line: str) -> Tuple[str, str]:
    """Extract a candidate GitHub username from a raw line string.
    
    Returns:
        (extracted_username, error_reason)
        If extraction succeeds, error_reason is empty string "".
    """
    text = raw_line.strip()
    if not text:
        return "", "empty_line"

    # 1. GitHub Pages URL format: https://username.github.io/path
    gh_pages_match = re.search(r'https?://([a-zA-Z0-9\-]+)\.github\.io', text, re.IGNORECASE)
    if gh_pages_match:
        candidate = gh_pages_match.group(1)
        return candidate, ""

    # 2. Standard GitHub URL format: https://github.com/username
    gh_url_match = re.search(r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9\-]+)', text, re.IGNORECASE)
    if gh_url_match:
        candidate = gh_url_match.group(1)
        return candidate, ""

    # 3. Text with notes/parentheses e.g., "gldinesh1211 (gldinesh)"
    if "(" in text or ")" in text:
        # Extract first non-space token before parenthesis or inside
        first_token = text.split("(")[0].strip()
        if first_token:
            candidate = first_token.lstrip("@").strip()
            return candidate, ""

    # 4. Non-GitHub URL check
    if text.startswith("http://") or text.startswith("https://"):
        return "", f"URL is not a GitHub link: '{text}'"

    # 5. Bare username or @username
    candidate = text.lstrip("@").strip()
    # Strip any trailing path or query if someone wrote username/repo
    candidate = candidate.split("/")[0].split("?")[0].split("#")[0]
    
    return candidate, ""


def validate_username(username: str) -> Tuple[bool, str]:
    """Validate whether username conforms to GitHub username criteria."""
    if not username:
        return False, "Empty username after parsing"

    if username.lower() in RESERVED_WORDS:
        return False, f"'{username}' is a reserved GitHub system route"

    if not GITHUB_USERNAME_REGEX.match(username):
        return False, f"'{username}' does not match valid GitHub username format (1-39 alphanumeric/hyphen chars, cannot start/end with hyphen)"

    return True, ""


def process_classmate_links(
    input_file: str = "data/classmate_links.txt",
    output_file: str = "data/classmate_usernames.txt"
) -> Dict[str, any]:
    """Process classmate_links.txt, clean usernames, deduplicate, and write output file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: '{input_file}'")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    total_input_lines = len(raw_lines)
    valid_usernames: List[str] = []
    seen_lower: Set[str] = set()
    failed_lines: List[Dict[str, any]] = []
    duplicate_count = 0
    empty_lines_count = 0

    for idx, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            empty_lines_count += 1
            continue

        candidate, parse_err = parse_and_clean_line(stripped)
        if parse_err and parse_err != "empty_line":
            failed_lines.append({
                "line_num": idx,
                "raw_text": stripped,
                "reason": parse_err
            })
            continue

        is_valid, val_err = validate_username(candidate)
        if not is_valid:
            failed_lines.append({
                "line_num": idx,
                "raw_text": stripped,
                "reason": val_err
            })
            continue

        # Deduplication check (case-insensitive)
        cand_lower = candidate.lower()
        if cand_lower in seen_lower:
            duplicate_count += 1
        else:
            seen_lower.add(cand_lower)
            valid_usernames.append(candidate)

    # Write cleaned usernames to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for uname in valid_usernames:
            f.write(f"{uname}\n")

    summary = {
        "input_file": input_file,
        "output_file": output_file,
        "total_input_lines": total_input_lines,
        "empty_lines_count": empty_lines_count,
        "valid_unique_usernames": len(valid_usernames),
        "duplicates_removed": duplicate_count,
        "failed_lines_count": len(failed_lines),
        "failed_lines": failed_lines,
        "usernames": valid_usernames,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Clean and validate classmate GitHub links/usernames.")
    parser.add_argument("--input-file", type=str, default="data/classmate_links.txt", help="Input links file")
    parser.add_argument("--output-file", type=str, default="data/classmate_usernames.txt", help="Output cleaned usernames file")
    args = parser.parse_args()

    summary = process_classmate_links(input_file=args.input_file, output_file=args.output_file)

    print("\n" + "=" * 60)
    print("  DEV LENS - CLASSMATE USERNAMES PREPARATION SUMMARY")
    print("=" * 60)
    print(f" Input File             : {summary['input_file']}")
    print(f" Output File            : {summary['output_file']}")
    print(f" Total Input Lines      : {summary['total_input_lines']}")
    print(f" Empty Lines Skipped    : {summary['empty_lines_count']}")
    print(f" Valid Unique Usernames : {summary['valid_unique_usernames']}")
    print(f" Duplicates Removed     : {summary['duplicates_removed']}")
    print(f" Failed Lines Count     : {summary['failed_lines_count']}")

    if summary["failed_lines"]:
        print("\n  [!] FAILED / UNPARSED LINES FOR MANUAL REVIEW:")
        for item in summary["failed_lines"]:
            print(f"    Line {item['line_num']:>3}: {item['raw_text']:<45} -> Reason: {item['reason']}")
    else:
        print("\n  [✓] All non-empty lines parsed and validated cleanly with 0 errors!")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
