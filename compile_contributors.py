import subprocess
import re
import json
import csv
import sys

def run_command(command, cwd="."):
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr}")
        return None
    return result.stdout

def get_pr_number(subject):
    match = re.search(r"\(#(\d+)\)$", subject.strip())
    if match:
        return match.group(1)
    return None

def get_pr_details(pr_number):
    output = run_command(f"gh pr view {pr_number} --json number,title,author,body")
    if not output:
        return None
    return json.loads(output)

def extract_prs_from_body(body):
    # Matches patterns like #123 or https://github.com/microsoft/onnxruntime/pull/123
    prs = re.findall(r"(?:#|/pull/)(\d+)", body)
    return list(set(prs))

def get_all_prs_in_branch(branch, limit=1000):
    print(f"Fetching history for {branch}...")
    log = run_command(f"git log {branch} -n {limit} --oneline")
    if not log:
        return {}

    all_prs = {} # pr_number -> {title, author, original_pr, cherry_pick_commit}

    for line in log.splitlines():
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        commit_id = parts[0]
        subject = parts[1]

        pr_num = get_pr_number(subject)
        if pr_num:
            details = get_pr_details(pr_num)
            if details:
                # Check if it's a cherry-pick round PR
                if "cherry pick" in subject.lower() or "cherry-pick" in subject.lower() or "cherrypick" in subject.lower():
                    print(f"Processing cherry-pick round PR #{pr_num}...")
                    original_pr_nums = extract_prs_from_body(details['body'])
                    for op_num in original_pr_nums:
                        if op_num == pr_num: continue # Avoid self-reference
                        op_details = get_pr_details(op_num)
                        if op_details:
                            all_prs[op_num] = {
                                'title': op_details['title'],
                                'author': op_details['author']['login'],
                                'cherry_pick_commit': commit_id,
                                'cherry_pick_pr': pr_num
                            }
                else:
                    all_prs[pr_num] = {
                        'title': details['title'],
                        'author': details['author']['login'],
                        'cherry_pick_commit': commit_id,
                        'cherry_pick_pr': None
                    }
        else:
            # Not a PR, use commit author
            author_info = run_command(f"git show -s --format=\"%an\" {commit_id}")
            if author_info:
                author_name = author_info.strip()
                # Try to map author name to github handle if possible, but for now just use name
                all_prs[f"commit_{commit_id}"] = {
                    'title': subject,
                    'author': author_name,
                    'cherry_pick_commit': commit_id,
                    'cherry_pick_pr': None
                }

    return all_prs

def main():
    branch_23 = "origin/rel-1.23.2"
    branch_24 = "origin/rel-1.24.1"

    prs_23 = get_all_prs_in_branch(branch_23)
    prs_24 = get_all_prs_in_branch(branch_24)

    new_pr_keys = set(prs_24.keys()) - set(prs_23.keys())

    contributors = {} # username -> count
    details = []

    for key in sorted(new_pr_keys, key=lambda x: str(x)):
        info = prs_24[key]
        author = info['author']
        contributors[author] = contributors.get(author, 0) + 1

        details.append({
            'original_id': key,
            'title': info['title'],
            'author': author,
            'cherry_pick_commit': info['cherry_pick_commit'],
            'cherry_pick_pr': info['cherry_pick_pr']
        })

    # Sort contributors by count descending
    sorted_contributors = sorted(contributors.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Summary ---")
    output_users = [f"@{u}" for u, c in sorted_contributors]
    print(", ".join(output_users))

    # Write details to CSV
    with open('detail.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['original_id', 'author', 'title', 'cherry_pick_commit', 'cherry_pick_pr'])
        writer.writeheader()
        for row in details:
            writer.writerow(row)

    print(f"\nDetailed information written to detail.csv. Total contributors: {len(contributors)}")

if __name__ == "__main__":
    main()
