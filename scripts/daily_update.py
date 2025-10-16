
#!/usr/bin/env python3
"""
Daily update script for agentic science research pipeline.
This script runs the daily update process including:
1. Fetching new arxiv papers
2. Generating research reports
3. Syncing data
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main function to run the daily update process."""
    print(f"Starting daily update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    # Step 1: Update arxiv papers
    if not run_command("python -m dw.arxiv.update", "Fetching new arxiv papers"):
        print("Failed to fetch arxiv papers")
        sys.exit(1)
    
    # Step 2: Generate research reports
    if not run_command("python -m pipeline.graph", "Generating research reports"):
        print("Failed to generate research reports")
        sys.exit(1)
    
    # Step 3: Sync data
    if not run_command("bash data/sync.sh", "Syncing data"):
        print("Failed to sync data")
        sys.exit(1)
    
    print(f"Daily update completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    import schedule
    import time
    
    # Schedule the job to run daily at 2:00 AM
    schedule.every().day.at("02:00").do(main)
    
    print("Daily update scheduler started. Will run at 2:00 AM every day.")
    print("Press Ctrl+C to stop the scheduler.")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
