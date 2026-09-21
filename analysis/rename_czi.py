#
# This script was utilised to rename all the czi files, filling gaps in their name with understores for `read_czi.py`. 
#

import os

# Using the absolute path to eliminate any directory confusion
target_folder = "/Users/angusgray/Desktop/V0d-Histology/HiPlexUp-V0d-Pipeline/raw_czi"

def sanitize_filenames(folder_path, dry_run=True):
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    print(f"Scanning folder: {os.path.abspath(folder_path)}")
    if dry_run:
        print("=== DRY RUN MODE: No files will be changed yet ===")
    else:
        print("=== LIVE MODE: Renaming files now ===")

    rename_count = 0

    # os.walk automatically searches all subdirectories and subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Check if there are spaces/gaps in the filename
            if " " in filename:
                old_file_path = os.path.join(root, filename)
                
                # Replace spaces with underscores
                new_filename = filename.replace(" ", "_")
                new_file_path = os.path.join(root, new_filename)
                
                # Print relative path from the raw_czi folder for cleaner logs
                rel_root = os.path.relpath(root, folder_path)
                display_dir = "" if rel_root == "." else f"[{rel_root}] "
                
                print(f"Match found in {display_dir}:")
                print(f"  Old: {filename}")
                print(f"  New: {new_filename}\n")
                
                rename_count += 1
                
                if not dry_run:
                    os.rename(old_file_path, new_file_path)

    print(f"Scan complete. Total files matching criteria: {rename_count}")
    if dry_run and rename_count > 0:
        print("\nTo apply these changes, change 'dry_run=True' to 'dry_run=False' at the bottom of the script.")

if __name__ == "__main__":
    # Test it with a dry run first!
    sanitize_filenames(target_folder, dry_run=True)
