#!/usr/bin/env python3
"""
Script to verify the cleaned datasets have consistent column structure.
"""

import csv
import os

def get_csv_headers(file_path):
    """Get CSV headers without using pandas to avoid NumPy issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            return headers
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # Define the cleaned CSV files
    cleaned_files = {
        'dataset_new.csv': './dataset_new.csv',
        'dataset_with_idea_density_new.csv': './dataset_with_idea_density_new.csv',
        'test_output_new.csv': './test_output_new.csv'
    }
    
    print("VERIFYING CLEANED DATASETS")
    print("=" * 50)
    
    # Get headers for each file
    file_headers = {}
    for name, path in cleaned_files.items():
        if os.path.exists(path):
            headers = get_csv_headers(path)
            if headers:
                file_headers[name] = headers
                print(f"\n{name}:")
                print(f"  Columns ({len(headers)}): {headers}")
            else:
                print(f"\n{name}: ERROR reading file")
        else:
            print(f"\n{name}: File not found")
    
    # Check consistency
    if len(file_headers) > 1:
        print(f"\nCONSISTENCY CHECK:")
        print("-" * 30)
        
        # Get first file's headers as reference
        reference_name = list(file_headers.keys())[0]
        reference_headers = file_headers[reference_name]
        
        all_consistent = True
        for name, headers in file_headers.items():
            if headers == reference_headers:
                print(f"✓ {name}: IDENTICAL to {reference_name}")
            else:
                print(f"✗ {name}: DIFFERENT from {reference_name}")
                all_consistent = False
                
                # Show differences
                missing = set(reference_headers) - set(headers)
                extra = set(headers) - set(reference_headers)
                if missing:
                    print(f"    Missing columns: {list(missing)}")
                if extra:
                    print(f"    Extra columns: {list(extra)}")
        
        if all_consistent:
            print(f"\n🎉 SUCCESS: All {len(file_headers)} cleaned datasets have identical column structure!")
            print(f"   Total columns: {len(reference_headers)}")
        else:
            print(f"\n❌ ISSUE: Datasets have inconsistent column structures")
    
    # Check for any remaining .1 columns
    print(f"\nCHECKING FOR REMAINING .1 COLUMNS:")
    print("-" * 40)
    
    found_dot_one = False
    for name, headers in file_headers.items():
        dot_one_cols = [col for col in headers if '.1' in col]
        if dot_one_cols:
            print(f"⚠️  {name} still has .1 columns: {dot_one_cols}")
            found_dot_one = True
        else:
            print(f"✓ {name}: No .1 columns found")
    
    if not found_dot_one:
        print(f"\n🎉 SUCCESS: No .1 columns found in any cleaned dataset!")

if __name__ == "__main__":
    main()
