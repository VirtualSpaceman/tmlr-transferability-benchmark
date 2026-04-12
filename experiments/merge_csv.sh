#!/bin/bash

# Function to merge two CSV files
merge_csv_files() {
    file1="$1"
    file2="$2"
    output_file="$3"
    
    # Check if both files exist
    if [[ ! -f "$file1" || ! -f "$file2" ]]; then
        echo "Error: One or both input files do not exist."
        exit 1
    fi

    # Merge the two CSV files (skip header of the second file)
    head -n 1 "$file1" > "$output_file"     # Add header from the first file
    tail -n +2 "$file1" >> "$output_file"    # Add data from the first file
    tail -n +2 "$file2" >> "$output_file"    # Add data from the second file
    
    echo "Merged file saved as $output_file"
}

# Example usage: ./merge_csv.sh file1.csv file2.csv output.csv
merge_csv_files "$1" "$2" "$3"
