# This is a quick and dirty script to print out the number of tuples used in each experiment of the tables
# It is not part of the main analysis pipeline
import csv
import io
import json
import sys


ridgeline_json_path = sys.argv[1]
tuple_count_csv_path = sys.argv[2]

def join_json_csv(json_data, csv_file):
    # Parse the JSON data

    # Read the CSV data into a dictionary for easy lookup
    csv_dict = {}
    for row in csv.reader(csv_file):
        prefix, scorers_label, input_count = row
        csv_dict[(prefix, scorers_label)] = input_count

    print(csv_dict)

    def split_prefix(raw):
        # Splits at the first underscore
        p = raw.find("_")
        return (raw[:p], raw[p+1:]) if p != -1 else ("", raw)

    # Iterate over the JSON data and output the required format
    for item in json_data:
        scorer_raw = item["scorer"]
        for prefix_raw in item["prefixes"]:
            # Normalize the prefix (considering null and empty values)
            prefix, scorers_label = split_prefix(((prefix_raw or "") + scorer_raw))

            input_count = csv_dict.get((prefix, scorers_label), None)
            if input_count is not None:
                print(f"{prefix},{scorers_label},{input_count}")

json_parsed = json.load(open(ridgeline_json_path))
tuple_count_csv_file = open(tuple_count_csv_path)
join_json_csv(json_parsed["rows"], tuple_count_csv_file)
