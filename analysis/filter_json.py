'''
Filter pandas dataframe by a list of values in a column.
'''
import argparse
import json
import sys


# Create the parser
parser = argparse.ArgumentParser(description='Filter pandas dataframe.')

# Add the arguments
parser.add_argument('--input_file', required=True, help='Input .json file  or - to read from the standard input.')
parser.add_argument('--output_file', required=True, help='Output .json file or - to write to the standard output.')
parser.add_argument('--keys', nargs='+', help='Space separated list of keys to select from the input file, in the '
                    'format parent/child/grandchild. If omitted, the whole input file is copied to the output file.')
parser.add_argument('--prefix', help='Adds the entire output under the specified key on the output file.')
parser.add_argument('--indent', nargs='?', const=4, type=int, help='If specified, the output will be formatted, and '
                    'indented by the specified number of spaces (4 by default). By default, the output is delivered '
                    'in a single unfomatted line.')
parser.add_argument('--sorted', action='store_true', help='If specified, the output will be sorted by the keys.')

# Parse the arguments
args = parser.parse_args()
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
KEYS = args.keys
PREFIX = args.prefix
INDENT = args.indent
SORTED = args.sorted

# Read the input data
if INPUT_FILE == '-':
    json_in = json.load(sys.stdin)
else:
    with open(INPUT_FILE, 'rt', encoding='utf-8') as f:
        json_in = json.load(f)

# Filter the data
if KEYS is None:
    json_out = json_in
else:
    json_out = {}
    try:
        for key_path in KEYS:
            json_in_value = json_in
            json_out_value = json_out
            key_elements = key_path.split('/')
            for key in key_elements[:-1]:
                key = int(key) if isinstance(json_in_value, list) else key
                json_out_value[key] = {}
                json_out_value = json_out_value[key]
                json_in_value = json_in_value[key]
            key = key_elements[-1]
            key = int(key) if isinstance(json_in_value, list) else key
            json_out_value[key] = json_in_value[key]
    except (KeyError, TypeError, IndexError) as e:
        raise ValueError(f'Could not find "{key_path}" in input. Element "{key}" failed') from e

if PREFIX is not None:
    json_out = {PREFIX: json_out}

# Write the output data
if OUTPUT_FILE == '-':
    try:
        json.dump(json_out, sys.stdout, indent=INDENT, sort_keys=SORTED)
    except BrokenPipeError:
        pass
else:
    with open(OUTPUT_FILE, 'wt', encoding='utf-8') as f:
        json_in = json.dump(json_out, f, indent=INDENT, sort_keys=SORTED)
