'''
Filter pandas dataframe by a list of values in a column.
'''
import argparse
import sys

import pandas as pd


# Create the parser
parser = argparse.ArgumentParser(description='Filter pandas dataframe.')

# Add the arguments
parser.add_argument('--input_file', required=True, help='Input .csv file  or - to read from the standard input.')
parser.add_argument('--output_file', required=True, help='Output .csv file or - to write to the standard output.')
parser.add_argument('--filter', help='Filter expression to eval(), using the variable df, if ommited, copies the input '
                    'to the output.')
parser.add_argument('--summarize', action='store_true', help='Summarizes the data. If --output_file is -, summary is '
                    'printed to stderr.')

# Parse the arguments
args = parser.parse_args()
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
FILTER = args.filter
SUMMARIZE = args.summarize

# Read the input data
if INPUT_FILE == '-':
    df = pd.read_csv(sys.stdin)
else:
    df = pd.read_csv(INPUT_FILE)

# Filter the data
if FILTER is not None:
    df = df[eval(FILTER, {'df': df}, {})]

# Write the output data
if OUTPUT_FILE == '-':
    try:
        df.to_csv(sys.stdout, index=False)
    except BrokenPipeError:
        pass
else:
    df.to_csv(OUTPUT_FILE, index=False)

# Print the summary
if SUMMARIZE:
    print(f'rows: {len(df)}, cols: {len(df.columns)}, column names:', ', '.join(df.columns),
          file=sys.stderr if OUTPUT_FILE == '-' else sys.stdout)
