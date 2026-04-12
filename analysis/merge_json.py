import argparse
from copy import deepcopy
import json
import sys


def recursive_merge(dict1, dict2, *, force=False):
    '''
    Recursively merges dict2 into dict1 with precedence for dict2's values.

    This function will merge two dictionaries together by iterating through dict2
    and adding its key-value pairs to dict1. If the same key is present in both
    dictionaries and the values are themselves dictionaries, it will recursively
    merge those sub-dictionaries. If there is a type mismatch between the values
    of the same key, it will raise a ValueError unless the `force` parameter is
    set to True, in which case the value from dict2 will overwrite the value in dict1.

    Args:
        dict1 (dict): The first dictionary to merge.
        dict2 (dict): The second dictionary to merge; values take precedence over dict1.
        force (bool): If True, allows a non-dictionary value to overwrite a dictionary
            value on collision; defaults to False.

    Returns:
        dict: The merged dictionary with dict2's values taking precedence.

    Raises:
        ValueError: If a collision occurs between a dictionary and non-dictionary value,
            and `force` is False.
    '''
    def _merge(d1, d2, force, path=''):
        for key in d2:
            if key in d1:
                d1_dict = isinstance(d1[key], dict)
                d2_dict = isinstance(d2[key], dict)
                if d1_dict and d2_dict:
                    d1[key] = _merge(d1[key], d2[key], force, f'{path}"{key}".')
                elif d1_dict != d2_dict and not force:
                    raise ValueError(f'Collision at {path}"{key}": dictionary and non-dictionary values.')
                else:
                    d1[key] = d2[key]
            else:
                d1[key] = d2[key]
        return d1
    return _merge(deepcopy(dict1), dict2, force)


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Merge multiple JSON files into one.')
    parser.add_argument('--inputs', nargs='+', required=True, help='List of input JSON files to merge. In the case of collisions, the later files in the list take precedence.')
    parser.add_argument('--output', required=True, help='Output JSON file where the merged result will be saved.')
    parser.add_argument('--force', action='store_true', help='If set, allows a non-dictionary value to overwrite a dictionary value on collision.')

    # Parse arguments
    args = parser.parse_args()

    # Initialize the result as an empty dictionary
    merged_result = {}

    # Iterate over input files and merge their contents
    try:
        for idx, input_file in enumerate(args.inputs):
            with open(input_file, 'rt', encoding='utf-8') as f:
                try:
                    # Load the current JSON file
                    current_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {input_file}: {e}")
                    return

                # If it's the first file, just assign it to merged_result
                # Otherwise, merge it with the existing merged_result
                if idx == 0:
                    merged_result = current_data
                else:
                    merged_result = recursive_merge(merged_result, current_data, force=args.force)
    except ValueError as e:
        if str(e).startswith('Collision'):
            print(f"Error merging files: {e}")
            return 1
        else:
            raise e

    # Write the merged result to the output file
    with open(args.output, 'wt', encoding='utf-8') as f:
        json.dump(merged_result, f, indent=4)

    print(f"Merged {len(args.inputs)} files into '{args.output}' successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
