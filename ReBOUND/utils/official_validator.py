# Submission validator for HuMob 2026
#
# Usage: python humob2026_validator.py <submission_file_path>
#
# Line numbers in error messages are 0-indexed.
#
# Assertions:
#   - The submission file can be read and parsed
#   - All required evaluation dates (Feb & Mar 2024, excluding NA days) are present
#   - No dates outside Feb & Mar 2024 are included (NA days are allowed and ignored)
#   - Each evaluation date contains at least one edge with a non-empty destination set
#   - No duplicate edges (guaranteed by dict structure; duplicate dates are rejected)
#   - All values are non-negative
#   - All origin and destination coordinates are within the valid area, or -1_-1

import sys
import ast
import re


# Valid lat-lon
LAT_MIN = 1
LAT_MAX = 70
LON_MIN = 1
LON_MAX = 100


# Prepare date_str set
feb_set = {"202402{:02d}".format(i) for i in range(1, 29 + 1)}
mar_set = {"202403{:02d}".format(i) for i in range(1, 31 + 1)}
feb_set.remove("20240202")
mar_set.remove("20240305")
date_str_set = feb_set.union(mar_set)


def error(message):
    print(message)
    sys.exit(1)

def in_valid_area(lat, lon):
    if lat >= LAT_MIN and lat <= LAT_MAX and lon >= LON_MIN and lon <= LON_MAX:
        return True
    if lat == -1 and lon == -1:
        return True
    return False

def parse_od_matrix(line_num, od_matrix_str):
    try:
        od_matrix = ast.literal_eval(od_matrix_str)
    except:
        error("Error at line {}: Could not parse the OD matrix in the second column.".format(line_num))

    # Check the type
    if type(od_matrix) != dict:
        error("Error at line {}: The OD matrix in the second column must be a dict.".format(line_num))

    # Emptiness check
    if len(od_matrix) == 0:
        error("Error at line {}: The OD matrix is empty.".format(line_num))

    # Key-value check for the outer dict
    for orig_key, v in od_matrix.items():
        # Check the key
        if not re.match(r"^-?\d+_-?\d+$", orig_key):
            error("Error at line {}: Origin key `{}` has an invalid format.".format(line_num, orig_key))

        # Check the lat-lon
        lat_str, lon_str = orig_key.split("_")
        lat = int(lat_str)
        lon = int(lon_str)
        if not in_valid_area(lat, lon):
            error("Error at line {}: The origin {} is outside the valid area.".format(line_num, orig_key))

        # Check the value type
        if type(v) != dict:
            error("Error at line {}: The value for key `{}` must be a dict.".format(line_num, orig_key))

        # Emptiness check
        if len(v) == 0:
            error("Error at line {}: No destinations for origin {}.".format(line_num, orig_key))

        # Inner loop
        for dest_key, weight_str in v.items():
            # Check the key
            if not re.match(r"^-?\d+_-?\d+$", dest_key):
                error("Error at line {}: Destination key `{}` under origin `{}` has an invalid format.".format(line_num, dest_key, orig_key))

            # Check the lat-lon
            lat_str, lon_str = dest_key.split("_")
            lat = int(lat_str)
            lon = int(lon_str)
            if not in_valid_area(lat, lon):
                error("Error at line {}: Destination {} under origin {} is outside the valid area.".format(line_num, dest_key, orig_key))
            
            # Check the value
            try:
                weight = float(weight_str)
            except:
                error("Error at line {}: The value {} for edge {} -> {} is not numeric.".format(line_num, weight_str, orig_key, dest_key))

            # Check the sign
            if weight < 0:
                error("Error at line {}: The value {} for edge {} -> {} is negative.".format(line_num, weight_str, orig_key, dest_key))

    return od_matrix

def main():
    argv = sys.argv

    # Check the file argument
    if len(argv) != 2:
        error("Usage: python humob2026_validator.py <submission_file_path>")

    fpath = argv[1]

    # Load the file
    lines = list()
    try:
        with open(fpath) as f:
            for l in f:
                l = l.rstrip()
                lines.append(l)
    except:
        error("File could not be read.")
    #print("The number of lines: {}.".format(len(lines)))

    # Emptiness check
    if len(lines) == 0 or (len(lines) == 1 and len(lines[0]) == 0):
        error("The file {} is empty.".format(fpath))

    # Parse the file
    subm_dict = dict()
    seen_date_set = set()
    for line_num, l in enumerate(lines):
        tpl = l.split("\t")
        # Must be two column
        if len(tpl) != 2:
            error("Error at line {}: Expected 2 tab-delimited columns.".format(line_num))

        date_str, od_matrix_str = tpl

        # Must be 8-digit number
        if not re.match(r"^\d{8}$", date_str):
            error("Error at line {}: The date must be in the format YYYYMMDD.".format(line_num))

        # Must be within the test period
        if not (date_str >= "20240201" and date_str <= "20240331"):
            error("Error at line {}: The date must be within the evaluation period 20240201 - 20240331.".format(line_num))

        # Duplication check for date
        if date_str in seen_date_set:
            error("Error at line {}: The date is duplicated.".format(line_num))
        seen_date_set.add(date_str)

        if date_str not in date_str_set:
            # No need to check the od_matrix for NA or other unknown dates
            continue

        od_matrix = parse_od_matrix(line_num, od_matrix_str)

        subm_dict[date_str] = od_matrix

    # Must cover the necessary dates
    subm_date_str_set = set(subm_dict.keys())
    date_str_set_diff = date_str_set - subm_date_str_set
    if len(date_str_set_diff) > 0:
        diff_list = sorted(list(date_str_set_diff))
        error("The submission is missing required dates: {}".format(diff_list))

    # ... Now everything should be okay
    print("Validation passed!")

if __name__ == "__main__":
    main()
