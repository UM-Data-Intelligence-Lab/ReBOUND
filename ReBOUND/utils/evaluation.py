import ast
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def target_dates():
    start = date(2024, 2, 1)
    dates = [int((start + timedelta(days=offset)).strftime("%Y%m%d")) for offset in range(60)]
    return [day for day in dates if day not in (20240202, 20240305)]


def grid_name(node):
    row, column = divmod(int(node), 41)
    return f"{row + 35}_{column + 30}"


def export_submission(path, prediction):
    dates = prediction["prediction_dates"].tolist()
    nodes, edges = prediction["active_nodes"], prediction["edges"]
    count = 0
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for day in target_dates():
            index = dates.index(day)
            record = {}
            for node, value in zip(nodes, prediction["candidate_diag"][index]):
                key = grid_name(node)
                record.setdefault(key, {})[key] = float(value)
                count += 1
            for edge, value in zip(edges, prediction["candidate_off"][index]):
                origin, destination = divmod(int(edge), 1476)
                record.setdefault(grid_name(origin), {})[grid_name(destination)] = float(value)
                count += 1
            stream.write(f"{day}\t{record!r}\n")
    return count


def verify_roundtrip(path, prediction):
    dates = prediction["prediction_dates"].tolist()
    seen, count = [], 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            date_text, payload = line.rstrip("\n").split("\t")
            day = int(date_text)
            seen.append(day)
            record = ast.literal_eval(payload)
            index = dates.index(day)
            expected = {}
            for node, value in zip(prediction["active_nodes"], prediction["candidate_diag"][index]):
                key = grid_name(node)
                expected[key, key] = float(value)
            for edge, value in zip(prediction["edges"], prediction["candidate_off"][index]):
                origin, destination = divmod(int(edge), 1476)
                expected[grid_name(origin), grid_name(destination)] = float(value)
            actual = {(origin, destination): value for origin, destinations in record.items()
                      for destination, value in destinations.items()}
            if actual != expected:
                raise ValueError(f"Submission round-trip mismatch on {day}.")
            count += len(actual)
    if seen != target_dates():
        raise ValueError("Submission has incorrect dates or ordering.")
    return count


def score_validation(prediction, observations, csv_path):
    if not np.array_equal(prediction["prediction_dates"], observations["dates"]):
        raise ValueError("Validation dates differ from the frozen Val23 protocol.")
    # Background sums include the truth outside the sparse prediction support.
    diagonal = np.sqrt((np.square(prediction["candidate_diag"] - observations["truth_diag"]).sum(1)
                        + observations["diag_background"]) / 1476) / 26.57
    off_diagonal = np.sqrt((np.square(prediction["candidate_off"] - observations["truth_off"]).sum(1)
                            + observations["off_background"]) / 2177100) / .0176
    results = {}
    for name, mask in [("Val23", np.ones(23, bool)), ("January9", observations["dates"] < 20240201),
                       ("April14", observations["dates"] >= 20240401)]:
        diag, off = float(diagonal[mask].mean()), float(off_diagonal[mask].mean())
        results[name] = dict(diag=diag, offdiag=off, combined=.5 * (diag + off))
    with Path(csv_path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "diagonal_nrmse", "off_diagonal_nrmse", "combined_nrmse"])
        for day, diag, off in zip(observations["dates"], diagonal, off_diagonal):
            writer.writerow([int(day), diag, off, .5 * (diag + off)])
    return results
