"""
profile_dataset.py
--------------------
Module 1 wrap-up: profiles the parsed AGC dataset (output of agc_parser.py)
and prints/saves a summary report. Use this after parsing the full
Comanche055 + Luminary099 source trees to sanity-check the dataset before
moving on to Module 2 (embeddings).

Usage:
    python3 src/ingestion/profile_dataset.py data/full_parsed/luminary.json data/full_parsed/comanche.json --out data/parsed/dataset_profile.json
"""

import json
import argparse
import statistics
from collections import Counter


def load_parsed_files(paths):
    all_files = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            all_files.extend(json.load(f))
    return all_files


def profile(all_files):
    total_files = len(all_files)
    routine_sizes = []  # instructions per routine
    opcode_counter = Counter()
    routines_per_file = []
    all_routines = []

    for file_data in all_files:
        routines = file_data.get("routines", [])
        routines_per_file.append(len(routines))
        for r in routines:
            n_instr = len(r["instructions"])
            routine_sizes.append(n_instr)
            all_routines.append({
                "file": file_data["filename"],
                "routine": r["name"],
                "n_instructions": n_instr,
                "n_references": len(r.get("references", [])),
            })
            for instr in r["instructions"]:
                if instr.get("opcode"):
                    opcode_counter[instr["opcode"]] += 1

    total_routines = len(routine_sizes)
    total_instructions = sum(routine_sizes)

    # Largest routines (candidates for "high complexity" — useful preview for Module 5)
    largest = sorted(all_routines, key=lambda r: r["n_instructions"], reverse=True)[:15]

    # Most-referenced opcodes (useful preview for Module 4's graph)
    top_opcodes = opcode_counter.most_common(20)

    report = {
        "total_files_parsed": total_files,
        "total_routines": total_routines,
        "total_instructions": total_instructions,
        "avg_routines_per_file": round(statistics.mean(routines_per_file), 2) if routines_per_file else 0,
        "avg_instructions_per_routine": round(statistics.mean(routine_sizes), 2) if routine_sizes else 0,
        "median_instructions_per_routine": statistics.median(routine_sizes) if routine_sizes else 0,
        "largest_routines": largest,
        "top_20_opcodes": top_opcodes,
    }
    return report


def print_summary(report):
    print("=" * 60)
    print("AGC DATASET PROFILE — Module 1 Summary")
    print("=" * 60)
    print(f"Files parsed:                 {report['total_files_parsed']}")
    print(f"Total routines:                {report['total_routines']}")
    print(f"Total instructions:            {report['total_instructions']}")
    print(f"Avg routines / file:           {report['avg_routines_per_file']}")
    print(f"Avg instructions / routine:    {report['avg_instructions_per_routine']}")
    print(f"Median instructions / routine: {report['median_instructions_per_routine']}")
    print()
    print("Top 10 largest routines (complexity preview for Module 5):")
    for r in report["largest_routines"][:10]:
        print(f"  {r['routine']:<20} {r['n_instructions']:>4} instructions   ({r['file']})")
    print()
    print("Top 10 most-used opcodes:")
    for op, count in report["top_20_opcodes"][:10]:
        print(f"  {op:<12} {count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Profile the parsed AGC dataset.")
    parser.add_argument("inputs", nargs="+", help="One or more parsed JSON files from agc_parser.py")
    parser.add_argument("--out", default=None, help="Optional path to save the full report as JSON")
    args = parser.parse_args()

    all_files = load_parsed_files(args.inputs)
    report = profile(all_files)
    print_summary(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {args.out}")


if __name__ == "__main__":
    main()