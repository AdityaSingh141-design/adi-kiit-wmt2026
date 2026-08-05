"""
=============================================================================
Script 04c — Merge Zero-Shot Competence into Scored Data
=============================================================================
Takes the zero-shot scored file (rewards of the BASE model output) and adds
a `zeroshot_competence` column to the main rl_scored.csv, aligned by row.

Produces: rl_scored_zscomp.csv  (what CGPO-zs reads)

Usage:
    python3 04c_merge_competence.py --config ../data2/config_v7.yaml
=============================================================================
"""
import os, sys, ast, argparse, yaml
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    main_path = cfg["scored_file"]
    zs_path   = cfg["scored_file"].replace(
        "rl_scored.csv", "rl_zeroshot_scored.csv")

    main_df = pd.read_csv(main_path)
    zs_df   = pd.read_csv(zs_path)

    for col in ["rewards"]:
        if isinstance(zs_df[col].iloc[0], str):
            zs_df[col] = zs_df[col].apply(ast.literal_eval)

    # Zero-shot competence = reward of the (single) zero-shot output.
    # candidates were [zs, zs] so both rewards equal; take index 0.
    zs_comp = [float(r[0]) for r in zs_df["rewards"]]

    if len(zs_comp) != len(main_df):
        print(f"ERROR: length mismatch zs={len(zs_comp)} main={len(main_df)}")
        sys.exit(1)

    # Sanity: src alignment
    mism = sum(1 for a, b in zip(main_df["src"].astype(str),
                                  zs_df["src"].astype(str)) if a != b)
    if mism > 0:
        print(f"WARNING: {mism} src mismatches between files — alignment off!")
    else:
        print("Src alignment verified (0 mismatches).")

    main_df["zeroshot_competence"] = zs_comp

    comp = np.array(zs_comp)
    print(f"Zero-shot competence: mean={comp.mean():.3f} "
          f"std={comp.std():.3f} min={comp.min():.3f} max={comp.max():.3f}")

    out = main_path.replace("rl_scored.csv", "rl_scored_zscomp.csv")
    main_df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()