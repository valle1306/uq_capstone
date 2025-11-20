"""
Compute conformal prediction sets and coverage for multiple UQ methods.

Expect prediction inputs in a directory structure like:
  runs/classification/<method>/predictions.npz
where the NPZ contains:
  - 'probs' : numpy array shape (N, C) of softmax probabilities
  - 'y': numpy array shape (N,) of integer labels
  - optionally 'split': array or dict with 'cal', 'test' indices

Outputs per-method JSON in runs/classification/<method>/conformal_results.json

Usage:
  python src/run_conformal_all.py --methods swag_adam swag_sgd ensemble dropout --cal_size 1044 --alpha 0.1

This script implements split conformal for multiclass using max-prob nonconformity (1 - p_true).
"""

import argparse
import json
import numpy as np
import os
from pathlib import Path


def compute_nonconformity(probs, y):
    # nonconformity = 1 - probability assigned to true class
    p_true = probs[np.arange(len(y)), y]
    return 1.0 - p_true


def split_conformal(probs, y, cal_idx, test_idx, alpha=0.1):
    # calibration nonconformity
    cal_nc = compute_nonconformity(probs[cal_idx], y[cal_idx])
    q = np.quantile(cal_nc, 1 - alpha, interpolation='higher') if hasattr(np, 'quantile') else np.quantile(cal_nc, 1 - alpha)

    # construct prediction sets on test
    test_nc = compute_nonconformity(probs[test_idx], y[test_idx])
    # a point is in the set if its nonconformity <= q
    in_set = test_nc <= q
    # because we're using max-prob nonconformity, set size is 1 for those where max prob > 1 - q
    # However, to be general, compute set sizes by counting classes with prob >= 1 - q
    threshold = 1.0 - q
    set_sizes = (probs[test_idx] >= threshold).sum(axis=1)
    coverage = float(np.mean(in_set))
    avg_set_size = float(np.mean(set_sizes))
    singleton_accuracy = None
    if np.all(set_sizes == 1):
        # singleton accuracy equals standard accuracy
        preds = probs[test_idx].argmax(axis=1)
        singleton_accuracy = float((preds == y[test_idx]).mean())

    return {
        'alpha': alpha,
        'quantile_q': float(q),
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'singleton_accuracy': singleton_accuracy
    }


def load_predictions(pred_path):
    data = np.load(pred_path)
    probs = data['probs']
    y = data['y']
    # support splits
    if 'cal_idx' in data and 'test_idx' in data:
        cal_idx = data['cal_idx']
        test_idx = data['test_idx']
    else:
        # default split: first N_cal as calibration, next N_test as test, if sizes provided
        # fall back: use half for cal, half for test
        N = len(y)
        half = N // 2
        cal_idx = np.arange(0, half)
        test_idx = np.arange(half, N)
    return probs, y, cal_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', required=True, help='List of methods subfolders to evaluate')
    parser.add_argument('--runs_dir', default='runs/classification', help='Base runs directory')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--cal_size', type=int, default=None)
    args = parser.parse_args()

    base = Path(args.runs_dir)
    results = {}
    for m in args.methods:
        mdir = base / m
        pred_path_npz = mdir / 'predictions.npz'
        if not pred_path_npz.exists():
            print(f"Predictions not found for {m} at {pred_path_npz}. Skipping.")
            continue
        probs, y, cal_idx, test_idx = load_predictions(str(pred_path_npz))
        # if user provided cal_size, use first cal_size as calibration
        if args.cal_size is not None:
            cal_idx = np.arange(0, min(args.cal_size, len(y)))
            test_idx = np.arange(min(args.cal_size, len(y)), len(y))

        res = split_conformal(probs, y, cal_idx, test_idx, alpha=args.alpha)
        out_path = mdir / 'conformal_results.json'
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=2)
        results[m] = res
        print(f"Wrote {out_path}")

    # summary
    summary_path = base / 'conformal_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {summary_path}")

if __name__ == '__main__':
    main()
