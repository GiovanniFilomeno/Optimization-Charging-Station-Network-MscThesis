#!/usr/bin/env python3
"""
EV Charging Station Network Optimization — Main Pipeline.

Orchestrates the full end-to-end pipeline:
    1. Preprocessing:    Clean raw Bundesnetzagentur data
    2. Statistics:       Generate descriptive analyses and plots
    3. Network:          Build spatial graphs for each year
    4. Analysis:         Compute topology metrics over time
    5. Prediction:       Train RF models for location prediction
    6. Optimization:     Run Genetic Algorithm for optimal placement
    7. Visualization:    Compare baseline vs. optimized networks

Usage:
    python main.py                     # Run full pipeline
    python main.py --step preprocess   # Run a single step
    python main.py --step optimize --year 2011 --stations 174

Available steps:
    preprocess, statistics, network, analysis, prediction, optimize, visualize
"""

import argparse
import logging
import sys

logger = logging.getLogger("pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="EV Charging Station Network Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        choices=[
            "preprocess", "statistics", "network",
            "analysis", "prediction", "optimize", "visualize",
        ],
        help="Run a single pipeline step (default: run all)",
    )
    parser.add_argument("--year", type=int, help="Target year for optimization")
    parser.add_argument("--stations", type=int, help="Number of new stations for GA")
    parser.add_argument(
        "--year-range", type=int, nargs=2, metavar=("START", "END"),
        help="Year range for network generation",
    )
    args = parser.parse_args()

    steps = (
        [args.step] if args.step
        else ["preprocess", "statistics", "network", "analysis", "prediction"]
    )

    for step in steps:
        logger.info("=" * 60)
        logger.info("STEP: %s", step.upper())
        logger.info("=" * 60)

        if step == "preprocess":
            from src.preprocessing import run
            run()

        elif step == "statistics":
            from src.statistics import run
            run()

        elif step == "network":
            from src.network import run
            yr_range = args.year_range or [2009, 2022]
            run(list(range(yr_range[0], yr_range[1] + 1)))

        elif step == "analysis":
            from src.analysis import run
            run()

        elif step == "prediction":
            from src.prediction import run
            run()

        elif step == "optimize":
            if not args.year or not args.stations:
                parser.error("--step optimize requires --year and --stations")
            from src.optimization import run
            run(args.year, args.stations)

        elif step == "visualize":
            from src.visualization import run
            run()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
