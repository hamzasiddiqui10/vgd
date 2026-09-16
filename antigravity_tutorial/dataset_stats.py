"""Dataset summary statistics calculation pipeline.

This script parses a JSON-encoded list of integers from the command line,
computes basic summary statistics (mean, median, sample variance), and
logs the results to a designated log file.
"""

import argparse
import json
import logging
import statistics
from typing import Dict, List


def setup_logger(log_file: str = "stats.log") -> logging.Logger:
    """Configures a file logger with standardized formatting.

    Args:
        log_file: Target path for the output log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("dataset_stats")
    logger.setLevel(logging.INFO)

    # Prevent handler duplication if called repeatedly
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    """Configures and parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing 'data'
            and optional 'log_file'.
    """
    parser = argparse.ArgumentParser(
        description="Compute summary statistics from a JSON-encoded integer array."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help='JSON string representing a list of integers, e.g., "[1, 2, 3, 4]"',
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="stats.log",
        help="Path to output log file (default: stats.log)",
    )
    return parser.parse_args()


def parse_json_integers(raw_json: str) -> List[int]:
    """Deserializes and validates a JSON string as a list of integers.

    Args:
        raw_json: Serialized JSON string to deserialize.

    Returns:
        List[int]: Validated list of integer values.

    Raises:
        ValueError: If JSON syntax is invalid or elements are non-integers.
    """
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON input: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}.")

    # Guard: Python's bool is a subclass of int, ensure strictly integers
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in parsed):
        raise ValueError("All elements in the JSON array must be integers.")

    return parsed


def compute_stats(data: List[int]) -> Dict[str, float]:
    """Computes mean, median, and sample variance for a list of integers.

    Args:
        data: List of integers to analyze. Must contain at least 2 elements
            for sample variance.

    Returns:
        Dict[str, float]: Dictionary with keys 'mean', 'median', and 'variance'.

    Raises:
        ValueError: If data contains fewer than 2 elements.
    """
    if not data or len(data) < 2:
        raise ValueError(
            "At least 2 data points are required to compute sample variance."
        )

    return {
        "mean": float(statistics.mean(data)),
        "median": float(statistics.median(data)),
        "variance": float(statistics.variance(data)),
    }


def main() -> None:
    """Coordinates CLI parsing, validation, computation, and logging."""
    args = parse_args()
    logger = setup_logger(args.log_file)

    try:
        dataset = parse_json_integers(args.data)
        stats = compute_stats(dataset)

        log_message = (
            f"Dataset size: {len(dataset)} | "
            f"Mean: {stats['mean']:.4f} | "
            f"Median: {stats['median']:.4f} | "
            f"Variance: {stats['variance']:.4f}"
        )
        logger.info(log_message)

    except (ValueError, NotImplementedError) as err:
        logger.error("Execution failed: %s", err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
