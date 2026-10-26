"""
One-command entry point.

    python run.py              # fetch data -> train -> backtest -> schedule jobs.json
    python run.py schedule     # just schedule jobs.json with the latest forecast
    python run.py --help
"""
from carbon_scheduler.pipeline import main

if __name__ == "__main__":
    main()
