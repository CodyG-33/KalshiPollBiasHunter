import os
import argparse
from src.pipeline import BiasHunterWithDataOps

def main():
    parser = argparse.ArgumentParser(description="Run Bias Hunter Analysis")
    parser.add_argument("--ticker", type=str, required=True, help="Kalshi Market Ticker (e.g. KXSENATETXR-26-KP)")
    parser.add_argument("--race", type=str, required=True, help="Race ID for Polling matching (e.g. 2026_REP_NOM _TX-SENATE)")
    parser.add_argument("--polls", type=str, default="polls.csv", help="Path to polls CSV file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.polls):
        print(f"Error: Could not find polls.csv at '{args.polls}'")
        return

    print("Initializing Bias Hunter Pipeline...")
    hunter = BiasHunterWithDataOps(args.polls)
    
    print(f"Running analysis for Ticker: {args.ticker} and Race: {args.race}...")
    try:
        results = hunter.full_analysis_with_dataops(
            market_ticker=args.ticker,
            race_id=args.race
        )
        report = hunter.generate_report(results)
        print(report)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
