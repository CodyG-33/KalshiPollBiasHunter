from datetime import datetime
from typing import Dict, Optional
import textwrap

from src.rag import PollingRAG
from src.kalshi import KalshiMarketDataTool
from src.dataops import LLMDataLabeler, LLMJudge

class BiasHunterWithDataOps:
    """
    FULL INTEGRATED SYSTEM: RAG + AGENT + DATAOPS
    """

    def __init__(self, polling_csv_path: str):
        self.poll_rag = PollingRAG(csv_path=polling_csv_path)
        self.kalshi = KalshiMarketDataTool()
        self.labeler = LLMDataLabeler()
        self.judge = LLMJudge()

    def full_analysis_with_dataops(
        self,
        market_ticker: str,
        race_id: str,
        as_of_date: Optional[str] = None,
        threshold: float = 0.10,
    ) -> Dict:

        results = {
            "market_ticker": market_ticker,
            "race_id": race_id,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. RAG
        poll_prob, evidence = self.poll_rag.compute_poll_implied_prob(race_id=race_id, as_of_date=as_of_date)
        results["rag_results"] = {
            "technique": "RAG",
            "poll_implied_prob": poll_prob,
            "num_polls_retrieved": len(evidence),
            "polling_evidence": evidence,
        }

        # 2. AGENT
        snapshot = self.kalshi.get_market_price_snapshot(market_ticker)
        if snapshot is None:
            results["agent_results"] = {"error": f"No market found for {market_ticker}"}
            market_prob = None
            volume = 0
        else:
            market_prob = snapshot["yes_price_pct"] / 100.0 if snapshot.get("yes_price_pct") is not None else None
            volume = snapshot.get("volume", 0)
            results["agent_results"] = {
                "title": snapshot.get("title"),
                "market_prob": market_prob,
                "volume": volume,
            }

        # 3. DATAOPS
        combined_data = {
            "market_prob": market_prob,
            "poll_prob": poll_prob,
            "volume": volume,
            "num_polls": len(evidence),
            "discrepancy": (market_prob - poll_prob) if (market_prob is not None and poll_prob is not None) else None,
        }
        labels = self.labeler.label(combined_data)
        results["dataops_labels"] = {"labels": labels}

        if market_prob is not None:
            judgment = self.judge.judge_efficiency(market_prob=market_prob, poll_prob=poll_prob, volume=volume)
            results["dataops_judgment"] = {"judgment": judgment}
        else:
            results["dataops_judgment"] = {"error": "Missing market probability"}

        # VERDICT
        if market_prob and poll_prob:
            discrepancy = market_prob - poll_prob
            if abs(discrepancy) < threshold:
                verdict = "FAIRLY_PRICED"
            elif discrepancy > 0:
                verdict = "OVERPRICED"
            else:
                verdict = "UNDERPRICED"

            results["final_verdict"] = verdict
            results["discrepancy"] = discrepancy
            results["discrepancy_pct"] = f"{discrepancy:.2%}"
        else:
            results["final_verdict"] = "INSUFFICIENT_DATA"

        return results

    def generate_report(self, analysis: Dict) -> str:
        report = []
        report.append("=" * 70)
        report.append("                    BIAS HUNTER ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"\\nMarket Ticker: {analysis.get('market_ticker')}")
        report.append(f"Race ID: {analysis.get('race_id')}")

        rag = analysis.get("rag_results", {})
        report.append(f"\\nPoll-Implied Probability: {rag.get('poll_implied_prob')}")
        
        agent = analysis.get("agent_results", {})
        report.append(f"Market Probability: {agent.get('market_prob')}")

        labels = analysis.get("dataops_labels", {}).get("labels", {})
        report.append(f"\\nSentiment: {labels.get('sentiment')}")
        report.append(textwrap.fill(labels.get('reasoning', ''), width=65, initial_indent="Reasoning: ", subsequent_indent="           "))

        judgment = analysis.get("dataops_judgment", {}).get("judgment", {})
        report.append(f"\\nVerdict: {judgment.get('verdict')}")
        report.append(f"Recommendation: {judgment.get('recommendation')}")

        report.append(f"\\n>>> FINAL VERDICT: {analysis.get('final_verdict')} <<<")
        return "\\n".join(report)
