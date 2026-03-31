# Bias Hunter

**Bias Hunter** is an automated poll-market overreaction detector. It combines static polling history with real-time prediction market odds from Kalshi to identify market inefficiencies driven by sentiment, fear, or herd behavior.

## Technologies Used
This system operates via an end-to-end AI pipeline integrating three specific architectures:
1. **RAG (Retrieval-Augmented Generation):** Establishes an empirically grounded "fair value" baseline derived from historical polling aggregates.
2. **Autonomous Tool Use (Agent):** Connects to the Kalshi API, dynamically retrieving live pricing and volume data.
3. **Automated LLM DataOps (Labeler & Judge):** Evaluates the discrepancy. An LLM labeler annotates the event with sentiment/risk flags, while an LLM Judge interprets the statistical significance and provides a final structural recommendation.

## Team
- Sydney Hinckley
- John Herbold
- Cody Gill

## Installation

```bash
pip install -r requirements.txt
```

Set your Azure OpenAI key via a `.env` file or environment variable:
```bash
export AZURE_API_KEY="your-key-here"
```

## Usage

Provide a Kalshi market ticker and its matching Race ID:

```bash
python main.py --ticker KXSENATETXR-26-KP --race "2026_REP_NOM _TX-SENATE"
```

Place your historical polling data in the `polls.csv` file (a sample is provided).
