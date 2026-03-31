import json
import re
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm_setup import build_llm

LABELING_PROMPT = ChatPromptTemplate.from_template(
    """

DATAOPS: LABELING TASK

You are a financial data labeling assistant. Your job is to
analyze market/polling data and assign structured labels.

DATA TO LABEL:
{data_point}

LABELING INSTRUCTIONS:
Analyze the data and provide labels as JSON with these exact fields:

1. sentiment: Assess the overall market sentiment
   - "bullish" = market/polls suggest positive outcome
   - "neutral" = no clear direction
   - "bearish" = market/polls suggest negative outcome

2. time_horizon: How soon is the event?
   - "immediate" = within days
   - "near_term" = within weeks
   - "long_term" = months or more

3. confidence_level: How reliable is the data?
   - "high" = large sample, consistent data
   - "medium" = moderate confidence
   - "low" = small sample, inconsistent, or stale data

4. risk_flags: List any concerns (choose all that apply):
   - "polling_bias" = polls may be systematically biased
   - "low_volume" = insufficient market liquidity
   - "high_volatility" = prices are unstable
   - "stale_data" = data may be outdated
   - "none" = no significant risks identified

5. reasoning: One sentence explaining your labels

Return ONLY valid JSON, no other text.
"""
)

class LLMDataLabeler:
    """
    TECHNIQUE 3a: DATAOPS LABELER
    """

    def __init__(self):
        self.llm = build_llm()
        self.parser = StrOutputParser()

    def label(self, data_point: Dict) -> Dict:
        chain = LABELING_PROMPT | self.llm | self.parser

        result = chain.invoke({
            "data_point": json.dumps(data_point, indent=2, default=str)
        })

        try:
            cleaned = re.sub(r'^```json\s*|\s*```$', '', result.strip())
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_response": result, "parse_error": True}


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """

You are an expert judge evaluating prediction market efficiency.
Your job is to assess whether the market price reflects fair value.

MARKET DATA TO EVALUATE:
- Market Probability: {market_prob}
- Poll Probability: {poll_prob}
- Discrepancy: {discrepancy}
- Trading Volume: {volume}

EVALUATION CRITERIA:
1. Is the discrepancy significant enough to suggest mispricing?
2. Does the volume indicate sufficient liquidity?
3. Are there signs of market inefficiency or bias?
4. How reliable is the polling data as a baseline?

Provide your judgment as JSON with these fields:

1. efficiency_score: float 0-10 (10 = perfectly efficient)

2. verdict: one of:
   - "efficient"
   - "slightly_inefficient"
   - "inefficient"
   - "highly_inefficient"

3. mispricing_direction: one of:
   - "overpriced" (market > fair)
   - "underpriced" (market < fair)
   - "fair"
   - "uncertain"

4. confidence: float 0-1

5. key_insights: list of 2-3 short observations

6. recommendation: one sentence suggestion

Return ONLY valid JSON.
"""
)

class LLMJudge:
    """
    TECHNIQUE 3b: DATAOPS JUDGE
    """

    def __init__(self):
        self.llm = build_llm()
        self.parser = StrOutputParser()

    def judge_efficiency(self, market_prob: float, poll_prob: float, volume: int = 0) -> Dict:
        discrepancy = None
        if market_prob is not None and poll_prob is not None:
            discrepancy = market_prob - poll_prob

        chain = JUDGE_PROMPT | self.llm | self.parser

        result = chain.invoke({
            "market_prob": f"{market_prob:.2%}" if market_prob is not None else "N/A",
            "poll_prob": f"{poll_prob:.2%}" if poll_prob is not None else "N/A",
            "discrepancy": f"{discrepancy:.2%}" if discrepancy is not None else "N/A",
            "volume": volume,
        })

        try:
            cleaned = re.sub(r'^```json\s*|\s*```$', '', result.strip())
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_response": result, "parse_error": True}
