# eval/smoke_test.py
# Quick check that RAGAS is installed and can talk to OpenAI

import asyncio
import os
from dotenv import load_dotenv
from ragas import SingleTurnSample
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from openai import AsyncOpenAI

load_dotenv()

async def main():
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm = llm_factory("gpt-3.5-turbo", client=client)

    scorer = Faithfulness(llm=llm)

    result = await scorer.ascore(
        user_input="What is NVIDIA's fiscal year 2025 revenue?",
        response="NVIDIA reported total revenue of $130.5 billion for fiscal year 2025.",
        retrieved_contexts=[
            "NVIDIA Corporation reported record revenue of $130.5 billion for fiscal year 2025, "
            "compared to $60.9 billion in fiscal year 2024, an increase of 114%."
        ]
    )

    print(f"✅ RAGAS is working! Faithfulness score: {result.value:.3f}")
    print("   (Expected: close to 1.0 — the answer is fully supported by context)")

asyncio.run(main())