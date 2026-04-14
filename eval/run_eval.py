# eval/run_eval.py
# RAGAS evaluation harness — runs all 20 golden set questions and scores with 4 metrics

import sys
import os
import json
import asyncio
from datetime import datetime

# add project root to path so we can import rag_basic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,      
    ContextPrecision,     
    ContextRecall,        
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

load_dotenv()

# ── 1. load pipeline ───────────────────────────────────────────────────────────
print("Loading RAG pipeline (this may take 30-60 seconds)...")
from rag_basic import get_answer
print("Pipeline ready.\n")

# ── 2. load golden set ─────────────────────────────────────────────────────────
GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
    golden_set = json.load(f)
print(f"Loaded {len(golden_set)} questions.\n")

# ── 3. set up RAGAS metrics ────────────────────────────────────────────────────
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
llm    = llm_factory("gpt-3.5-turbo", client=client)

embeddings_model = embedding_factory("openai", model="text-embedding-3-small", client=client)

faithfulness = Faithfulness(llm=llm)
relevancy    = AnswerRelevancy(llm=llm, embeddings=embeddings_model)
precision    = ContextPrecision(llm=llm)
recall       = ContextRecall(llm=llm)

# ── 4. evaluate one question ───────────────────────────────────────────────────
async def evaluate_one(item: dict, index: int) -> dict:
    question     = item["question"]
    ground_truth = item["ground_truth"]
    document     = item["document"]

    print(f"[{index+1:02d}/20] {question[:65]}...")

    # run your pipeline
    answer, contexts = get_answer(question)

    # guard against pipeline errors 
    if not contexts:
        print(f"       ⚠ Pipeline returned no contexts, skipping scoring.")
        return {
            "question": question, "document": document,
            "answer": answer, "ground_truth": ground_truth,
            "faithfulness": None, "response_relevancy": None,
            "context_precision": None, "context_recall": None,
        }

    # score with all 4 metrics
    f_score = await faithfulness.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )

    r_score = await relevancy.ascore(
        user_input=question,
        response=answer
    )

    p_score = await precision.ascore(
        user_input=question,
        retrieved_contexts=contexts,
        reference=ground_truth
    )

    rc_score = await recall.ascore(
        user_input=question,
        retrieved_contexts=contexts,
        reference=ground_truth
    )

    result = {
        "question":           question,
        "document":           document,
        "answer":             answer,
        "ground_truth":       ground_truth,
        "faithfulness":       round(float(f_score.value)  if hasattr(f_score,  'value') else float(f_score),  3),
        "response_relevancy": round(float(r_score.value)  if hasattr(r_score,  'value') else float(r_score),  3),
        "context_precision":  round(float(p_score.value)  if hasattr(p_score,  'value') else float(p_score),  3),
        "context_recall":     round(float(rc_score.value) if hasattr(rc_score, 'value') else float(rc_score), 3),
    }

    print(f"       F={result['faithfulness']:.2f}  "
          f"RR={result['response_relevancy']:.2f}  "
          f"CP={result['context_precision']:.2f}  "
          f"CR={result['context_recall']:.2f}\n")

    return result


# ── 5. run full eval ───────────────────────────────────────────────────────────
async def run_eval():
    print("="*60)
    print("STARTING EVALUATION")
    print("="*60 + "\n")

    results = []
    for i, item in enumerate(golden_set):
        result = await evaluate_one(item, i)
        results.append(result)

    # filter out any skipped questions for averaging
    scored = [r for r in results if r["faithfulness"] is not None]

    avg = {
        "faithfulness":       round(sum(r["faithfulness"]       for r in scored) / len(scored), 3),
        "response_relevancy": round(sum(r["response_relevancy"] for r in scored) / len(scored), 3),
        "context_precision":  round(sum(r["context_precision"]  for r in scored) / len(scored), 3),
        "context_recall":     round(sum(r["context_recall"]     for r in scored) / len(scored), 3),
    }

    # ── 6. save to results/ ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(
        os.path.dirname(__file__), "results", f"eval_{timestamp}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": timestamp, "averages": avg, "results": results},
            f, indent=2, ensure_ascii=False
        )

    # ── 7. print summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<25} {'Score':>7}  {'Bar'}")
    print("-"*60)
    labels = {
        "faithfulness":       "Faithfulness",
        "response_relevancy": "Response Relevancy",
        "context_precision":  "Context Precision",
        "context_recall":     "Context Recall",
    }
    for key, label in labels.items():
        score = avg[key]
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{label:<25} {score:>6.3f}  {bar}")

    print("\n" + "-"*60)
    print(f"Questions scored:  {len(scored)}/20")
    print(f"Results saved to:  {out_path}")
    print("="*60)


asyncio.run(run_eval())