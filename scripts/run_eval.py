import argparse
import json
import time
from pathlib import Path

from aml_copilot import evaluation

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Retrieval and answer-quality evaluation")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM-judged part")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    results = {"retrieval": evaluation.retrieval_scores(questions, k=args.k)}
    if not args.retrieval_only:
        results["generation"] = evaluation.generation_scores(questions, k=args.k)

    print(json.dumps(results, indent=2))
    out = ROOT / "eval" / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
