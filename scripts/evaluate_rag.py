from __future__ import annotations

import argparse
import csv
from pathlib import Path

from app.rag import RenovationRAG


DEFAULT_OUTPUT_PATH = Path("data/eval/rag_answers.csv")


def load_questions(path: Path) -> list[str]:
    questions: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        question = line.strip()
        if question and not question.startswith("#"):
            questions.append(question)

    return questions


def save_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "answer",
                "intent",
                "rewritten_query",
                "points_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions_path", help="Path to txt file with one question per line")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions_path)
    output_path = Path(args.output)

    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"No questions found in {questions_path}")

    rag = RenovationRAG()
    rows: list[dict] = []

    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question}")
        result = rag.answer(question)
        print(result.answer)
        print()

        rows.append(
            {
                "question": question,
                "answer": result.answer,
                "intent": result.intent,
                "rewritten_query": result.rewritten_query,
                "points_count": len(result.points),
            }
        )

    save_results(rows, output_path)
    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
