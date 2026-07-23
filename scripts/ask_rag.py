from __future__ import annotations

import argparse

from app.rag import RenovationRAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="User question for the renovation assistant")
    parser.add_argument("--debug", action="store_true", help="Print routing metadata")
    args = parser.parse_args()

    rag = RenovationRAG()
    result = rag.answer(args.question)

    print(result.answer)

    if args.debug:
        print()
        print(f"intent: {result.intent}")
        print(f"rewritten_query: {result.rewritten_query}")
        print(f"points: {len(result.points)}")


if __name__ == "__main__":
    main()
