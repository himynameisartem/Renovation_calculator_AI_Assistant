from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка цепочки CV JSON -> RAG -> предварительная смета"
    )
    parser.add_argument("--cv-json", required=True, help="Путь к room_analysis.json")
    parser.add_argument("--room-type", default="other", help="living, kitchen, bathroom, hallway или other")
    parser.add_argument("--room-name", default="", help="Название помещения")
    parser.add_argument("--area", type=float, required=True, help="Площадь пола, м²")
    parser.add_argument("--height", type=float, default=2.7, help="Высота помещения, м")
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant URL (по умолчанию Docker на http://localhost:6333)",
    )
    parser.add_argument("--debug", action="store_true", help="Показать поисковый запрос и найденные данные")
    args = parser.parse_args()

    os.environ["QDRANT_URL"] = args.qdrant_url
    if args.qdrant_url.startswith(("http://localhost", "http://127.0.0.1")):
        os.environ["QDRANT_API_KEY"] = ""

    from app.photo_estimate import PhotoEstimateService, load_cv_json

    service = PhotoEstimateService()
    result = service.answer(
        cv_result=load_cv_json(args.cv_json),
        room_type=args.room_type,
        room_name=args.room_name,
        area_m2=args.area,
        height_m=args.height,
    )
    print(result.answer)

    if args.debug:
        print("\n=== ПОИСКОВЫЙ ЗАПРОС ===")
        print(result.search_query)
        print("\n=== ИСПОЛЬЗОВАННЫЕ ДАННЫЕ ===")
        print(result.context)
        print(f"\nИсточников: {len(result.points)}")


if __name__ == "__main__":
    main()
