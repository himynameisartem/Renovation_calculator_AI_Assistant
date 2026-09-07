from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


AREA_RE = re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:м\s*[²2]|кв\.?\s*м)", re.IGNORECASE)
STAGE_DISPLAY_NAMES = {
    "demolition": "демонтажные работы",
    "preparation": "подготовительные работы",
    "rough_finish": "черновые отделочные работы",
    "electrical": "электромонтажные работы",
    "plumbing": "сантехнические работы",
    "heating": "работы по отоплению",
    "finish": "чистовые отделочные работы",
}
WORK_AREA_DISPLAY_NAMES = {
    "walls": "стены",
    "floor": "полы",
    "ceiling": "потолки",
    "doors_windows": "двери и окна",
    "slopes": "откосы",
    "plumbing": "сантехника",
    "electrical": "электрика",
    "other": "прочие работы",
}
DEFAULT_BENCHMARKS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cleaned"
    / "estimate_benchmarks.json"
)


@dataclass(slots=True)
class EstimateRange:
    stage_code: str
    target_area_m2: float
    minimum: float
    maximum: float
    examples_count: int
    work_area_codes: tuple[str, ...]
    work_items: tuple[str, ...]

    def as_context(self) -> str:
        work_areas = ", ".join(
            WORK_AREA_DISPLAY_NAMES.get(code, code)
            for code in self.work_area_codes
        )
        work_items = "; ".join(self.work_items)
        return (
            "Расчёт по сопоставимым реальным объектам:\n"
            f"- площадь пользователя: {self.target_area_m2:.2f} м²\n"
            f"- минимальный ориентир: {self.minimum:.2f} ₽\n"
            f"- максимальный ориентир: {self.maximum:.2f} ₽\n"
            f"- количество сопоставимых объектов: {self.examples_count}\n"
            f"- разделы работ, встречавшиеся в расчётах этапа: {work_areas}\n"
            f"- конкретные позиции, встречавшиеся в расчётах этапа: {work_items}\n"
            "Используй эту вилку как итог расчёта и не заменяй её другим тарифом. "
            "При общем вопросе о составе сначала перечисли разделы. Конкретные позиции "
            "приводи только как примеры и не называй их исчерпывающим перечнем для любого объекта."
        )


@dataclass(slots=True)
class WholeRenovationRange:
    target_area_m2: float
    current_minimum: float
    examples_minimum: float
    examples_maximum: float
    examples_count: int
    included_stage_codes: tuple[str, ...]

    def as_context(self) -> str:
        included_stages = ", ".join(
            STAGE_DISPLAY_NAMES.get(code, code)
            for code in self.included_stage_codes
        )
        return (
            "Два разных ориентира полной стоимости ремонта:\n"
            f"- площадь пользователя: {self.target_area_m2:.2f} м²\n"
            f"- минимальный порог по базовому тарифу: {self.current_minimum:.2f} ₽\n"
            f"- реалистичный предварительный бюджет по сопоставимым полным объектам: "
            f"от {self.examples_minimum:.2f} ₽ "
            f"до {self.examples_maximum:.2f} ₽\n"
            f"- количество сопоставимых объектов: {self.examples_count}\n"
            f"- этапы, входящие в бюджет по сопоставимым объектам: {included_stages}\n"
            "При вопросе о составе перечисляй только входящие этапы. "
            "Не перечисляй отсутствующие или неподтверждённые позиции, если пользователь "
            "не спросил о конкретной позиции прямо.\n"
            "Не объединяй эти показатели в одну вилку. Сначала отдельно назови минимальный "
            "порог по базовому тарифу, затем отдельно реалистичный предварительный бюджет. "
            "Не пиши: «стоимость от базового порога, но варьируется от другой суммы»."
        )


class EstimateCalculator:
    def __init__(self, benchmarks_path: str | Path = DEFAULT_BENCHMARKS_PATH) -> None:
        self.benchmarks_path = Path(benchmarks_path)
        self.benchmarks = self._load_benchmarks()

    def calculate(self, stage_code: str, target_area_m2: float) -> EstimateRange | None:
        if not stage_code or stage_code == "unknown" or target_area_m2 <= 0:
            return None

        comparable_stages = [
            stage
            for benchmark in self.benchmarks
            for stage in benchmark.get("stages", [])
            if stage.get("stage_code") == stage_code
            and stage.get("scope") == "full_object"
            and float(stage.get("cost_per_m2") or 0) > 0
        ]
        rates = [float(stage["cost_per_m2"]) for stage in comparable_stages]
        if not rates:
            return None

        found_work_area_codes = {
            str(code)
            for stage in comparable_stages
            for code in stage.get("work_area_codes", [])
            if code
        }
        work_area_codes = tuple(
            code for code in WORK_AREA_DISPLAY_NAMES
            if code in found_work_area_codes
        )
        work_items = tuple(sorted({
            str(item)
            for stage in comparable_stages
            for item in stage.get("work_items", [])
            if item
        }))

        return EstimateRange(
            stage_code=stage_code,
            target_area_m2=target_area_m2,
            minimum=round(min(rates) * target_area_m2, 2),
            maximum=round(max(rates) * target_area_m2, 2),
            examples_count=len(rates),
            work_area_codes=work_area_codes,
            work_items=work_items,
        )

    def calculate_whole_renovation(
        self,
        target_area_m2: float,
        current_minimum_per_m2: float = 15000.0,
    ) -> WholeRenovationRange | None:
        if target_area_m2 <= 0:
            return None

        comparable_benchmarks = [
            benchmark
            for benchmark in self.benchmarks
            if benchmark.get("object_scope") == "full_object"
            and float(benchmark.get("object_cost_per_m2") or 0) > 0
        ]
        rates = [
            float(benchmark["object_cost_per_m2"])
            for benchmark in comparable_benchmarks
        ]
        if not rates:
            return None

        stage_sets = [
            {
                str(stage.get("stage_code"))
                for stage in benchmark.get("stages", [])
                if stage.get("stage_code")
            }
            for benchmark in comparable_benchmarks
        ]
        common_stage_codes = set.intersection(*stage_sets) if stage_sets else set()
        included_stage_codes = tuple(
            code for code in STAGE_DISPLAY_NAMES
            if code in common_stage_codes
        )

        return WholeRenovationRange(
            target_area_m2=target_area_m2,
            current_minimum=round(current_minimum_per_m2 * target_area_m2, 2),
            examples_minimum=round(min(rates) * target_area_m2, 2),
            examples_maximum=round(max(rates) * target_area_m2, 2),
            examples_count=len(rates),
            included_stage_codes=included_stage_codes,
        )

    def _load_benchmarks(self) -> list[dict]:
        if not self.benchmarks_path.exists():
            return []
        data = json.loads(self.benchmarks_path.read_text(encoding="utf-8"))
        return list(data.get("benchmarks", []))


def extract_area_m2(text: str) -> float | None:
    match = AREA_RE.search(text or "")
    if not match:
        return None
    return float(match.group(1).replace(",", "."))
