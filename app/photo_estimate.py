from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

from app.rag import RenovationRAG
from app.vector_store import QdrantVectorStore
from qdrant_client.http import models


SURFACE_KEYS = ("walls", "floor", "ceiling")

MATERIAL_RULES: dict[str, dict[str, str]] = {
    "bare_unfinished_wall": {
        "surface": "walls",
        "description": "голые стены без чистовой отделки",
        "guidance": (
            "Демонтаж настенного покрытия не нужен. Скрытую старую штукатурку не предполагать. "
            "Для дальнейшей отделки могут понадобиться грунтование, штукатурка и шпаклевание."
        ),
        "search": "подготовка голых стен грунтование штукатурка шпаклевание стен",
    },
    "exposed_drywall": {
        "surface": "walls",
        "description": "открытые листы гипсокартона",
        "guidance": (
            "Не считать гипсокартон штукатуркой. Если обшивка сохраняется, её демонтаж не нужен; "
            "могут понадобиться заделка швов, грунтование и шпаклевание. Демонтаж обшивки считать "
            "только возможным вариантом, а не обязательной работой."
        ),
        "search": "заделка швов гипсокартона грунтование шпаклевание демонтаж обшивки ГКЛ",
    },
    "unfinished_plaster_or_putty": {
        "surface": "walls",
        "description": "стены с существующей штукатуркой или шпаклёвкой",
        "guidance": (
            "Не добавлять повторно уже выполненную штукатурку или шпаклёвку. Не добавлять их снятие "
            "без видимых дефектов или отдельного требования пользователя."
        ),
        "search": "подготовленные стены существующая штукатурка шпаклевка без демонтажа",
    },
    "wallpaper": {
        "surface": "walls",
        "description": "обои на стенах",
        "guidance": (
            "Нужно снятие обоев. Многослойные или трудноотделяемые обои использовать как верхнюю "
            "границу, но не утверждать этот вариант без дополнительных признаков."
        ),
        "search": "снятие обоев снятие многослойных трудноотделяемых обоев стены",
    },
    "tile": {
        "surface": "walls",
        "description": "настенная плитка",
        "guidance": (
            "Нужен демонтаж настенной плитки. Отбивку плиточного клея считать отдельной возможной "
            "работой, потому что наличие и состояние клеевого слоя по фотографии не определяются."
        ),
        "search": "демонтаж керамической плитки со стен отбивка плиточного клея стены",
    },
    "rigid_decorative_cladding": {
        "surface": "walls",
        "description": "жёсткая декоративная облицовка стен",
        "guidance": (
            "Нужен демонтаж облицовки. Точный тип может быть панелью, декоративным кирпичом, камнем "
            "или другой жёсткой облицовкой, поэтому использовать вилку по подходящим вариантам, "
            "не называя один конкретный материал установленным фактом."
        ),
        "search": "демонтаж стеновых панелей ПВХ ДСП ГКЛ декоративного кирпича камня облицовки",
    },
    "bare_unfinished_floor": {
        "surface": "floor",
        "description": "пол без чистового покрытия",
        "guidance": (
            "Демонтаж напольного покрытия не нужен. Наличие, материал и толщина стяжки по фото не "
            "определяются, поэтому демонтаж стяжки не включать как обязательную работу."
        ),
        "search": "пол без покрытия подготовка основания стяжка без демонтажа покрытия",
    },
    "carpet": {
        "surface": "floor",
        "description": "ковровое покрытие",
        "guidance": (
            "Если это свободно лежащий ковёр, оплачиваемый демонтаж покрытия не нужен. Если это "
            "ковролин, возможен демонтаж; клеевое крепление считать отдельной неопределённостью."
        ),
        "search": "демонтаж ковролина коврового покрытия на клею",
    },
    "tile_or_stone": {
        "surface": "floor",
        "description": "плитка или камень на полу",
        "guidance": (
            "Нужен демонтаж напольной плитки или камня. Демонтаж клея, стяжки и гидроизоляции не "
            "включать как обязательный без дополнительных данных."
        ),
        "search": "демонтаж плитки пола камня плиточного клея пол",
    },
    "vinyl_or_linoleum": {
        "surface": "floor",
        "description": "линолеум или виниловое покрытие",
        "guidance": (
            "Нужен демонтаж линолеума или винилового покрытия. Вариант покрытия на клею использовать "
            "как верхнюю границу, а не как установленный факт."
        ),
        "search": "демонтаж линолеума винилового покрытия линолеума на клею",
    },
    "wood_flooring": {
        "surface": "floor",
        "description": "деревянное напольное покрытие",
        "guidance": (
            "Нужен демонтаж видимого деревянного покрытия. Точный тип может быть ламинатом, "
            "паркетной доской, штучным паркетом или дощатым полом; использовать диапазон подходящих "
            "работ. Лаги и подложку не считать видимыми по фото."
        ),
        "search": "демонтаж ламината паркетной доски штучного паркета деревянного пола",
    },
    "bare_unfinished_ceiling": {
        "surface": "ceiling",
        "description": "потолок без чистовой отделки",
        "guidance": (
            "Демонтаж потолочного покрытия не нужен. Не добавлять снятие краски, побелки или "
            "штукатурки, если модель их не обнаружила."
        ),
        "search": "подготовка голого бетонного потолка без демонтажа покрытия",
    },
    "drywall_ceiling": {
        "surface": "ceiling",
        "description": "потолок из гипсокартона",
        "guidance": (
            "При полном удалении существующей отделки возможен демонтаж подвесного потолка ГКЛ. "
            "Каркас отдельно не считать, если его демонтаж не подтверждён."
        ),
        "search": "демонтаж подвесного потолка ГКЛ гипсокартона",
    },
    "glued_light_ceiling_finish": {
        "surface": "ceiling",
        "description": "наклеенная лёгкая отделка потолка",
        "guidance": (
            "Нужно снятие наклеенной отделки: потолочных обоев или полистирольных плиток. Точный "
            "вариант выбирать как диапазон, если он не различим по фото."
        ),
        "search": "снятие потолочных обоев полистирольных плиток потолок",
    },
    "modular_or_panel_ceiling": {
        "surface": "ceiling",
        "description": "модульный, кассетный, реечный или панельный потолок",
        "guidance": (
            "Нужен демонтаж соответствующего подвесного или панельного потолка. Точный тип конструкции "
            "не утверждать, если он не определён моделью."
        ),
        "search": "демонтаж реечного кассетного подвесного потолка Армстронг панелей",
    },
    "stretch_ceiling": {
        "surface": "ceiling",
        "description": "натяжной потолок",
        "guidance": "Нужен демонтаж натяжного потолка, включая профиль только если это входит в найденную позицию.",
        "search": "демонтаж натяжного потолка включая профиль",
    },
}

ROOM_TYPE_NAMES = {
    "living": "жилая комната",
    "kitchen": "кухня",
    "bathroom": "ванная или санузел",
    "hallway": "коридор или прихожая",
    "other": "помещение",
}


@dataclass(slots=True)
class PhotoEstimateAnswer:
    answer: str
    search_query: str
    points: list[Any]
    context: str


def _normalized_detections(cv_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in SURFACE_KEYS}
    for surface in SURFACE_KEYS:
        raw_items = cv_result.get(surface, [])
        if not isinstance(raw_items, list):
            raise ValueError(f"Поле {surface} должно быть списком")
        seen: set[str] = set()
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise ValueError(f"Элементы {surface} должны быть объектами")
            material = str(raw_item.get("material") or "").strip()
            if not material or material in seen:
                continue
            if material not in MATERIAL_RULES:
                raise ValueError(f"Неизвестный класс материала: {material}")
            expected_surface = MATERIAL_RULES[material]["surface"]
            if expected_surface != surface:
                raise ValueError(f"Материал {material} не относится к поверхности {surface}")
            confidence = float(raw_item.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Некорректная уверенность для {material}: {confidence}")
            result[surface].append({"material": material, "confidence": confidence})
            seen.add(material)
    return result


def _measurement_context(area_m2: float, height_m: float) -> str:
    # Это только диапазон для предварительной оценки: точный периметр и площадь проёмов неизвестны.
    square_wall_area = 4.0 * math.sqrt(area_m2) * height_m
    minimum = square_wall_area * 0.80
    maximum = square_wall_area * 1.20
    return (
        f"Площадь пола и потолка: {area_m2:.2f} м². "
        f"Оценочный диапазон площади стен: {minimum:.2f}–{maximum:.2f} м². "
        "Диапазон получен из площади пола и высоты; точные размеры, периметр, проёмы и доли разных "
        "покрытий неизвестны. Не выдавай его за точный замер."
    )


def _deduplicate_points(points: list[Any]) -> list[Any]:
    result = []
    seen: set[str] = set()
    for point in points:
        key = str(getattr(point, "id", "")) or repr(getattr(point, "payload", None))
        if key in seen:
            continue
        seen.add(key)
        result.append(point)
    return result


MATERIAL_PRICE_ITEM_IDS: dict[str, tuple[str, ...]] = {
    "bare_unfinished_wall": (),
    "exposed_drywall": (),
    "unfinished_plaster_or_putty": (),
    "wallpaper": (
        "snyatie_oboev",
        "snyatie_mnogosloynyh_trudnootdelyaemyh_oboev",
    ),
    "tile": ("demontazh_keramicheskoy_plitki",),
    "rigid_decorative_cladding": (
        "demontazh_paneley_pvh_bez_karkasa",
        "demontazh_oblitsovki_sten_iz_dsp_gkl_bez_karkasa",
        "demontazh_paneley_pvh_s_karkasom",
        "demontazh_obshivki_sten_iz_gkl_na_metallokarkase",
    ),
    "bare_unfinished_floor": (),
    "carpet": (
        "demontazh_linoleuma_kovrolina",
        "demontazh_linoleuma_kovrolina_na_kleyu",
    ),
    "tile_or_stone": ("demontazh_plitki",),
    "vinyl_or_linoleum": (
        "demontazh_linoleuma_kovrolina",
        "demontazh_linoleuma_kovrolina_na_kleyu",
    ),
    "wood_flooring": (
        "demontazh_laminata_parketnoy_doski",
        "demontazh_parketa_shtuchnogo_elochka",
        "demontazh_parketa_schitovogo",
        "demontazh_derevyannyh_polov",
    ),
    "bare_unfinished_ceiling": (),
    "drywall_ceiling": ("demontazh_podvesnyh_potolkov_gkl",),
    "glued_light_ceiling_finish": (
        "snyatie_oboev_potolok",
        "snyatie_polistirolnyh_plitok",
    ),
    "modular_or_panel_ceiling": (
        "demontazh_paneley_kassetnogo_potolka_bez_profilya_armstrong",
        "demontazh_kassetnogo_podvesnogo_potolka_armstrong",
        "demontazh_reechnogo_potolka",
    ),
    "stretch_ceiling": ("demontazh_natyazhnogo_potolka_vklyuchaya_profil",),
}

ZERO_RATE_POSSIBLE = {
    "bare_unfinished_wall",
    "exposed_drywall",
    "unfinished_plaster_or_putty",
    "bare_unfinished_floor",
    "carpet",
    "bare_unfinished_ceiling",
}

FINISHED_MATERIALS = {
    "wallpaper",
    "tile",
    "rigid_decorative_cladding",
    "carpet",
    "tile_or_stone",
    "vinyl_or_linoleum",
    "wood_flooring",
    "drywall_ceiling",
    "glued_light_ceiling_finish",
    "modular_or_panel_ceiling",
    "stretch_ceiling",
}

USER_VISIBLE_CONFIDENCE = 0.80

CONFIDENT_DEMOLITION_ACTIONS = {
    "wallpaper": "снятие обоев",
    "tile": "демонтаж настенной плитки",
    "rigid_decorative_cladding": "демонтаж декоративной облицовки стен",
    "carpet": "снятие коврового покрытия",
    "tile_or_stone": "демонтаж напольной плитки или камня",
    "vinyl_or_linoleum": "демонтаж линолеума или винилового покрытия",
    "wood_flooring": "демонтаж деревянного напольного покрытия",
    "drywall_ceiling": "демонтаж потолка из гипсокартона",
    "glued_light_ceiling_finish": "снятие наклеенной отделки потолка",
    "modular_or_panel_ceiling": "демонтаж модульного или панельного потолка",
    "stretch_ceiling": "демонтаж натяжного потолка",
}

SURFACE_NAMES = {
    "walls": "Стены",
    "floor": "Пол",
    "ceiling": "Потолок",
}

ROUGH_FINISH_MATERIALS = {
    "bare_unfinished_wall",
    "exposed_drywall",
    "unfinished_plaster_or_putty",
    "bare_unfinished_floor",
    "bare_unfinished_ceiling",
}


HIDDEN_WORKS = {
    "bare_unfinished_wall": "Подготовка голых стен зависит от требуемой новой отделки и пока не рассчитана.",
    "exposed_drywall": "Заделка швов и шпаклевание зависят от состояния листов и требуемой отделки.",
    "unfinished_plaster_or_putty": "Повторная штукатурка или шпаклёвка не включена без данных о дефектах.",
    "wallpaper": "Состояние основания после снятия обоев заранее неизвестно.",
    "tile": "Отбивка плиточного клея и старой штукатурки не включена: эти слои по фото не видны.",
    "rigid_decorative_cladding": "Каркас и способ крепления облицовки по фото не определяются.",
    "bare_unfinished_floor": "Стяжка и её толщина по фото не определяются.",
    "carpet": "Свободно лежащий ковёр не требует платного демонтажа; крепление ковролина по фото неизвестно.",
    "tile_or_stone": "Плиточный клей, стяжка и гидроизоляция не включены: это скрытые слои.",
    "vinyl_or_linoleum": "Клеевое крепление используется только для верхней границы вилки.",
    "wood_flooring": "Подложка, лаги и конструкция основания по фото не определяются.",
    "bare_unfinished_ceiling": "Подготовка потолка зависит от требуемой новой отделки и пока не рассчитана.",
    "drywall_ceiling": "Скрытый каркас отдельно не рассчитан.",
    "glued_light_ceiling_finish": "Состояние клея и основания после снятия отделки заранее неизвестно.",
    "modular_or_panel_ceiling": "Скрытый каркас и точная конструкция потолка по фото не определяются.",
    "stretch_ceiling": "Дополнительные потолочные конструкции за полотном по фото не определяются.",
}


class PhotoEstimateService:
    def __init__(
        self,
        rag: RenovationRAG | None = None,
        estimates_collection: str | None = None,
    ) -> None:
        self.rag = rag or RenovationRAG()
        self.estimates_store = QdrantVectorStore(
            collection_name=(
                estimates_collection
                or os.getenv("QDRANT_ESTIMATES_COLLECTION")
                or "renovation_docs_estimates_test"
            )
        )

    def answer(
        self,
        cv_result: dict[str, Any],
        room_type: str,
        area_m2: float,
        height_m: float,
        room_name: str = "",
        top_k_estimates: int = 3,
    ) -> PhotoEstimateAnswer:
        if area_m2 <= 0:
            raise ValueError("Площадь должна быть больше нуля")
        if height_m <= 0:
            raise ValueError("Высота должна быть больше нуля")

        detections = _normalized_detections(cv_result)
        detected_items = [
            (surface, item)
            for surface in SURFACE_KEYS
            for item in detections[surface]
        ]
        price_item_ids = sorted({
            item_id
            for _, item in detected_items
            for item_id in MATERIAL_PRICE_ITEM_IDS[item["material"]]
        })
        pricing_items, pricing_records = self._load_pricing_items(price_item_ids)

        room_display = ROOM_TYPE_NAMES.get(
            room_type,
            room_type.strip() or ROOM_TYPE_NAMES["other"],
        )
        search_queries = [
            (
                f"{MATERIAL_RULES[item['material']]['search']}; "
                f"раздел {surface}; помещение {room_display}"
            )
            for surface, item in detected_items
        ]
        estimate_groups: dict[str, list[Any]] = {}
        if search_queries:
            for (surface, item), query in zip(detected_items, search_queries):
                material = item["material"]
                stage_code = (
                    "rough_finish" if material in ROUGH_FINISH_MATERIALS else "demolition"
                )
                estimate_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_type",
                            match=models.MatchValue(value="estimate"),
                        ),
                        models.FieldCondition(
                            key="stage_code",
                            match=models.MatchValue(value=stage_code),
                        ),
                        models.FieldCondition(
                            key="work_area_code",
                            match=models.MatchValue(value=surface),
                        ),
                    ]
                )
                query_vector = self.rag.embedding_client.embed_texts([query])[0]
                estimate_groups[material] = self.estimates_store.search(
                    query_vector=query_vector,
                    limit=top_k_estimates,
                    query_filter=estimate_filter,
                )

        search_query = (
            f"Предварительная смета: {room_display}, площадь {area_m2:.2f} м², "
            f"высота {height_m:.2f} м. "
            + ". ".join(search_queries)
        )
        calculations = self._calculate_surfaces(
            detections=detections,
            pricing_items=pricing_items,
            area_m2=area_m2,
            height_m=height_m,
        )
        renovation_range = self.rag.estimate_calculator.calculate_whole_renovation(
            target_area_m2=area_m2,
        )
        answer = self._format_answer(
            detections=detections,
            calculations=calculations,
            pricing_items=pricing_items,
            renovation_range=renovation_range,
        )
        context = self._build_debug_context(
            detected_items=detected_items,
            pricing_items=pricing_items,
            estimate_groups=estimate_groups,
        )
        points = _deduplicate_points([
            *pricing_records,
            *(point for points in estimate_groups.values() for point in points),
        ])
        return PhotoEstimateAnswer(
            answer=answer,
            search_query=search_query,
            points=points,
            context=context,
        )

    def _load_pricing_items(
        self,
        item_ids: list[str],
    ) -> tuple[dict[str, dict[str, Any]], list[Any]]:
        if not item_ids:
            return {}, []
        records, _ = self.rag.vector_store.client.scroll(
            collection_name=self.rag.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value="pricing"),
                    ),
                    models.FieldCondition(
                        key="item_id",
                        match=models.MatchAny(any=item_ids),
                    ),
                ]
            ),
            limit=max(20, len(item_ids) * 2),
            with_payload=True,
            with_vectors=False,
        )
        items = {
            str(record.payload.get("item_id")): dict(record.payload)
            for record in records
            if record.payload and record.payload.get("item_id")
        }
        return items, records

    def _calculate_surfaces(
        self,
        detections: dict[str, list[dict[str, Any]]],
        pricing_items: dict[str, dict[str, Any]],
        area_m2: float,
        height_m: float,
    ) -> dict[str, dict[str, Any]]:
        square_wall_area = 4.0 * math.sqrt(area_m2) * height_m
        surface_areas = {
            "walls": (square_wall_area * 0.80, square_wall_area * 1.20),
            "floor": (area_m2, area_m2),
            "ceiling": (area_m2, area_m2),
        }
        result: dict[str, dict[str, Any]] = {}
        for surface in SURFACE_KEYS:
            materials = [item["material"] for item in detections[surface]]
            rates: list[float] = []
            work_options: list[dict[str, Any]] = []
            missing_item_ids: list[str] = []
            for material in materials:
                if material in ZERO_RATE_POSSIBLE:
                    rates.append(0.0)
                material_options = []
                for item_id in MATERIAL_PRICE_ITEM_IDS[material]:
                    price_item = pricing_items.get(item_id)
                    if price_item is None:
                        missing_item_ids.append(item_id)
                        continue
                    price = float(price_item["price"])
                    rates.append(price)
                    material_options.append({
                        "item_id": item_id,
                        "title": str(price_item["item_title"]),
                        "unit": str(price_item["unit"]),
                        "price": price,
                    })
                work_options.append({
                    "material": material,
                    "description": MATERIAL_RULES[material]["description"],
                    "options": material_options,
                })

            area_min, area_max = surface_areas[surface]
            rate_min = min(rates) if rates else 0.0
            rate_max = max(rates) if rates else 0.0
            result[surface] = {
                "area_min": area_min,
                "area_max": area_max,
                "rate_min": rate_min,
                "rate_max": rate_max,
                "cost_min": area_min * rate_min,
                "cost_max": area_max * rate_max,
                "work_options": work_options,
                "missing_item_ids": missing_item_ids,
            }
        return result

    def _format_answer(
        self,
        detections: dict[str, list[dict[str, Any]]],
        calculations: dict[str, dict[str, Any]],
        pricing_items: dict[str, dict[str, Any]],
        renovation_range: Any | None,
    ) -> str:
        detected_materials = {
            item["material"]
            for surface in SURFACE_KEYS
            for item in detections[surface]
        }
        has_missing_surface = any(not detections[surface] for surface in SURFACE_KEYS)
        demolition_may_be_needed = bool(detected_materials & FINISHED_MATERIALS) or has_missing_surface

        lines = ["Для этой комнаты предварительно понадобятся:"]
        if demolition_may_be_needed:
            confident_actions = self._confident_demolition_actions(detections)
            if confident_actions:
                lines.append(
                    "- Судя по фотографии, для подготовительного демонтажа потребуется "
                    f"{self._join_actions(confident_actions)} и другие демонтажные работы."
                )
            else:
                lines.append("- Подготовительный демонтаж существующей отделки; объём уточняется после осмотра.")
        lines.extend([
            "- Подготовка помещения и оснований.",
            "- Черновые отделочные работы.",
            "- Электромонтажные и другие необходимые инженерные работы.",
            "- Чистовая отделка.",
        ])

        if renovation_range is not None:
            total_min = renovation_range.examples_minimum
            total_max = renovation_range.examples_maximum
        else:
            total_min = sum(item["cost_min"] for item in calculations.values())
            total_max = sum(item["cost_max"] for item in calculations.values())
        lines.extend([
            "",
            (
                "Предварительная стоимость полного ремонта комнаты: "
                f"{self._money(total_min)}–{self._money(total_max)} ₽."
            ),
            "Это ориентир по сопоставимым сметам; точная стоимость зависит от замера, состояния оснований и выбранных решений.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _confident_demolition_actions(
        detections: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        actions: list[str] = []
        for surface in SURFACE_KEYS:
            for item in detections[surface]:
                if item["confidence"] <= USER_VISIBLE_CONFIDENCE:
                    continue
                action = CONFIDENT_DEMOLITION_ACTIONS.get(item["material"])
                if action and action not in actions:
                    actions.append(action)
        return actions

    @staticmethod
    def _join_actions(actions: list[str]) -> str:
        return ", ".join(actions)

    def _build_debug_context(
        self,
        detected_items: list[tuple[str, dict[str, Any]]],
        pricing_items: dict[str, dict[str, Any]],
        estimate_groups: dict[str, list[Any]],
    ) -> str:
        blocks = []
        for surface, item in detected_items:
            material = item["material"]
            price_lines = []
            for item_id in MATERIAL_PRICE_ITEM_IDS[material]:
                price_item = pricing_items.get(item_id)
                if price_item:
                    price_lines.append(
                        f"- {price_item['item_title']}: {price_item['price']} {price_item['unit']}"
                    )
            estimates = estimate_groups.get(material, [])
            estimate_context = self.rag.build_context(estimates, max_chars_per_point=1000)
            blocks.append(
                f"{surface} / {material}\n"
                f"Прайс:\n{chr(10).join(price_lines) or '- обязательный демонтаж не выбран'}\n"
                f"Примеры реальных смет:\n{estimate_context or '- подходящие примеры не найдены'}"
            )
        return "\n\n==========\n\n".join(blocks)

    @staticmethod
    def _money(value: float) -> str:
        return f"{value:,.0f}".replace(",", " ")

    @staticmethod
    def _unit(unit: str) -> str:
        return {"кв.м": "м²", "м/п": "м", "шт.": "шт."}.get(unit, unit)


def load_cv_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Корень CV JSON должен быть объектом")
    return data
