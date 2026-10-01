from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.http import models

from app.embeddings import OpenAIEmbeddingClient
from app.estimate_calculator import EstimateCalculator, extract_area_m2
from app.vector_store import QdrantVectorStore

load_dotenv()

Intent = Literal[
    "contacts",
    "service_area",
    "service_scope",
    "payment_terms",
    "remote_estimate",
    "broad_pricing",
    "specific_pricing",
    "service_confirmation",
    "general_rag",
    "out_of_scope",
]

SourceMode = Literal[
    "all",
    "website_only",
    "website_and_estimates",
    "estimates_only",
    "pricing_only",
]
PricingScope = Literal[
    "whole_renovation",
    "work_stage",
    "single_work",
    "not_applicable",
]
WorkStage = Literal[
    "demolition",
    "preparation",
    "rough_finish",
    "finish",
    "plumbing",
    "electrical",
    "heating",
    "unknown",
]
MAX_QUESTION_CHARS = 500
PHOTO_OBJECT_CONTEXT_MARKER = "[PHOTO_OBJECT_CONTEXT]"
EMPTY_QUESTION_MESSAGE = "Напишите вопрос о ремонте или услугах компании."
LONG_QUESTION_MESSAGE = "Вопрос слишком длинный. Сформулируйте его короче, до 1000 символов."

SERVICE_FACTS = (
    "Ремонт квартир под ключ, кухни, ванной, санузла, новостроек, вторички, "
    "капитальный и дизайнерский ремонт."
)
PRICING_FACTS = """
Цена полного ремонта: от 15 000 ₽/м². Если указана площадь, базовый ориентир = площадь × 15 000 ₽.
Точная цена зависит от площади, состояния, состава работ, материалов и пожеланий.
""".strip()
REMOTE_ESTIMATE_FACTS = (
    "Предварительный расчет возможен до выезда; для точной сметы нужны детали или замер."
)
SERVICE_AREA_FACTS = "Москва. Область и другие регионы уточнять у менеджера."
PAYMENT_FACTS = "Договор, фиксированная смета, гарантия до 3 лет, оплата по факту."
CONTACT_PHONE = "+7 (915) 830-36-00"
CONTACT_FACTS = f"Телефон: {CONTACT_PHONE}."


@dataclass(slots=True)
class RAGRoute:
    intent: Intent
    rewritten_query: str
    pricing_scope: PricingScope = "not_applicable"
    work_stage: WorkStage = "unknown"
    raw: str = ""


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    intent: Intent
    rewritten_query: str
    points: list
    context: str


class RenovationRAG:
    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        project: str | None = None,
        min_retrieval_score: float = 0.35,
        fast_min_retrieval_score: float = 0.25,
    ) -> None:
        use_yandex = bool(os.getenv("YANDEX_API_KEY"))

        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "renovation_docs")
        self.chat_model = (
            chat_model
            or os.getenv("YANDEX_CHAT_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL")
            or "gpt-4.1-nano"
        )
        self.min_retrieval_score = min_retrieval_score
        self.fast_min_retrieval_score = fast_min_retrieval_score

        resolved_api_key = api_key or os.getenv("YANDEX_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("YANDEX_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        resolved_project = project or os.getenv("YANDEX_FOLDER_ID") or os.getenv("OPENAI_PROJECT")
        resolved_embedding_model = (
            embedding_model
            or os.getenv("YANDEX_QUERY_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )

        if not resolved_api_key:
            raise ValueError("YANDEX_API_KEY or OPENAI_API_KEY is not set")

        self.llm_client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            project=resolved_project,
        )
        self.embedding_client = OpenAIEmbeddingClient(
            model=resolved_embedding_model,
            batch_size=100,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            project=resolved_project,
            encoding_format="float" if use_yandex else None,
        )
        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            host=qdrant_host,
            port=qdrant_port,
        )
        self.estimate_calculator = EstimateCalculator()

    def answer_fast(self, question: str, top_k: int = 6, top_n: int = 3) -> RAGAnswer:
        validation_error = self.validate_question(question)
        if validation_error:
            return RAGAnswer(
                answer=validation_error,
                intent="out_of_scope",
                rewritten_query="",
                points=[],
                context="",
            )

        question = self.normalize_question(question)

        candidates = self.retrieve_candidates(
            query=question,
            source_mode="all",
            top_k=top_k,
        )

        if not candidates:
            return RAGAnswer(
                answer="Я помогаю только с вопросами о ремонте и услугах компании.",
                intent="out_of_scope",
                rewritten_query=question,
                points=[],
                context="",
            )

        best_score = candidates[0].score or 0.0
        if best_score < self.fast_min_retrieval_score:
            return RAGAnswer(
                answer="Я помогаю только с вопросами о ремонте и услугах компании.",
                intent="out_of_scope",
                rewritten_query=question,
                points=candidates,
                context="",
            )

        best_points = candidates[:top_n]
        context = self.build_context(best_points, max_chars_per_point=700)
        route = RAGRoute(intent="general_rag", rewritten_query=question)
        answer = self.generate_answer_fast(question=question, context=context)

        return RAGAnswer(
            answer=answer,
            intent=route.intent,
            rewritten_query=route.rewritten_query,
            points=best_points,
            context=context,
        )

    def answer(
        self,
        question: str,
        top_k: int = 8,
        top_n: int = 3,
        history: list[dict[str, str]] | None = None,
    ) -> RAGAnswer:
        validation_error = self.validate_question(question)
        if validation_error:
            return RAGAnswer(
                answer=validation_error,
                intent="out_of_scope",
                rewritten_query="",
                points=[],
                context="",
            )

        question = self.normalize_question(question)
        history_text = self.format_history(history)
        photo_object_context = self.photo_object_context(history)

        route = self.classify_question(question, history_text=history_text)

        if route.intent == "out_of_scope":
            return RAGAnswer(
                answer="Я помогаю только с вопросами о ремонте и услугах компании.",
                intent=route.intent,
                rewritten_query=route.rewritten_query,
                points=[],
                context="",
            )

        if route.intent == "contacts":
            return RAGAnswer(
                answer=(
                    f"Связаться с менеджером можно по телефону {CONTACT_PHONE}. "
                    "Также можно оставить заявку в приложении или на сайте."
                ),
                intent=route.intent,
                rewritten_query=route.rewritten_query,
                points=[],
                context="",
            )

        search_query = route.rewritten_query or question
        calculated_context = ""
        target_area_m2 = extract_area_m2(search_query) or extract_area_m2(question)
        if target_area_m2 is not None:
            if route.pricing_scope == "work_stage":
                estimate_range = self.estimate_calculator.calculate(
                    stage_code=route.work_stage,
                    target_area_m2=target_area_m2,
                )
                if estimate_range is not None:
                    calculated_context = estimate_range.as_context()
            elif route.pricing_scope == "whole_renovation":
                renovation_range = self.estimate_calculator.calculate_whole_renovation(
                    target_area_m2=target_area_m2,
                )
                if renovation_range is not None:
                    calculated_context = renovation_range.as_context()

        candidates = self.retrieve_candidates(
            query=search_query,
            source_mode=self.source_mode_for_route(route),
            top_k=top_k,
        )

        if not candidates and not calculated_context and not photo_object_context:
            return self._manager_fallback(route, points=[], context="")

        best_points = []
        if candidates:
            best_points = self.rerank_chunks(
                question=question,
                candidates=candidates,
                top_n=top_n,
                history_text=history_text,
            )

        if not best_points and candidates:
            best_points = candidates[:top_n]

        context = self.build_context(best_points)
        if calculated_context:
            context = (
                f"{calculated_context}\n\n---\n\n{context}"
                if context
                else calculated_context
            )
        if photo_object_context:
            current_object_context = (
                "Подтверждённый результат фото-расчёта для текущего объекта:\n"
                f"{photo_object_context}"
            )
            context = (
                f"{current_object_context}\n\n---\n\n{context}"
                if context
                else current_object_context
            )
        answer = self.generate_answer(
            question=question,
            route=route,
            context=context,
            history_text=history_text,
        )

        return RAGAnswer(
            answer=answer,
            intent=route.intent,
            rewritten_query=search_query,
            points=best_points,
            context=context,
        )

    def classify_question(self, question: str, history_text: str = "") -> RAGRoute:
        prompt = f"""
Определи тип вопроса пользователя для ассистента компании по ремонту квартир.

Верни только JSON без пояснений:
{{"intent": "...", "pricing_scope": "...", "work_stage": "...", "rewritten_query": "..."}}

Допустимые intent:
- contacts: пользователь просит телефон, контакты, WhatsApp, Telegram, связь с менеджером, куда написать или оставить заявку.
- service_area: вопрос про Москву, область, регион, выезд, где компания работает.
- service_scope: вопрос про тип объекта: только квартиры или также офисы, дома, коттеджи, склады, коммерческие или нежилые помещения.
- payment_terms: вопрос про рассрочку, оплату, этапы оплаты, оплату по факту.
- remote_estimate: вопрос можно ли оценить цену без выезда, без замера, по фото, предварительно или узнать порядок цен.
- broad_pricing: общая стоимость ремонта помещения или квартиры: ремонт кухни, ванной, санузла, однушки, двушки, квартиры 40 м², порядок стоимости ремонта.
- specific_pricing: цена конкретной ремонтной работы из прайса: демонтаж плитки, установка мойки, монтаж подрозетника, поклейка обоев и т.п. Вопросы об организации процесса и условиях оказания услуг сюда не относятся.
- service_confirmation: вопрос выполняет ли компания конкретную услугу ремонта: дизайнерский ремонт, капитальный ремонт, ремонт в новостройке, ремонт ванной под ключ и т.п.
- general_rag: другой вопрос по ремонту, организации процесса и условиям оказания услуг компании.
- out_of_scope: вопрос не связан с ремонтом, отделкой, дизайном, ценами, сроками, контактами или услугами компании.

Допустимые pricing_scope:
- whole_renovation: стоимость всего ремонта объекта или помещения целиком;
- work_stage: стоимость целого этапа или комплекса однотипных работ по объекту, например всего демонтажа, всей электрики или всей сантехники;
- single_work: цена одной конкретной операции или позиции;
- not_applicable: вопрос не требует расчёта стоимости.

Допустимые work_stage:
- demolition: демонтажные работы;
- preparation: подготовительные работы;
- rough_finish: черновые отделочные работы;
- finish: чистовые отделочные работы;
- plumbing: сантехнические работы;
- electrical: электромонтажные работы;
- heating: отопление;
- unknown: этап не указан или вопрос не относится к стоимости этапа.

rewritten_query:
- для поиска перепиши вопрос формально и исправь опечатки;
- если текущий вопрос ссылается на предыдущие сообщения, восстанови из истории предмет, помещение, площадь и вид работ;
- rewritten_query должен быть самостоятельным и понятным без истории диалога;
- сохрани смысл;
- не добавляй факты;
- для contacts и out_of_scope можно оставить пустую строку.

Примеры:
"скок выйдет ремнот аднушки" -> {{"intent":"broad_pricing","pricing_scope":"whole_renovation","work_stage":"unknown","rewritten_query":"стоимость ремонта однокомнатной квартиры"}}
"сколько стоит весь демонтаж квартиры" -> {{"intent":"broad_pricing","pricing_scope":"work_stage","work_stage":"demolition","rewritten_query":"стоимость комплекса демонтажных работ по квартире"}}
"сколько стоит демонтаж плитки" -> {{"intent":"specific_pricing","pricing_scope":"single_work","work_stage":"demolition","rewritten_query":"стоимость демонтажа плитки"}}
"а вы ванную под ключ делаете или нет" -> {{"intent":"service_confirmation","pricing_scope":"not_applicable","work_stage":"unknown","rewritten_query":"ремонт ванной комнаты под ключ"}}
"можно без замера хотябы примерно понять цену" -> {{"intent":"remote_estimate","pricing_scope":"whole_renovation","work_stage":"unknown","rewritten_query":"предварительная оценка стоимости ремонта без замера"}}
"объясни квантовую механику" -> {{"intent":"out_of_scope","pricing_scope":"not_applicable","work_stage":"unknown","rewritten_query":""}}

История текущего диалога:
{history_text or "История отсутствует."}

Вопрос пользователя:
{question}
""".strip()

        response = self.llm_client.responses.create(
            model=self.chat_model,
            input=[
                {
                    "role": "system",
                    "content": "Ты классифицируешь вопросы для RAG. Верни только валидный JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=120,
            temperature=0,
        )

        raw = (response.output_text or "").strip()

        try:
            data = self._loads_json_object(raw)
        except json.JSONDecodeError:
            data = {
                "intent": "general_rag",
                "pricing_scope": "not_applicable",
                "work_stage": "unknown",
                "rewritten_query": question,
            }

        intent = str(data.get("intent", "general_rag")).strip()
        if intent not in self.intents():
            intent = "general_rag"

        pricing_scope = str(data.get("pricing_scope", "not_applicable")).strip()
        if pricing_scope not in self.pricing_scopes():
            pricing_scope = "not_applicable"

        work_stage = str(data.get("work_stage", "unknown")).strip()
        if work_stage not in self.work_stages():
            work_stage = "unknown"

        return RAGRoute(
            intent=intent,  # type: ignore[arg-type]
            rewritten_query=str(data.get("rewritten_query", "") or "").strip(),
            pricing_scope=pricing_scope,  # type: ignore[arg-type]
            work_stage=work_stage,  # type: ignore[arg-type]
            raw=raw,
        )

    def retrieve_candidates(
        self,
        query: str,
        source_mode: SourceMode = "all",
        top_k: int = 8,
    ) -> list:
        query_vector = self.embedding_client.embed_texts([query])[0]
        query_filter = self.query_filter_for_source_mode(source_mode)

        return self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

    def validate_question(self, question: str) -> str | None:
        normalized = self.normalize_question(question)

        if not normalized:
            return EMPTY_QUESTION_MESSAGE

        if len(normalized) > MAX_QUESTION_CHARS:
            return LONG_QUESTION_MESSAGE

        return None

    def normalize_question(self, question: str) -> str:
        return " ".join((question or "").split()).strip()

    def format_history(self, history: list[dict[str, str]] | None) -> str:
        if not history:
            return ""

        lines: list[str] = []
        for item in history[-10:]:
            role = str(item.get("role") or "").strip()
            content = self.normalize_question(str(item.get("content") or ""))
            if role not in {"user", "assistant"} or not content:
                continue
            speaker = "Пользователь" if role == "user" else "Ассистент"
            lines.append(f"{speaker}: {content[:MAX_QUESTION_CHARS]}")
        return "\n".join(lines)

    def photo_object_context(self, history: list[dict[str, str]] | None) -> str:
        if not history:
            return ""
        for item in reversed(history[-10:]):
            content = self.normalize_question(str(item.get("content") or ""))
            if PHOTO_OBJECT_CONTEXT_MARKER not in content:
                continue
            return content.replace(PHOTO_OBJECT_CONTEXT_MARKER, "", 1).strip()[:2_000]
        return ""

    def rerank_chunks(
        self,
        question: str,
        candidates: list,
        top_n: int = 3,
        history_text: str = "",
    ) -> list:
        items = []

        for index, point in enumerate(candidates):
            payload = point.payload or {}
            text = (payload.get("text") or "").strip()
            if not text:
                continue

            title = payload.get("title") or payload.get("h1") or ""
            items.append(
                {
                    "index": index,
                    "title": title,
                    "text": text[:1600],
                }
            )

        if not items:
            return []

        prompt = f"""
История текущего диалога:
{history_text or "История отсутствует."}

Вопрос пользователя:
{question}

Фрагменты:
{json.dumps(items, ensure_ascii=False)}

Выбери фрагменты, которые лучше всего помогают ответить на вопрос.
Верни только JSON:
{{"indexes": [0, 1, 2]}}

Правила:
- точный действующий тариф или цена за м², соответствующие запросу, имеют приоритет и должны быть выбраны;
- сопоставляй масштаб запроса и примера: квартиру целиком сравнивай с квартирой целиком, помещение с таким же помещением, отдельную работу с такой же работой;
- не выбирай стоимость отдельного помещения или отдельной операции как основу расчёта для всей квартиры;
- если пользователь указал площадь и просит общую стоимость, предпочитай несколько сопоставимых примеров, где одновременно указаны исходная площадь и стоимость нужного состава работ;
- для расчёта вилки выбирай примеры с одинаковым составом и охватом работ;
- историческую смету считай примером конкретного объекта, а не действующим прайсом;
- если фрагмент не помогает ответить, не выбирай его.
""".strip()

        response = self.llm_client.responses.create(
            model=self.chat_model,
            input=[
                {
                    "role": "system",
                    "content": "Ты выбираешь релевантные фрагменты для RAG. Верни только валидный JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=80,
            temperature=0,
        )

        raw = (response.output_text or "").strip()

        try:
            selected_indexes = self._loads_json_object(raw).get("indexes", [])
        except json.JSONDecodeError:
            selected_indexes = []

        selected: list = []
        seen: set[int] = set()

        for value in selected_indexes:
            if not isinstance(value, int):
                continue
            if value in seen or value < 0 or value >= len(candidates):
                continue
            selected.append(candidates[value])
            seen.add(value)
            if len(selected) >= top_n:
                break

        return selected

    def generate_answer(
        self,
        question: str,
        route: RAGRoute,
        context: str,
        history_text: str = "",
    ) -> str:
        prompt = f"""
История текущего диалога:
{history_text or "История отсутствует."}

Вопрос пользователя:
{question}

Нормализованный поисковый запрос:
{route.rewritten_query or question}

Тип вопроса:
{route.intent}

Масштаб расчёта:
{route.pricing_scope}

Проверенные факты компании:
{self.company_facts_for_route(route)}

Найденные данные:
{context}

Дополнительные правила:
{self.instructions_for_route(route)}

Если пользователь указал площадь и найдены сопоставимые примеры с исходной площадью и стоимостью:
1. Для каждого примера вычисли стоимость на 1 м².
2. Пересчитай каждую ставку на площадь пользователя.
3. Назови минимальный и максимальный результаты как ориентировочную вилку.
4. Не подменяй стоимость всей квартиры стоимостью отдельного помещения или отдельной операции.
5. В ответе не перечисляй внутренние документы и не употребляй выражения «историческая смета» или «в найденных данных».

Если пользователь спрашивает, что входит в ранее рассчитанную стоимость:
1. Для полного ремонта перечисляй входящие этапы, а для отдельного этапа — входящие разделы работ.
2. Не добавляй в состав стоимости услуги из общего описания сайта.
3. Не перечисляй то, что не входит или не подтверждено, если пользователь спрашивает только о включённом составе.
4. Если пользователь прямо спрашивает, входит ли конкретная позиция, подтверждай её включение только по данным самого расчёта.

Ответь коротко, естественно и только по найденным данным.
Используй историю для понимания продолжения разговора. Факты, цены и условия бери из найденных данных, а не из обычных прежних ответов ассистента. Подтверждённый результат фото-расчёта текущего объекта в найденных данных считай фактом этого диалога и используй в последующих вопросах про этот объект.
Если найденные данные дают хотя бы частичный полезный ответ, сначала дай этот ответ.
Если точности не хватает, добавь, что детали лучше уточнить у менеджера.
""".strip()

        response = self.llm_client.responses.create(
            model=self.chat_model,
            input=[
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=150,
            temperature=0,
        )

        return (response.output_text or "").strip()

    def generate_answer_fast(self, question: str, context: str) -> str:
        prompt = f"""
Вопрос пользователя:
{question}

Найденные данные:
{context}

Сформулируй полезный короткий ответ по найденным данным.
Если вопрос про общую стоимость ремонта и найдена цена за м², используй ее как ориентир.
Если пользователь указал площадь и найдена цена за м², посчитай примерную стоимость.
Если пользователь не спрашивал о цене, не называй цены и не добавляй сведения о стоимости.
Если найденные данные относятся к ремонту, но точной информации не хватает, дай полезную часть и предложи уточнить детали у менеджера.
Если найденные данные не помогают ответить на вопрос, скажи, что помогаешь только с вопросами о ремонте и услугах компании.
""".strip()

        response = self.llm_client.responses.create(
            model=self.chat_model,
            input=[
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=160,
            temperature=0,
        )

        return (response.output_text or "").strip()

    def build_context(self, points: list, max_chars_per_point: int | None = None) -> str:
        parts: list[str] = []

        for point in points:
            payload = point.payload or {}
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            if max_chars_per_point and len(text) > max_chars_per_point:
                text = text[:max_chars_per_point].rsplit(" ", 1)[0].strip()

            title = payload.get("title") or payload.get("h1") or ""
            url = payload.get("url") or ""

            block = []
            if title:
                block.append(f"Заголовок: {title}")
            if url:
                block.append(f"URL: {url}")
            block.append(text)

            parts.append("\n".join(block))

        return "\n\n---\n\n".join(parts)

    def query_filter_for_source_mode(self, source_mode: SourceMode) -> models.Filter | None:
        if source_mode == "pricing_only":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value="pricing"),
                    )
                ]
            )

        if source_mode == "estimates_only":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value="estimate"),
                    )
                ]
            )

        if source_mode == "website_only":
            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value="pricing"),
                    ),
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value="estimate"),
                    ),
                ]
            )

        if source_mode == "website_and_estimates":
            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value="pricing"),
                    )
                ]
            )

        return None

    def source_mode_for_route(self, route: RAGRoute) -> SourceMode:
        if route.intent == "specific_pricing":
            return "pricing_only"

        if route.pricing_scope == "work_stage":
            return "estimates_only"

        if route.intent in {
            "remote_estimate",
            "broad_pricing",
        }:
            return "website_and_estimates"

        if route.intent in {
            "service_area",
            "service_scope",
            "payment_terms",
            "service_confirmation",
            "general_rag",
        }:
            return "website_only"

        return "all"

    def company_facts_for_route(self, route: RAGRoute) -> str:
        if route.pricing_scope in {"work_stage", "single_work"}:
            return (
                "Общий тариф ремонта от 15 000 ₽/м² относится ко всему ремонту "
                "и не является ценой отдельного этапа или отдельной работы."
            )

        if route.intent == "broad_pricing":
            return PRICING_FACTS
        if route.intent == "remote_estimate":
            return f"{PRICING_FACTS}\n{REMOTE_ESTIMATE_FACTS}"
        if route.intent == "service_scope":
            return SERVICE_FACTS
        if route.intent == "service_confirmation":
            return "Конкретную услугу подтверждай только по найденным документам."
        if route.intent == "service_area":
            return SERVICE_AREA_FACTS
        if route.intent == "payment_terms":
            return PAYMENT_FACTS
        if route.intent == "contacts":
            return CONTACT_FACTS

        return "Для этого вопроса используй только найденные документы."

    def instructions_for_route(self, route: RAGRoute) -> str:
        common = (
            "Не упоминай контекст, базу знаний, найденные данные или внутреннюю классификацию. "
            "Отвечай только на заданный вопрос и не добавляй несвязанные услуги, цены или условия."
        )

        if route.intent == "specific_pricing":
            return common + " Используй только точные цены и единицы измерения из найденных данных."

        if route.intent == "broad_pricing" and route.pricing_scope == "work_stage":
            return common + " Рассчитывай стоимость этапа только по сопоставимым примерам этого этапа из реальных объектов. Не применяй к этапу общий тариф ремонта за м². Если указана площадь, пересчитай найденные ставки на площадь пользователя и назови ориентировочную вилку."

        if route.intent == "broad_pricing":
            return common + " Точный действующий тариф за м² используй как минимальный базовый порог, а не как полную смету. Если пользователь указал площадь, рассчитай этот порог. Реалистичный диапазон по сопоставимым полным объектам называй отдельно как предварительный бюджет. Не соединяй базовый порог и реалистичный диапазон в одну противоречивую вилку. Если площадь не указана, не придумывай метраж. Не используй цены отдельных работ или помещений как цену всего объекта. Не раскрывай пользователю внутренние документы и источники расчёта."

        if route.intent == "remote_estimate":
            return common + " Если есть сопоставимые примеры с площадью и стоимостью, пересчитай их на площадь пользователя и назови итоговую вилку. Объясни, что точность зависит от состояния помещения и состава работ. Не используй цены отдельных работ или помещений как цену всего объекта. Не раскрывай пользователю внутренние документы и источники расчёта."

        if route.intent == "service_confirmation":
            return common + " Если найденные данные подтверждают такой вид ремонта или близкую услугу, ответь полезно и прямо. Не называй цены. Если явного подтверждения нет, предложи уточнить у менеджера."

        if route.intent == "service_scope":
            return common + " Не отвечай уверенно да или нет по объектам вне квартир, если это явно не подтверждено. Лучше предложи уточнить у менеджера."

        if route.intent == "service_area":
            return common + " По географии работы отвечай только по явно подтвержденным данным. Если регион не подтвержден, предложи уточнить у менеджера."

        if route.intent == "payment_terms":
            return common + " По оплате и рассрочке отвечай только по явно подтвержденным данным. Если условий нет, предложи уточнить у менеджера."

        return common

    def system_prompt(self) -> str:
        return """
Ты — помощник компании по ремонту квартир.

Правила:
1. Отвечай только по переданным найденным данным.
2. Если есть точная цена, срок, гарантия, формат работы или контакт — используй их в ответе.
3. Если вопрос про стоимость, считай только из явных данных. Не придумывай свои цифры, площади, проценты, сроки или диапазоны.
4. Используй цену за м² только тогда, когда она относится к тому же масштабу и составу работ, о которых спрашивает пользователь.
5. Если подходящая цена за м² есть и пользователь указал площадь, можешь посчитать примерную стоимость. Если площадь не указана, не придумывай метраж.
6. Если услуга, тип объекта, регион, рассрочка, замер, закупка материалов или другой факт не подтвержден явно, не отвечай уверенно "да" или "нет". Вместо этого скажи, что это лучше уточнить у менеджера.
7. Если информации недостаточно, но найденные данные частично помогают, сначала дай полезную подтвержденную часть ответа, а потом предложи уточнить детали у менеджера.
8. Если вопрос вообще не связан с ремонтом и услугами компании, коротко скажи, что ты помогаешь только по вопросам ремонта и услуг компании.
9. Не используй общие знания, если их нет в найденных данных.
10. Не упоминай базу знаний, контекст или найденные данные.
11. Отвечай по-русски, коротко и естественно, максимум 4-5 предложений.
12. Если вопрос про общую стоимость ремонта, порядок цен или оценку без выезда, не используй отдельные цены на конкретные работы как ответ на общий вопрос.
13. Не называй компанию по имени, если пользователь сам об этом не спрашивал.
14. Не перефразируй факты так, чтобы менялся смысл. Например, "фиксированная смета" не означает "фиксированная цена".
15. Не комментируй сам вопрос пользователя и не говори фразы вроде "в вашем вопросе есть упоминание".
16. Не говори фразы вроде "в контексте указано", "в контексте есть", "по найденным данным".
17. Не заменяй полезный ответ фразой про менеджера, если в найденных данных есть подходящая информация.
18. Не называй цену, если пользователь не спрашивал о стоимости, цене или порядке расчёта.
19. Свойство одной услуги нельзя переносить на соседнюю услугу. Например, слова «бесплатно», «включено», «гарантия» или цена относятся только к той услуге, для которой это указано явно. Упоминание нескольких услуг через «и» не означает, что свойство первой автоматически относится к остальным.
20. Не утверждай, что одна услуга входит в другую, если такая связь прямо не указана.
""".strip()

    def fast_system_prompt(self) -> str:
        return """
Ты — помощник компании по ремонту квартир.
Отвечай только по фактам и найденным данным.
Не выдумывай цены, сроки, регионы и услуги.
Если есть цена за м² и площадь, посчитай ориентир.
Если данных мало, дай полезную часть и предложи уточнить детали у менеджера.
На вопросы не по ремонту отвечай: "Я помогаю только с вопросами о ремонте и услугах компании."
Пиши коротко, естественно, без слов "контекст", "база знаний", "найденные данные".
""".strip()

    def _manager_fallback(self, route: RAGRoute, points: list, context: str) -> RAGAnswer:
        return RAGAnswer(
            answer="По этому вопросу лучше уточнить детали у менеджера.",
            intent=route.intent,
            rewritten_query=route.rewritten_query,
            points=points,
            context=context,
        )

    @staticmethod
    def intents() -> set[str]:
        return {
            "contacts",
            "service_area",
            "service_scope",
            "payment_terms",
            "remote_estimate",
            "broad_pricing",
            "specific_pricing",
            "service_confirmation",
            "general_rag",
            "out_of_scope",
        }

    @staticmethod
    def pricing_scopes() -> set[str]:
        return {
            "whole_renovation",
            "work_stage",
            "single_work",
            "not_applicable",
        }

    @staticmethod
    def work_stages() -> set[str]:
        return {
            "demolition",
            "preparation",
            "rough_finish",
            "finish",
            "plumbing",
            "electrical",
            "heating",
            "unknown",
        }

    @staticmethod
    def _loads_json_object(raw: str) -> dict:
        text = raw.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        data = json.loads(text)
        if not isinstance(data, dict):
            raise json.JSONDecodeError("JSON value is not an object", text, 0)

        return data
