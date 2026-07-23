from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.http import models

from app.embeddings import OpenAIEmbeddingClient
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

SourceMode = Literal["all", "website_only", "pricing_only"]
MAX_QUESTION_CHARS = 500
EMPTY_QUESTION_MESSAGE = "Напишите вопрос о ремонте или услугах компании."
LONG_QUESTION_MESSAGE = "Вопрос слишком длинный. Сформулируйте его короче, до 1000 символов."

COMPANY_FACTS = """
Факты: ремонт квартир под ключ, кухни, ванной, санузла, новостроек, вторички, капитальный и дизайнерский ремонт.
Цена: от 15 000 ₽/м². Если указана площадь, ориентир = площадь × 15 000 ₽.
Точная цена зависит от площади, состояния, состава работ, материалов и пожеланий.
Предварительный расчет возможен до выезда; для точной сметы нужны детали/замер.
География: Москва. Область и другие регионы уточнять у менеджера.
Условия: договор, фиксированная смета, гарантия до 3 лет, оплата по факту.
Контакт: +7(499)372-47-47.
""".strip()


@dataclass(slots=True)
class RAGRoute:
    intent: Intent
    rewritten_query: str
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
        collection_name: str = "renovation_docs",
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

        self.collection_name = collection_name
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
            collection_name=collection_name,
            host=qdrant_host,
            port=qdrant_port,
        )

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

    def answer(self, question: str, top_k: int = 8, top_n: int = 3) -> RAGAnswer:
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

        route = self.classify_question(question)

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
                    "Связаться с менеджером можно по телефону +7(499)372-47-47. "
                    "Также можно оставить заявку в приложении или на сайте."
                ),
                intent=route.intent,
                rewritten_query=route.rewritten_query,
                points=[],
                context="",
            )

        search_query = route.rewritten_query or question
        candidates = self.retrieve_candidates(
            query=search_query,
            source_mode=self.source_mode_for_intent(route.intent),
            top_k=top_k,
        )

        if not candidates:
            return self._manager_fallback(route, points=[], context="")

        best_points = self.rerank_chunks(
            question=question,
            candidates=candidates,
            top_n=top_n,
        )

        if not best_points:
            best_points = candidates[:top_n]

        context = self.build_context(best_points)
        answer = self.generate_answer(
            question=question,
            route=route,
            context=context,
        )

        return RAGAnswer(
            answer=answer,
            intent=route.intent,
            rewritten_query=search_query,
            points=best_points,
            context=context,
        )

    def classify_question(self, question: str) -> RAGRoute:
        prompt = f"""
Определи тип вопроса пользователя для ассистента компании по ремонту квартир.

Верни только JSON без пояснений:
{{"intent": "...", "rewritten_query": "..."}}

Допустимые intent:
- contacts: пользователь просит телефон, контакты, WhatsApp, Telegram, связь с менеджером, куда написать или оставить заявку.
- service_area: вопрос про Москву, область, регион, выезд, где компания работает.
- service_scope: вопрос про тип объекта: только квартиры или также офисы, дома, коттеджи, склады, коммерческие или нежилые помещения.
- payment_terms: вопрос про рассрочку, оплату, этапы оплаты, оплату по факту.
- remote_estimate: вопрос можно ли оценить цену без выезда, без замера, по фото, предварительно или узнать порядок цен.
- broad_pricing: общая стоимость ремонта помещения или квартиры: ремонт кухни, ванной, санузла, однушки, двушки, квартиры 40 м², порядок стоимости ремонта.
- specific_pricing: цена конкретной работы из прайса: демонтаж плитки, установка мойки, монтаж подрозетника, поклейка обоев и т.п.
- service_confirmation: вопрос выполняет ли компания конкретную услугу ремонта: дизайнерский ремонт, капитальный ремонт, ремонт в новостройке, ремонт ванной под ключ и т.п.
- general_rag: другой вопрос по ремонту и услугам компании.
- out_of_scope: вопрос не связан с ремонтом, отделкой, дизайном, ценами, сроками, контактами или услугами компании.

rewritten_query:
- для поиска перепиши вопрос формально и исправь опечатки;
- сохрани смысл;
- не добавляй факты;
- для contacts и out_of_scope можно оставить пустую строку.

Примеры:
"скок выйдет ремнот аднушки" -> {{"intent":"broad_pricing","rewritten_query":"стоимость ремонта однокомнатной квартиры"}}
"а вы ванную под ключ делаете или нет" -> {{"intent":"service_confirmation","rewritten_query":"ремонт ванной комнаты под ключ"}}
"можно без замера хотябы примерно понять цену" -> {{"intent":"remote_estimate","rewritten_query":"предварительная оценка стоимости ремонта без замера"}}
"сколько стоит демонтаж плитки" -> {{"intent":"specific_pricing","rewritten_query":"стоимость демонтажа плитки"}}
"объясни квантовую механику" -> {{"intent":"out_of_scope","rewritten_query":""}}

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
            data = {"intent": "general_rag", "rewritten_query": question}

        intent = str(data.get("intent", "general_rag")).strip()
        if intent not in self.intents():
            intent = "general_rag"

        return RAGRoute(
            intent=intent,  # type: ignore[arg-type]
            rewritten_query=str(data.get("rewritten_query", "") or "").strip(),
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

    def rerank_chunks(self, question: str, candidates: list, top_n: int = 3) -> list:
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
Вопрос пользователя:
{question}

Фрагменты:
{json.dumps(items, ensure_ascii=False)}

Выбери фрагменты, которые лучше всего помогают ответить на вопрос.
Верни только JSON:
{{"indexes": [0, 1, 2]}}

Правила:
- не выбирай фрагменты с ценами отдельных работ, если вопрос про общую стоимость ремонта помещения;
- не выбирай фрагменты про установку отдельного предмета, если вопрос про ремонт помещения под ключ;
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

    def generate_answer(self, question: str, route: RAGRoute, context: str) -> str:
        prompt = f"""
Вопрос пользователя:
{question}

Нормализованный поисковый запрос:
{route.rewritten_query or question}

Тип вопроса:
{route.intent}

Найденные данные:
{context}

Дополнительные правила:
{self.instructions_for_intent(route.intent)}

Ответь коротко, естественно и только по найденным данным.
Если найденные данные дают хотя бы частичный полезный ответ, сначала дай этот ответ.
Если точности не хватает, добавь, что детали лучше уточнить у менеджера.
""".strip()

        response = self.llm_client.responses.create(
            model=self.chat_model,
            input=[
                {"role": "system", "content": self.fast_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=120,
            temperature=0,
        )

        return (response.output_text or "").strip()

    def generate_answer_fast(self, question: str, context: str) -> str:
        prompt = f"""
Вопрос пользователя:
{question}

Проверенные факты компании:
{COMPANY_FACTS}

Найденные данные:
{context}

Сформулируй полезный короткий ответ по найденным данным.
Проверенные факты компании имеют приоритет, если найденные фрагменты противоречат им или слишком узкие.
Если вопрос про общую стоимость ремонта и найдена цена за м², используй ее как ориентир.
Если пользователь указал площадь и найдена цена за м², посчитай примерную стоимость.
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

        if source_mode == "website_only":
            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value="pricing"),
                    )
                ]
            )

        return None

    def source_mode_for_intent(self, intent: Intent) -> SourceMode:
        if intent == "specific_pricing":
            return "pricing_only"

        if intent in {
            "service_area",
            "service_scope",
            "payment_terms",
            "remote_estimate",
            "broad_pricing",
            "service_confirmation",
            "general_rag",
        }:
            return "website_only"

        return "all"

    def instructions_for_intent(self, intent: Intent) -> str:
        common = "Не упоминай контекст, базу знаний, найденные данные или внутреннюю классификацию."

        if intent == "specific_pricing":
            return common + " Используй только точные цены и единицы измерения из найденных данных."

        if intent == "broad_pricing":
            return common + " Если есть цена за м² и пользователь указал площадь, обязательно посчитай ориентир. Если площадь не указана, не придумывай метраж. Не используй цены отдельных работ как цену всего ремонта."

        if intent == "remote_estimate":
            return common + " Если есть общий ориентир цены за м², используй его. Объясни, что предварительный ориентир возможен, но точность зависит от площади, состояния помещения и состава работ. Не используй цены отдельных работ как общий расчет."

        if intent == "service_confirmation":
            return common + " Если найденные данные подтверждают такой вид ремонта или близкую услугу, ответь полезно и прямо. Не называй цены. Если явного подтверждения нет, предложи уточнить у менеджера."

        if intent == "service_scope":
            return common + " Не отвечай уверенно да или нет по объектам вне квартир, если это явно не подтверждено. Лучше предложи уточнить у менеджера."

        if intent == "service_area":
            return common + " По географии работы отвечай только по явно подтвержденным данным. Если регион не подтвержден, предложи уточнить у менеджера."

        if intent == "payment_terms":
            return common + " По оплате и рассрочке отвечай только по явно подтвержденным данным. Если условий нет, предложи уточнить у менеджера."

        return common

    def system_prompt(self) -> str:
        return """
Ты — помощник компании по ремонту квартир.

Правила:
1. Отвечай только по переданным найденным данным.
2. Если есть точная цена, срок, гарантия, формат работы или контакт — используй их в ответе.
3. Если вопрос про стоимость, считай только из явных данных. Не придумывай свои цифры, площади, проценты, сроки или диапазоны.
4. Если есть цена за м² и пользователь указал площадь, можешь посчитать примерную стоимость.
5. Если есть цена за м², но площадь не указана, можно сказать только, что стоимость считается от цены за м², без выдуманных площадей.
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
