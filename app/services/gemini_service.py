import json
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib import request

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from app.core.config import settings

_vertex_initialized = False
logger = logging.getLogger(__name__)


def _fallback_sections(topic_title: str, topic_description: str) -> List[Dict[str, str]]:
    return [
        {
            "heading": "탐구 동기와 문제의식",
            "content": f"본 탐구는 '{topic_title}'를 주제로 선정하였다. {topic_description}".strip(),
        },
        {
            "heading": "교과 개념 정리",
            "content": "주제와 직접 연결되는 교과 개념을 먼저 정의하고, 보고서 전개에 필요한 핵심 용어를 정리한다.",
        },
        {
            "heading": "심화 이론과 확장 개념",
            "content": "중고등학교 수준을 넘어서는 이론이나 개념을 선정하고, 해당 개념이 왜 이 주제에서 중요한지 설명한다.",
        },
        {
            "heading": "활용 사례와 수학적 해석",
            "content": "수학 개념이 실제 문제, 기술, 공학, 데이터 분석에 어떻게 활용되는지 구체적으로 설명하고 수학적 구조를 해석한다.",
        },
        {
            "heading": "결론과 한계",
            "content": "탐구 결과를 요약하고 한계 및 후속 탐구 방향을 제시한다.",
        },
        {
            "heading": "생활기록부용 활동 요약",
            "content": (
                "교과 개념을 실제 문제와 연결해 심화 탐구를 수행함.\n"
                "주제와 관련된 상위 수준 이론을 학습하고 그 의미를 정리함.\n"
                "수학적 개념이 실제 현상 또는 기술에 적용되는 방식을 분석함.\n"
                "탐구 과정에서 개념 간 연결과 한계를 스스로 점검함.\n"
                "활동을 통해 수학적 모델링과 심화 학습의 필요성을 이해함."
            ),
        },
    ]


def _sections_to_legacy_fields(title: str, sections: List[Dict[str, str]]) -> Dict[str, Any]:
    normalized = [s for s in sections if isinstance(s, dict)]

    def _section(idx: int, fallback: str) -> str:
        if idx < len(normalized):
            value = str(normalized[idx].get("content", "")).strip()
            if value:
                return value
        return fallback

    intro = _section(0, f"본 탐구는 '{title}'를 주제로 진행하였다.")
    background = _section(1, "핵심 개념과 관련 이론을 정리하였다.")
    methodology = _section(2, "탐구 절차와 분석 기준을 설계하였다.")
    analysis = _section(3, "수집한 근거를 바탕으로 분석을 수행하였다.")
    conclusion = _section(4, "탐구 결과와 한계, 후속 탐구 방향을 정리하였다.")

    return {
        "research_question": f"{title}를 통해 교과 개념이 실제 문제 해결에 어떻게 연결되는가?",
        "abstract": intro,
        "introduction": intro,
        "background": background,
        "methodology": methodology,
        "analysis": analysis,
        "limitations": conclusion,
        "conclusion": conclusion,
    }


def _normalize_sections(report: Dict[str, Any], topic_title: str, topic_description: str) -> Dict[str, Any]:
    raw_sections = report.get("sections")
    sections: List[Dict[str, str]] = []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading", "")).strip()
            content = str(item.get("content", "")).strip()
            if heading and content:
                sections.append({"heading": heading, "content": content})

    if not sections:
        sections = _fallback_sections(topic_title, topic_description)

    has_summary = any("생활기록부" in section.get("heading", "") for section in sections)
    if not has_summary:
        sections.append(_fallback_sections(topic_title, topic_description)[-1])

    report["sections"] = sections
    report.setdefault("title", topic_title)
    report.update({k: report.get(k) or v for k, v in _sections_to_legacy_fields(str(report["title"]), sections).items()})
    report.setdefault("references", ["[1] 교과서 기반 개념 정리", "[2] 추가 검토 자료"])
    return report


def _has_vertex_config() -> bool:
    return bool(settings.GOOGLE_CLOUD_PROJECT and settings.GEMINI_MODEL)


def _has_openai_config() -> bool:
    return bool(settings.OPENAI_API_KEY and settings.OPENAI_MODEL)


def provider_status() -> Dict[str, bool]:
    return {
        "openai_compatible": _has_openai_config(),
        "vertex_ai": _has_vertex_config(),
    }


def _pick_provider_order() -> Tuple[str, ...]:
    # Prefer OpenAI-compatible first when configured, otherwise Vertex.
    if _has_openai_config() and _has_vertex_config():
        return ("openai", "vertex")
    if _has_openai_config():
        return ("openai",)
    if _has_vertex_config():
        return ("vertex",)
    return tuple()


def _safe_json_loads(text: str) -> Optional[Any]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        return json.loads(cleaned.strip())
    except Exception:
        return None


def ensure_vertex_initialized() -> None:
    global _vertex_initialized
    if not _vertex_initialized and _has_vertex_config():
        vertexai.init(
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION,
        )
        _vertex_initialized = True


def _call_openai_chat(prompt: str, expect_json: bool = False) -> Optional[str]:
    if not _has_openai_config():
        return None

    effective_base = settings.OPENAI_API_BASE.rstrip("/")
    model_name = settings.OPENAI_MODEL.strip()

    # Auto-correct common provider mismatch.
    if model_name.startswith("deepseek") and "openai.com" in effective_base:
        effective_base = "https://api.deepseek.com/v1"
    if model_name.startswith("gpt-") and "deepseek.com" in effective_base:
        effective_base = "https://api.openai.com/v1"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a precise academic assistant for Korean high school research reports."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    if expect_json:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{effective_base}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        },
    )

    try:
        with request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            return parsed["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("OpenAI-compatible chat call failed: %s", exc)
        return None


def _difficulty_label(difficulty: int) -> str:
    if difficulty < 40:
        return "기본"
    if difficulty < 75:
        return "심화"
    return "도전"


def _fallback_topic(
    subject: str,
    unit_large: str,
    unit_medium: Optional[str],
    unit_small: Optional[str],
    career: str,
    difficulty: int,
) -> Dict[str, Any]:
    chosen_unit = unit_small or unit_medium or unit_large
    return {
        "topic_id": str(uuid.uuid4()),
        "title": f"{chosen_unit} 개념을 활용한 {career or '실생활'} 문제 모델링 연구",
        "reasoning": (
            f"{subject} 교과 핵심 개념을 {career or '관심 분야'}와 연결해 "
            "세특 및 심화 탐구보고서에 활용 가능한 주제로 설계했습니다."
        ),
        "description": (
            f"{unit_large} 단원 개념을 바탕으로 연구 질문을 만들고, "
            "모형 설계-분석-한계 검토를 수행하는 탐구입니다."
        ),
        "tags": [subject, unit_large, career or "탐구"],
        "difficulty": _difficulty_label(difficulty),
        "related_subjects": ["정보", "통계"],
    }


async def _generate_with_vertex(prompt: str, expect_json: bool) -> Optional[str]:
    if not _has_vertex_config():
        return None
    try:
        ensure_vertex_initialized()
        model = GenerativeModel(settings.GEMINI_MODEL)
        if expect_json:
            response = await model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(response_mime_type="application/json"),
            )
        else:
            response = await model.generate_content_async(prompt)
        return (response.text or "").strip()
    except Exception as exc:
        logger.warning("Vertex generation failed: %s", exc)
        return None


async def generate_structured_json(prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    for provider in _pick_provider_order():
        if provider == "openai":
            text = await asyncio.to_thread(_call_openai_chat, prompt, True)
        else:
            text = await _generate_with_vertex(prompt, expect_json=True)
        if not text:
            logger.warning("Structured generation returned no content from provider=%s", provider)
            continue
        parsed = _safe_json_loads(text)
        if isinstance(parsed, dict):
            result = dict(fallback)
            result.update(parsed)
            return result
        logger.warning("Structured generation returned unparsable content from provider=%s", provider)

    return fallback


async def generate_text(prompt: str, fallback: str) -> str:
    for provider in _pick_provider_order():
        if provider == "openai":
            text = await asyncio.to_thread(_call_openai_chat, prompt, False)
        else:
            text = await _generate_with_vertex(prompt, expect_json=False)
        if text:
            return text.strip()
        logger.warning("Text generation returned no content from provider=%s", provider)

    return fallback


async def generate_topics_from_gemini(
    subject: str,
    unit_large: str,
    career: str,
    difficulty: int,
    unit_medium: Optional[str] = None,
    unit_small: Optional[str] = None,
) -> List[Dict[str, Any]]:
    fallback = _fallback_topic(subject, unit_large, unit_medium, unit_small, career, difficulty)

    prompt = f"""
고등학생 심화탐구 주제 1개를 생성하세요.
입력:
- 과목: {subject}
- 대주제: {unit_large}
- 중주제: {unit_medium or '선택 안함'}
- 소주제: {unit_small or '선택 안함'}
- 진로/관심: {career or '미입력'}
- 난이도: {difficulty}

반드시 JSON 객체로 출력:
{{
  "title": "...",
  "reasoning": "...",
  "description": "...",
  "tags": ["..."],
  "difficulty": "기본|심화|도전",
  "related_subjects": ["..."]
}}
"""

    generated = await generate_structured_json(prompt, fallback={
        "title": fallback["title"],
        "reasoning": fallback["reasoning"],
        "description": fallback["description"],
        "tags": fallback["tags"],
        "difficulty": fallback["difficulty"],
        "related_subjects": fallback["related_subjects"],
    })

    generated["topic_id"] = str(uuid.uuid4())
    return [generated]


async def generate_report_content(
    topic_title: str,
    topic_description: str,
    custom_instructions: str = "",
) -> Dict[str, Any]:
    fallback = {
        "title": topic_title,
        "sections": _fallback_sections(topic_title, topic_description),
        "references": ["[1] 교과서 기반 개념 정리"],
    }
    fallback.update(_sections_to_legacy_fields(topic_title, fallback["sections"]))

    prompt = f"""
고등학생 심화탐구 보고서 최종본 수준으로 JSON을 생성하세요.
주제: {topic_title}
설명: {topic_description}
추가 지시: {custom_instructions or '없음'}

JSON 키:
- title
- sections (배열)
- references (문자열 배열)

조건:
1) sections는 5~8개.
2) 각 section은 {{ "heading": "...", "content": "..." }} 형식.
3) heading은 주제에 맞게 자유롭게 생성하고, 고정 목차를 복사하지 말 것.
4) content는 각 섹션마다 6~10문장.
5) 교과 개념과 실제 적용 사이의 연결 문장을 포함.
6) 반드시 최소 1개 이상의 '심화 이론' 섹션을 포함하고, 중고등학교 수준을 넘어서는 개념의 정의와 의미를 자세히 설명할 것.
7) 실험, 시뮬레이션, 코드 실행, 데이터 수집을 실제로 수행하지 않았다면 수행한 것처럼 쓰지 말 것.
8) 실제 실험을 쓰려면 계산 과정, 모델, 수학적 근거를 충분히 설명해야 하며, 그렇지 않으면 개념 설명과 활용 분석 중심으로 쓸 것.
9) 활용 사례에서는 수학 개념이 실제 기술/공학/사회 현상에 어떻게 적용되는지 구체적으로 설명할 것.
10) 수식/정량 분석 가능성이 있으면 설명에 포함하고, 가능한 경우 수학적 의미를 깊이 있게 해설할 것.
11) 마지막 섹션은 반드시 '생활기록부용 활동 요약'으로 하고, 5줄 내외로 학생이 배운 점과 심화 개념을 포함할 것.
12) 추측성 표현 금지, 근거 중심.
13) references는 최소 2개 항목.
"""

    result = await generate_structured_json(prompt, fallback)
    return _normalize_sections(result, topic_title, topic_description)


async def critique_report(report: Dict[str, Any], rubric: str) -> Dict[str, Any]:
    fallback = {
        "approved": False,
        "feedback": "근거 문장과 교과 개념 연결을 보강하세요.",
        "score": 70,
    }

    prompt = f"""
다음 보고서를 엄격하게 평가하세요.
평가기준:\n{rubric}

보고서(JSON):\n{json.dumps(report, ensure_ascii=False)}

반드시 JSON:
{{
  "approved": true/false,
  "feedback": "구체적 수정 지시",
  "score": 0~100 정수
}}
"""

    result = await generate_structured_json(prompt, fallback)
    result["approved"] = bool(result.get("approved", False))
    try:
        result["score"] = int(result.get("score", 70))
    except Exception:
        result["score"] = 70
    return result


async def rewrite_report_with_feedback(
    report: Dict[str, Any],
    feedback: str,
    custom_instructions: str,
) -> Dict[str, Any]:
    fallback = dict(report)

    prompt = f"""
다음 보고서를 피드백에 맞춰 재작성하세요.
피드백: {feedback}
추가 지시: {custom_instructions or '없음'}
현재 보고서(JSON): {json.dumps(report, ensure_ascii=False)}

반드시 JSON으로 반환:
- title
- sections
- references

중요 규칙:
1) 심화 이론 설명을 약하게 쓰지 말고, 중고등학교 수준을 넘어서는 개념을 분명하게 드러낼 것.
2) 실제로 수행하지 않은 실험이나 시뮬레이션을 사실처럼 쓰지 말 것.
3) 실험을 유지하려면 수학적 절차와 계산 근거를 구체적으로 보강할 것.
4) 마지막 섹션은 반드시 '생활기록부용 활동 요약'으로 유지할 것.
"""

    result = await generate_structured_json(prompt, fallback)
    return _normalize_sections(result, str(report.get("title", "보고서")), str(report.get("introduction", "")))


async def chat_about_report(
    report_title: str,
    report_content: Dict[str, Any],
    user_message: str,
) -> str:
    fallback_reply = (
        "보고서 맥락을 기준으로 답변합니다. "
        f"질문: {user_message}\n"
        "핵심 조언: 주장-근거-해석 구조로 문단을 보강하고, 근거 출처를 함께 제시하세요."
    )

    prompt = f"""
당신은 심화탐구 보고서 첨삭 조교입니다.
보고서 제목: {report_title}
보고서 내용(JSON): {json.dumps(report_content, ensure_ascii=False)}
학생 질문: {user_message}

요구사항:
- 한국어로 간결하게 답변
- 문장 수정이 필요하면 예시 1개 포함
- 근거 없는 단정 금지
"""

    return await generate_text(prompt, fallback_reply)
