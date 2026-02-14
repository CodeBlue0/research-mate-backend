import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from urllib import request

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from app.core.config import settings

_vertex_initialized = False


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
    except Exception:
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
    except Exception:
        return None


async def generate_structured_json(prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    for provider in _pick_provider_order():
        if provider == "openai":
            text = _call_openai_chat(prompt, expect_json=True)
        else:
            text = await _generate_with_vertex(prompt, expect_json=True)
        if not text:
            continue
        parsed = _safe_json_loads(text)
        if isinstance(parsed, dict):
            result = dict(fallback)
            result.update(parsed)
            return result

    return fallback


async def generate_text(prompt: str, fallback: str) -> str:
    for provider in _pick_provider_order():
        if provider == "openai":
            text = _call_openai_chat(prompt, expect_json=False)
        else:
            text = await _generate_with_vertex(prompt, expect_json=False)
        if text:
            return text.strip()

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
        "research_question": f"{topic_title}에서 교과 개념이 실제 문제 해결에 어떻게 기여하는가?",
        "abstract": f"본 연구는 {topic_title}를 주제로 고등학교 교과 개념의 확장 가능성을 분석한다.",
        "introduction": f"본 탐구는 '{topic_title}'를 주제로 선정하였다. {topic_description}",
        "background": "핵심 개념 정의와 관련 이론을 교과서 기반으로 정리하였다.",
        "methodology": "연구 질문 설정, 모델 구성, 결과 해석 절차로 탐구를 수행하였다.",
        "analysis": "절차에 따라 결과를 정리하고 변수별 영향을 비교 분석하였다.",
        "limitations": "데이터와 가정의 한계가 결과 해석에 미치는 영향을 검토하였다.",
        "conclusion": "탐구 결과와 한계를 정리하고 후속 연구 방향을 제시하였다.",
        "references": ["[1] 교과서 기반 개념 정리"],
    }

    prompt = f"""
고등학생 심화탐구 보고서 최종본 수준으로 JSON을 생성하세요.
주제: {topic_title}
설명: {topic_description}
추가 지시: {custom_instructions or '없음'}

JSON 키:
- title
- research_question
- abstract
- introduction
- background
- methodology
- analysis
- limitations
- conclusion
- references (문자열 배열)

조건:
1) 각 본문 항목은 6~10문장.
2) 교과 개념과 실제 적용 사이의 연결 문장을 포함.
3) 수식/정량 분석 가능성이 있으면 설명에 포함.
4) 추측성 표현 금지, 근거 중심.
5) references는 최소 2개 항목.
"""

    return await generate_structured_json(prompt, fallback)


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
- introduction
- background
- methodology
- conclusion
"""

    return await generate_structured_json(prompt, fallback)


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
