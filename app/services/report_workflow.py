from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional, TypedDict

from app.core.config import settings
from app.services import gemini_service
from app.services.rag_service import format_context, retrieve_textbook_context

try:
    from langgraph.graph import StateGraph, END  # type: ignore
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore
    END = "END"  # type: ignore


class ReportState(TypedDict, total=False):
    subject: str
    unit_large: str
    unit_medium: str | None
    unit_small: str | None
    topic_title: str
    topic_description: str
    custom_instructions: str
    rag_context: str
    plan: str
    report: Dict[str, Any]
    critique: Dict[str, Any]
    revision_count: int
    provider: Dict[str, bool]


RUBRIC = """
1) 교과서 개념 연결성: 대주제/중주제/소주제와 문단이 직접 연결되어 있는가
2) 논리 구조: 주장-근거-해석 순서가 유지되는가
3) 학술성: 용어 사용이 정확하고 과장/추측이 적은가
4) 심화성: 중고등학교 수준을 넘어서는 개념이나 이론이 분명하게 설명되는가
5) 탐구 방법: 실제로 하지 않은 실험/시뮬레이션을 사실처럼 쓰지 않았는가, 또는 실험을 썼다면 수학적 절차가 충분히 설명되는가
6) 활용성: 개념이 실제 기술, 공학, 사회 현상 등에 어떻게 연결되는지 구체적인가
7) 결론 품질: 결과 요약, 한계, 후속 탐구 제안이 있는가
8) 생활기록부 요약: 마지막 5줄 내외 활동 요약이 있고, 배운 점과 심화 개념이 포함되는가
""".strip()

ProgressCallback = Callable[[int, str, str], Awaitable[None]]


async def _emit_progress(
    callback: Optional[ProgressCallback],
    percent: int,
    phase: str,
    message: str,
) -> None:
    if callback is None:
        return
    await callback(percent, phase, message)


async def _step_retrieve(state: ReportState, callback: Optional[ProgressCallback] = None) -> ReportState:
    await _emit_progress(callback, 10, "retrieve", "교과서 RAG 컨텍스트를 수집하고 있습니다.")
    chunks = retrieve_textbook_context(
        subject=state["subject"],
        unit_large=state["unit_large"],
        unit_medium=state.get("unit_medium"),
        unit_small=state.get("unit_small"),
        topic_title=state["topic_title"],
        top_k=4,
    )
    state["rag_context"] = format_context(chunks)
    await _emit_progress(callback, 24, "retrieve", "교과서 근거 문맥 추출을 완료했습니다.")
    return state


async def _step_plan(state: ReportState, callback: Optional[ProgressCallback] = None) -> ReportState:
    await _emit_progress(callback, 36, "plan", "탐구 질문과 분석 절차를 설계하고 있습니다.")
    fallback = (
        "연구 질문 3개를 설정한다.\n"
        "1) 교과 개념이 주제에 어떻게 적용되는가\n"
        "2) 어떤 수학적 모델/분석 절차를 선택할 것인가\n"
        "3) 결과의 한계와 확장 가능성은 무엇인가"
    )
    prompt = f"""
다음 정보를 바탕으로 고등학생 심화탐구 보고서의 고난도 계획을 작성하세요.
주제: {state['topic_title']}
설명: {state.get('topic_description', '')}
교과서 문맥:
{state.get('rag_context', '')}

요구사항:
- Markdown 형식
- 탐구 질문 4개 이상
- 분석 절차를 단계별(자료-모형-검증-한계)로 제시
- 실제 보고서 작성 시 사용할 핵심 근거와 반례 가능성 표시
"""
    state["plan"] = await gemini_service.generate_text(prompt, fallback)
    await _emit_progress(callback, 48, "plan", "탐구 계획 수립을 완료했습니다.")
    return state


async def _step_generate(state: ReportState, callback: Optional[ProgressCallback] = None) -> ReportState:
    await _emit_progress(callback, 62, "generate", "보고서 초안을 생성하고 있습니다.")
    instructions = state.get("custom_instructions", "")
    strict_instructions = (
        f"{instructions}\n\n"
        "아래 교과서 문맥과 계획을 반드시 반영하세요.\n"
        "보고서 문체는 학술 보고서 수준으로 유지하고, 문장 밀도를 높이세요.\n"
        "각 섹션에서 핵심 용어를 정의하고 근거-해석-시사점 순서로 작성하세요.\n"
        f"[교과서 문맥]\n{state.get('rag_context', '')}\n\n"
        f"[탐구 계획]\n{state.get('plan', '')}"
    )
    state["report"] = await gemini_service.generate_report_content(
        topic_title=state["topic_title"],
        topic_description=state.get("topic_description", ""),
        custom_instructions=strict_instructions,
    )
    await _emit_progress(callback, 74, "generate", "초안 생성을 완료했습니다.")
    return state


async def _step_critique(state: ReportState, callback: Optional[ProgressCallback] = None) -> ReportState:
    await _emit_progress(callback, 80, "critique", "품질 점검 및 채점 중입니다.")
    critique = await gemini_service.critique_report(state.get("report", {}), RUBRIC)
    state["critique"] = critique
    await _emit_progress(callback, 86, "critique", "품질 점검을 완료했습니다.")
    return state


async def _step_rewrite(state: ReportState, callback: Optional[ProgressCallback] = None) -> ReportState:
    await _emit_progress(callback, 90, "rewrite", "피드백 기반 재작성 라운드를 진행합니다.")
    feedback = (state.get("critique") or {}).get("feedback", "")
    rewritten = await gemini_service.rewrite_report_with_feedback(
        report=state.get("report", {}),
        feedback=feedback,
        custom_instructions=state.get("custom_instructions", ""),
    )
    state["report"] = rewritten
    state["revision_count"] = state.get("revision_count", 0) + 1
    await _emit_progress(callback, 94, "rewrite", "재작성 라운드를 완료했습니다.")
    return state


async def _step_finalize(state: ReportState, callback: Optional[ProgressCallback] = None) -> ReportState:
    await _emit_progress(callback, 97, "finalize", "최종 검증 및 저장 형식을 정리하고 있습니다.")
    report = state.get("report") or {}
    required_keys = [
        "title",
        "research_question",
        "abstract",
        "introduction",
        "background",
        "methodology",
        "analysis",
        "limitations",
        "conclusion",
    ]
    for key in required_keys:
        value = report.get(key)
        if not isinstance(value, str) or len(value.strip()) < 20:
            report[key] = "해당 섹션은 품질 점검에서 보완이 필요합니다. 교과서 근거를 추가해 확장하세요."

    rag_context = state.get("rag_context", "")
    refs = []
    for line in rag_context.splitlines():
        if line.startswith("[") and "]" in line:
            refs.append(line.split(":", 1)[0])
    report["references"] = refs[:4]
    report["quality"] = state.get("critique", {})
    report["plan"] = state.get("plan", "")
    report["provider"] = state.get("provider", {})
    report["pipeline"] = {
        "langgraph_enabled": settings.USE_LANGGRAPH and LANGGRAPH_AVAILABLE,
        "revisions": state.get("revision_count", 0),
    }

    state["report"] = report
    await _emit_progress(callback, 100, "finalize", "보고서 생성이 완료되었습니다.")
    return state


def _need_rewrite(state: ReportState) -> str:
    critique = state.get("critique") or {}
    revision_count = state.get("revision_count", 0)
    approved = bool(critique.get("approved", False))
    score = int(critique.get("score", 0))

    if approved or score >= 85 or revision_count >= settings.MAX_REPORT_REVISIONS:
        return "finalize"
    return "rewrite"


async def run_report_workflow(
    *,
    subject: str,
    unit_large: str,
    unit_medium: str | None,
    unit_small: str | None,
    topic_title: str,
    topic_description: str,
    custom_instructions: str,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    init_state: ReportState = {
        "subject": subject,
        "unit_large": unit_large,
        "unit_medium": unit_medium,
        "unit_small": unit_small,
        "topic_title": topic_title,
        "topic_description": topic_description,
        "custom_instructions": custom_instructions,
        "revision_count": 0,
        "provider": gemini_service.provider_status(),
    }

    if settings.USE_LANGGRAPH and not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph is required but not available. Install with `pip install langgraph`.")

    if settings.USE_LANGGRAPH and LANGGRAPH_AVAILABLE and StateGraph is not None:
        workflow = StateGraph(ReportState)
        async def retrieve_node(state: ReportState) -> ReportState:
            return await _step_retrieve(state, on_progress)

        async def plan_node(state: ReportState) -> ReportState:
            return await _step_plan(state, on_progress)

        async def generate_node(state: ReportState) -> ReportState:
            return await _step_generate(state, on_progress)

        async def critique_node(state: ReportState) -> ReportState:
            return await _step_critique(state, on_progress)

        async def rewrite_node(state: ReportState) -> ReportState:
            return await _step_rewrite(state, on_progress)

        async def finalize_node(state: ReportState) -> ReportState:
            return await _step_finalize(state, on_progress)

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("plan", plan_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("critique", critique_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("finalize", finalize_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "plan")
        workflow.add_edge("plan", "generate")
        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique",
            _need_rewrite,
            {
                "rewrite": "rewrite",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("rewrite", "critique")
        workflow.add_edge("finalize", END)

        app = workflow.compile()
        result = await app.ainvoke(init_state)
        return result.get("report", {})

    state = await _step_retrieve(init_state, on_progress)
    state = await _step_plan(state, on_progress)
    state = await _step_generate(state, on_progress)
    state = await _step_critique(state, on_progress)
    while _need_rewrite(state) == "rewrite":
        state = await _step_rewrite(state, on_progress)
        state = await _step_critique(state, on_progress)
    state = await _step_finalize(state, on_progress)
    return state.get("report", {})
