from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnswerSchema:
    answer: str
    justification: List[str]


def parse_answer_schema(text: str) -> Optional[AnswerSchema]:
    if not text or "Answer:" not in text or "Justification:" not in text:
        return None

    try:
        ans_idx = text.index("Answer:")
        just_idx = text.index("Justification:")
    except ValueError:
        return None

    if just_idx <= ans_idx:
        return None

    answer_text = text[ans_idx + len("Answer:"):just_idx].strip()
    just_text = text[just_idx + len("Justification:"):].strip()
    if not answer_text:
        return None

    reasons = []
    for line in just_text.splitlines():
        line = line.strip()
        if line.startswith("-"):
            reason = line[1:].strip()
            if reason:
                reasons.append(reason)

    if len(reasons) < 2:
        return None

    return AnswerSchema(answer=answer_text, justification=reasons)


def build_repair_prompt(query: str, context_payload: str, broken_output: str) -> str:
    return (
        f"Original Query: {query}\n\n"
        f"Context Payload:\n{context_payload}\n\n"
        f"Broken Output:\n{broken_output}\n\n"
        "Repair the output to exactly this schema:\n"
        "Answer:\n<concise direct answer>\n\n"
        "Justification:\n"
        "- <reason grounded in context>\n"
        "- <reason grounded in context>\n\n"
        "Rules:\n"
        "- Keep claims strictly grounded in provided context.\n"
        "- If missing context, explicitly say 'insufficient context'.\n"
        "- Do not add extra sections."
    )
