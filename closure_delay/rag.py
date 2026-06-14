from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RAG_INSTRUCTION = (
    "Use only the provided context. Give a concise answer and cite the document IDs "
    "that support the answer."
)

DEFAULT_RAG_CONDITIONS = [
    {
        "condition": "no_suffix",
        "family": "baseline",
        "suffix": "",
    },
    {
        "condition": "irrelevant_clear_format",
        "family": "irrelevant",
        "suffix": "Please keep the response clear and well formatted.",
    },
    {
        "condition": "verbose_only",
        "family": "verbose",
        "suffix": "Please provide a detailed explanation before giving the final answer.",
    },
    {
        "condition": "manual_verification",
        "family": "manual",
        "suffix": (
            "Before giving the final answer, verify the answer against the retrieved "
            "context and check whether the evidence is consistent."
        ),
    },
    {
        "condition": "structured_multistage",
        "family": "structured",
        "suffix": (
            "Before giving the final answer, reason in short independent stages. "
            "First identify the minimal supporting evidence. Then cross-check it "
            "against another retrieved passage if available. Then check whether any "
            "retrieved passage conflicts with the answer. Avoid repetition, and give "
            "the final answer after the staged verification is complete."
        ),
    },
]


@dataclass(frozen=True)
class RagContext:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RagRecord:
    id: str
    question: str
    answer: str
    contexts: list[RagContext]
    supporting_doc_ids: list[str] = field(default_factory=list)
    answer_aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_rag_records(
    path: str | Path,
    *,
    n_samples: int | None = None,
    seed: int = 42,
    top_k_contexts: int | None = None,
    max_context_chars: int | None = 1600,
) -> list[RagRecord]:
    raw_items = _load_json_or_jsonl(Path(path))
    records = [
        _record_from_mapping(item, index, top_k_contexts=top_k_contexts, max_context_chars=max_context_chars)
        for index, item in enumerate(raw_items)
    ]
    records = [record for record in records if record.question and record.contexts]
    if n_samples is not None:
        rng = random.Random(int(seed))
        rng.shuffle(records)
        records = records[: min(int(n_samples), len(records))]
    return records


def load_rag_conditions(path: str | Path | None = None, optimized_suffix_json: str | Path | None = None) -> list[dict]:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("conditions", [])
        conditions = [dict(item) for item in payload]
    else:
        conditions = [dict(item) for item in DEFAULT_RAG_CONDITIONS]

    if optimized_suffix_json:
        suffix_path = Path(optimized_suffix_json)
        if suffix_path.exists():
            payload = json.loads(suffix_path.read_text(encoding="utf-8"))
            suffix = str(payload.get("suffix", "")).strip()
            if suffix:
                conditions.append(
                    {
                        "condition": "optimized_structured",
                        "family": "optimized",
                        "suffix": suffix,
                    }
                )
    _validate_conditions(conditions)
    return conditions


def format_rag_prompt(
    record: RagRecord,
    *,
    instruction: str = DEFAULT_RAG_INSTRUCTION,
    top_k_contexts: int | None = None,
) -> str:
    contexts = record.contexts[:top_k_contexts] if top_k_contexts else record.contexts
    context_blocks = []
    for context in contexts:
        title = f" Title: {context.title}" if context.title else ""
        context_blocks.append(f"[{context.doc_id}]{title}\n{context.text}")
    return (
        "You are answering a question using the retrieved context below.\n\n"
        "Context:\n"
        + "\n\n".join(context_blocks)
        + "\n\nQuestion:\n"
        + record.question
        + "\n\nInstruction:\n"
        + instruction
    )


def answer_match_metrics(response_text: str, answer: str, aliases: Sequence[str] | None = None) -> dict:
    candidates = [answer, *(aliases or [])]
    normalized_response = normalize_answer(response_text)
    matches = [candidate for candidate in candidates if candidate and normalize_answer(candidate) in normalized_response]
    predicted = extract_final_answer_text(response_text)
    predicted_norm = normalize_answer(predicted)
    exact_matches = [candidate for candidate in candidates if normalize_answer(candidate) == predicted_norm and predicted_norm]
    f1_scores = [_token_f1(predicted, candidate) for candidate in candidates if candidate]
    best_f1 = max(f1_scores) if f1_scores else None
    return {
        "answer_contains": bool(matches),
        "answer_match_count": int(len(matches)),
        "matched_answer": matches[0] if matches else "",
        "answer_predicted_text": predicted,
        "answer_exact_match": bool(exact_matches),
        "answer_f1": best_f1,
        "answer_correct_proxy": bool(exact_matches) or bool(best_f1 is not None and best_f1 >= 0.80),
    }


def evidence_closure_metrics(
    response_text: str,
    response_ids: Sequence[int],
    tokenizer,
    *,
    answer: str,
    supporting_doc_ids: Sequence[str] | None = None,
    answer_aliases: Sequence[str] | None = None,
    require_answer_for_evidence_closure: bool = False,
) -> dict:
    supporting = normalize_doc_ids(supporting_doc_ids or [])
    cited = citation_doc_ids(response_text)
    citation = citation_metrics(cited, supporting)
    answer_match = answer_match_metrics(response_text, answer, answer_aliases)

    support_char = _support_coverage_char(response_text, supporting)
    answer_char = _answer_char(response_text, answer, answer_aliases)
    evidence_char = support_char
    if require_answer_for_evidence_closure and support_char is not None:
        if answer_char is None:
            evidence_char = None
        else:
            evidence_char = max(support_char, answer_char)

    evidence_token = _char_to_token(response_ids, tokenizer, evidence_char)
    answer_token = _char_to_token(response_ids, tokenizer, answer_char)
    generated_tokens = len(response_ids)
    post_evidence = int(generated_tokens - evidence_token) if evidence_token is not None else None

    return {
        **citation,
        **answer_match,
        "evidence_closure_char": evidence_char,
        "evidence_closure_token": evidence_token,
        "answer_first_char": answer_char,
        "answer_first_token": answer_token,
        "post_evidence_tokens": post_evidence,
        "uncited_tail_sentence_rate": uncited_tail_sentence_rate(response_text, evidence_char),
    }


def rag_stage_summary(response_text: str) -> dict:
    text = response_text or ""
    lowered = text.lower()
    cited = citation_doc_ids(text)
    citation_switches = 0
    for previous, current in zip(cited, cited[1:]):
        if previous != current:
            citation_switches += 1
    return {
        "evidence_stage_count": _count_patterns(
            lowered,
            [
                "supporting evidence",
                "evidence",
                "retrieved context",
                "according to doc",
                "according to the context",
                "minimal supporting",
            ],
        ),
        "cross_check_stage_count": _count_patterns(
            lowered,
            [
                "cross-check",
                "cross check",
                "another passage",
                "another retrieved",
                "compare",
                "consistent with",
            ],
        ),
        "conflict_check_stage_count": _count_patterns(
            lowered,
            [
                "conflict",
                "contradict",
                "inconsistent",
                "no conflicting",
                "does not conflict",
            ],
        ),
        "citation_count": int(len(cited)),
        "citation_switch_count": int(citation_switches),
    }


def citation_doc_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in re.finditer(r"(?:\[|\b)(?:doc|document)\s*[-_ ]?(\d+)(?:\]|\b)", text or "", flags=re.IGNORECASE):
        doc_id = f"Doc {int(match.group(1))}"
        seen.append(doc_id)
    return seen


def normalize_doc_ids(doc_ids: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in doc_ids:
        text = str(raw).strip()
        if not text:
            continue
        match = re.search(r"(?:doc|document)\s*[-_ ]?(\d+)", text, flags=re.IGNORECASE)
        if match:
            value = f"Doc {int(match.group(1))}"
        else:
            value = text
        if value not in out:
            out.append(value)
    return out


def citation_metrics(cited_doc_ids: Sequence[str], supporting_doc_ids: Sequence[str]) -> dict:
    cited = normalize_doc_ids(cited_doc_ids)
    supporting = normalize_doc_ids(supporting_doc_ids)
    cited_set = set(cited)
    supporting_set = set(supporting)
    if not supporting_set:
        return {
            "cited_doc_ids_json": json.dumps(cited, ensure_ascii=False),
            "supporting_doc_ids_json": json.dumps(supporting, ensure_ascii=False),
            "citation_precision": None,
            "citation_recall": None,
            "support_coverage": None,
            "answer_supported": None,
        }
    true_positive = len(cited_set & supporting_set)
    precision = true_positive / len(cited_set) if cited_set else None
    recall = true_positive / len(supporting_set)
    return {
        "cited_doc_ids_json": json.dumps(cited, ensure_ascii=False),
        "supporting_doc_ids_json": json.dumps(supporting, ensure_ascii=False),
        "citation_precision": precision,
        "citation_recall": recall,
        "support_coverage": recall,
        "answer_supported": bool(supporting_set and true_positive == len(supporting_set)),
    }


def uncited_tail_sentence_rate(response_text: str, evidence_char: int | None) -> float | None:
    if evidence_char is None:
        return None
    tail = (response_text or "")[int(evidence_char) :]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", tail) if len(part.strip().split()) >= 5]
    if not sentences:
        return 0.0
    uncited = [sentence for sentence in sentences if not citation_doc_ids(sentence)]
    return float(len(uncited) / len(sentences))


def normalize_answer(text: str) -> str:
    lowered = (text or "").lower()
    lowered = "".join(" " if char in string.punctuation else char for char in lowered)
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    return " ".join(lowered.split())


def extract_final_answer_text(response_text: str) -> str:
    text = (response_text or "").strip()
    if not text:
        return ""
    patterns = [
        r"(?:final answer|answer)\s*[:\-]\s*(.+)",
        r"(?:therefore|thus|hence),?\s+(?:the\s+)?answer\s+is\s+(.+)",
        r"(?:so),?\s+(?:the\s+)?answer\s+is\s+(.+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if not matches:
            continue
        candidate = matches[-1].group(1).strip()
        return _first_answer_span(candidate)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _first_answer_span(nonempty_lines[-1]) if nonempty_lines else _first_answer_span(text)


def _first_answer_span(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return ""
    parts = re.split(r"(?<=[.!?])\s+|\n", candidate, maxsplit=1)
    return parts[0].strip(" \t\n\r`*")


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        count = pred_counts.get(token, 0)
        if count <= 0:
            continue
        overlap += 1
        pred_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return float(2.0 * precision * recall / max(precision + recall, 1e-12))


def _load_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping):
        for key in ("data", "examples", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value]
    raise ValueError(f"Unsupported RAG dataset format: {path}")


def _record_from_mapping(
    item: Mapping[str, Any],
    index: int,
    *,
    top_k_contexts: int | None,
    max_context_chars: int | None,
) -> RagRecord:
    question = str(item.get("question") or item.get("query") or item.get("prompt") or "")
    answer = _answer_from_item(item)
    contexts = _contexts_from_item(item, top_k_contexts=top_k_contexts, max_context_chars=max_context_chars)
    title_to_doc = {context.title: context.doc_id for context in contexts if context.title}
    support_ids = _supporting_doc_ids_from_item(item, title_to_doc)
    aliases = item.get("answer_aliases") or item.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return RagRecord(
        id=str(item.get("id") or item.get("_id") or f"rag_{index}"),
        question=question,
        answer=answer,
        contexts=contexts,
        supporting_doc_ids=support_ids,
        answer_aliases=[str(alias) for alias in aliases],
        metadata={key: value for key, value in item.items() if key not in {"contexts", "context", "retrieved_contexts"}},
    )


def _answer_from_item(item: Mapping[str, Any]) -> str:
    value = item.get("answer")
    if value is None:
        value = item.get("answers")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, Mapping):
        texts = value.get("text") or value.get("answers")
        if isinstance(texts, list):
            return str(texts[0]) if texts else ""
    return str(value or "")


def _contexts_from_item(
    item: Mapping[str, Any],
    *,
    top_k_contexts: int | None,
    max_context_chars: int | None,
) -> list[RagContext]:
    raw = item.get("contexts") or item.get("retrieved_contexts") or item.get("context") or []
    contexts: list[RagContext] = []
    if isinstance(raw, str):
        raw = [{"text": raw}]
    for index, entry in enumerate(raw):
        if top_k_contexts is not None and len(contexts) >= int(top_k_contexts):
            break
        title = ""
        text = ""
        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, Mapping):
            title = str(entry.get("title") or entry.get("doc_id") or entry.get("id") or "")
            text = str(entry.get("text") or entry.get("contents") or entry.get("content") or "")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            title = str(entry[0])
            sentences = entry[1]
            if isinstance(sentences, list):
                text = " ".join(str(sentence) for sentence in sentences)
            else:
                text = str(sentences)
        text = " ".join(text.split())
        if max_context_chars is not None and int(max_context_chars) > 0:
            text = text[: int(max_context_chars)]
        if text:
            contexts.append(RagContext(doc_id=f"Doc {len(contexts) + 1}", title=title, text=text))
    return contexts


def _supporting_doc_ids_from_item(item: Mapping[str, Any], title_to_doc: Mapping[str, str]) -> list[str]:
    explicit = item.get("supporting_doc_ids") or item.get("supporting_docs") or item.get("gold_doc_ids")
    values: list[str] = []
    if explicit:
        values = [str(value) for value in explicit]
    supporting_facts = item.get("supporting_facts") or item.get("supporting_facts_titles")
    if supporting_facts:
        for fact in supporting_facts:
            title = None
            if isinstance(fact, str):
                title = fact
            elif isinstance(fact, (list, tuple)) and fact:
                title = str(fact[0])
            elif isinstance(fact, Mapping):
                title = str(fact.get("title") or fact.get("doc") or "")
            if title:
                values.append(title_to_doc.get(title, title))
    mapped = [title_to_doc.get(value, value) for value in values]
    return normalize_doc_ids(mapped)


def _support_coverage_char(response_text: str, supporting_doc_ids: Sequence[str]) -> int | None:
    if not supporting_doc_ids:
        return None
    positions = []
    for doc_id in normalize_doc_ids(supporting_doc_ids):
        match = re.search(
            rf"(?:\[|\b)(?:doc|document)\s*[-_ ]?{re.escape(doc_id.split()[-1])}(?:\]|\b)",
            response_text or "",
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        positions.append(match.end())
    return int(max(positions)) if positions else None


def _answer_char(response_text: str, answer: str, aliases: Sequence[str] | None = None) -> int | None:
    lowered = (response_text or "").lower()
    candidates = [answer, *(aliases or [])]
    best = None
    for candidate in candidates:
        candidate_text = str(candidate or "").strip().lower()
        if not candidate_text:
            continue
        index = lowered.find(candidate_text)
        if index >= 0:
            end = index + len(candidate_text)
            if best is None or end < best:
                best = end
    return best


def _char_to_token(response_ids: Sequence[int], tokenizer, char_index: int | None) -> int | None:
    if char_index is None or tokenizer is None or not response_ids:
        return None
    target = int(char_index)
    for index in range(len(response_ids)):
        prefix = tokenizer.decode(list(response_ids[: index + 1]), skip_special_tokens=True)
        if len(prefix) >= target:
            return int(index + 1)
    return len(response_ids) if response_ids else None


def _count_patterns(text: str, patterns: Sequence[str]) -> int:
    return int(sum(text.count(pattern) for pattern in patterns))


def _validate_conditions(conditions: Sequence[Mapping[str, Any]]) -> None:
    names: set[str] = set()
    for index, condition in enumerate(conditions):
        name = str(condition.get("condition") or "")
        if not name:
            raise ValueError(f"condition {index} is missing condition")
        if name in names:
            raise ValueError(f"duplicate condition: {name}")
        names.add(name)
        if "suffix" not in condition:
            raise ValueError(f"condition {name} is missing suffix")
