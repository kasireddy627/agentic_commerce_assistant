from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from models import ScoredProduct, UserRequirement


def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    api_key: str,
) -> Dict[str, Any]:
    """Calls Gemini or OpenAI and returns parsed JSON."""
    if provider == "gemini":
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return json.loads(resp.text or "{}")

    if provider == "openai":
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def resolve_llm_credentials(
    provider_label: str,
    ui_api_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Maps UI selection + env vars to (provider_id, api_key)."""
    if "Gemini" in provider_label:
        key = (ui_api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        return ("gemini", key) if key else ("gemini", None)
    if "OpenAI" in provider_label:
        key = (ui_api_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        return ("openai", key) if key else ("openai", None)
    return (None, None)


def parse_requirements_llm(
    query: str,
    provider: str,
    api_key: str,
) -> UserRequirement:
    """Uses an LLM to extract structured UserRequirement from natural language."""
    system_prompt = (
        "You are an e-commerce requirement parser. Extract structured shopping constraints from the user's query.\n"
        "Return ONLY valid JSON with these exact keys:\n"
        '{"category": "laptop|headphones|monitor|smartphone|null", '
        '"max_budget_inr": number_or_null, '
        '"min_budget_inr": number_or_null, '
        '"min_ram_gb": integer_or_null, '
        '"min_storage_gb": integer_or_null, '
        '"processor_preference": string_or_null, '
        '"use_case": string_or_null, '
        '"must_have_keywords": [string], '
        '"preferred_brands": [string], '
        '"min_rating": number (default 4.0), '
        '"max_delivery_days": integer_or_null}\n'
        "Note: '50k' in India means 50000 INR. '1TB' means 1024 GB."
    )
    data = _call_llm_json(system_prompt, f"User query: {query}", provider, api_key)
    return UserRequirement(
        raw_query=query,
        category=data.get("category") if data.get("category") not in ("null", "", None) else None,
        max_budget_inr=data.get("max_budget_inr"),
        min_budget_inr=data.get("min_budget_inr"),
        min_ram_gb=data.get("min_ram_gb"),
        min_storage_gb=data.get("min_storage_gb"),
        processor_preference=data.get("processor_preference"),
        use_case=data.get("use_case"),
        must_have_keywords=data.get("must_have_keywords") or [],
        preferred_brands=data.get("preferred_brands") or [],
        min_rating=float(data.get("min_rating") or 4.0),
        max_delivery_days=data.get("max_delivery_days"),
    )


def rank_and_explain_llm(
    req: UserRequirement,
    candidates: List[ScoredProduct],
    provider: str,
    api_key: str,
) -> List[ScoredProduct]:
    """Uses an LLM to evaluate trade-offs and re-rank scored candidates for the user's use case."""
    summary_items = [
        {
            "id": c.product.id,
            "title": c.product.title,
            "price_inr": c.product.price_inr,
            "effective_price_inr": c.effective_price_inr,
            "rating": c.product.rating,
            "specs": c.product.specs,
            "review_summary": c.product.review_summary,
            "base_score": c.match_score,
        }
        for c in candidates
    ]
    system_prompt = (
        "You are an expert AI shopping agent. Evaluate these candidate products against the user's requirements.\n"
        "Return ONLY valid JSON:\n"
        '{"rankings": [{"id": "PROD_ID", "match_score": 0-100, "matched_criteria": ["short point"], "tradeoffs": ["short tradeoff"]}]}\n'
        "Order rankings from best to worst fit."
    )
    user_prompt = (
        f"Requirements: {req.model_dump_json()}\n"
        f"Candidates: {json.dumps(summary_items)}"
    )
    data = _call_llm_json(system_prompt, user_prompt, provider, api_key)
    rank_map = {item["id"]: item for item in data.get("rankings", []) if "id" in item}

    updated: List[ScoredProduct] = []
    for c in candidates:
        llm_info = rank_map.get(c.product.id)
        if llm_info:
            updated.append(
                c.model_copy(
                    update={
                        "match_score": float(max(0.0, min(100.0, llm_info.get("match_score", c.match_score)))),
                        "matched_criteria": llm_info.get("matched_criteria") or c.matched_criteria,
                        "tradeoffs": llm_info.get("tradeoffs") or c.tradeoffs,
                    }
                )
            )
        else:
            updated.append(c)

    updated.sort(key=lambda x: (-x.match_score, x.effective_price_inr or x.product.price_inr))
    return updated


def generate_checkout_rationale_llm(
    req: UserRequirement,
    selected: ScoredProduct,
    provider: str,
    api_key: str,
) -> str:
    """Generates a crisp 2-sentence buyer rationale for Gate 4 checkout approval."""
    system_prompt = (
        "You are a shopping agent summarizing why the selected product is the best buy. "
        'Return ONLY JSON: {"rationale": "2-sentence concise explanation"}'
    )
    user_prompt = (
        f"User query: {req.raw_query}\n"
        f"Selected: {selected.product.title} at ₹{selected.effective_price_inr:,.0f}\n"
        f"Pros: {selected.matched_criteria}\n"
        f"Tradeoffs: {selected.tradeoffs}"
    )
    data = _call_llm_json(system_prompt, user_prompt, provider, api_key)
    return str(data.get("rationale", "")).strip()
