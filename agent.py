from __future__ import annotations

from typing import List, Optional

from llm_client import (
    generate_checkout_rationale_llm,
    parse_requirements_llm,
    rank_and_explain_llm,
)
from models import (
    AgentPhase,
    AgentResponse,
    Order,
    PendingApproval,
    Product,
    ScoredProduct,
    ToolCallTrace,
    UserRequirement,
)
from tools import CommerceToolkit


class ShoppingAgent:
    """
    LLM-Powered Step-by-Step Human-in-the-Loop (HITL) Agentic Commerce Assistant.
    Uses Gemini or OpenAI when an API key is supplied (with automatic deterministic fallback):
      Gate 1: Approve Parsed Requirements (step_1_parse_requirements)
      Gate 2: Approve Discovered Catalog Candidates (step_2_discover_products)
      Gate 3: Approve Ranked Selection (step_3_evaluate_and_rank)
      Gate 4: Approve Checkout & Coupon Summary (step_4_prepare_checkout)
      Final:  Execute Order & Decrement Stock (step_5_place_order)
    """

    def __init__(self, toolkit: CommerceToolkit) -> None:
        self.toolkit = toolkit

    def step_1_parse_requirements(
        self,
        user_query: str,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AgentResponse:
        engine_used = "deterministic"
        req: UserRequirement
        if llm_provider and api_key:
            try:
                req = parse_requirements_llm(user_query, llm_provider, api_key)
                engine_used = f"llm:{llm_provider}"
            except Exception:
                req = self.toolkit.parse_requirements(user_query)
                engine_used = "deterministic_fallback"
        else:
            req = self.toolkit.parse_requirements(user_query)

        trace = ToolCallTrace(
            step=1,
            phase=AgentPhase.AWAITING_REQUIREMENT_APPROVAL,
            thought=f"Extracting structured constraints from user prompt ({engine_used}).",
            tool_name="llm_parse_requirements" if engine_used.startswith("llm:") else "parse_requirements",
            tool_args={"query": user_query, "engine": engine_used},
            tool_output_summary=(
                f"Category={req.category or 'any'}, MaxBudget=₹{req.max_budget_inr or 'none'}, "
                f"MinRAM={req.min_ram_gb or 'any'}GB, MinSSD={req.min_storage_gb or 'any'}GB, "
                f"UseCase={req.use_case or 'general'}"
            ),
        )
        msg = (
            f"**Step 1 — Requirement Check (`{engine_used}`, Awaiting Your Approval):**\n"
            "I parsed your request into the constraints below. Review or adjust them, then approve to search the catalog."
        )
        return AgentResponse(
            message=msg,
            phase=AgentPhase.AWAITING_REQUIREMENT_APPROVAL,
            requirements=req,
            traces=[trace],
        )

    def step_2_discover_products(
        self,
        req: UserRequirement,
        prior_traces: Optional[List[ToolCallTrace]] = None,
    ) -> AgentResponse:
        traces = list(prior_traces or [])
        candidates = self.toolkit.search_catalog(req)
        relaxed_search = False
        if not candidates and req.max_budget_inr is not None:
            relaxed_req = req.model_copy(update={"max_budget_inr": round(req.max_budget_inr * 1.15, 2)})
            candidates = self.toolkit.search_catalog(relaxed_req)
            relaxed_search = bool(candidates)

        trace = ToolCallTrace(
            step=len(traces) + 1,
            phase=AgentPhase.AWAITING_DISCOVERY_APPROVAL,
            thought="Searching in-stock store inventory against approved requirements.",
            tool_name="search_catalog",
            tool_args={
                "category": req.category,
                "max_budget_inr": req.max_budget_inr,
                "relaxed_15pct": relaxed_search,
            },
            tool_output_summary=f"Found {len(candidates)} in-stock candidate(s).",
        )
        traces.append(trace)

        if not candidates:
            return AgentResponse(
                message="No in-stock products matched those requirements. Adjust your budget or specs and try again.",
                phase=AgentPhase.IDLE,
                requirements=req,
                traces=traces,
            )

        prefix = "⚠️ No exact match within strict budget, so I included options within +15% headroom.\n\n" if relaxed_search else ""
        msg = (
            f"{prefix}**Step 2 — Catalog Discovery (Awaiting Your Approval):**\n"
            f"I found **{len(candidates)}** in-stock product(s). Approve to run spec scoring and coupon evaluation."
        )
        return AgentResponse(
            message=msg,
            phase=AgentPhase.AWAITING_DISCOVERY_APPROVAL,
            requirements=req,
            discovered_products=candidates,
            traces=traces,
        )

    def step_3_evaluate_and_rank(
        self,
        req: UserRequirement,
        candidates: List[Product],
        prior_traces: Optional[List[ToolCallTrace]] = None,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AgentResponse:
        traces = list(prior_traces or [])
        ranked: List[ScoredProduct] = self.toolkit.evaluate_and_rank(candidates, req)

        engine_used = "deterministic"
        if llm_provider and api_key and ranked:
            try:
                ranked = rank_and_explain_llm(req, ranked, llm_provider, api_key)
                engine_used = f"llm:{llm_provider}"
            except Exception:
                engine_used = "deterministic_fallback"

        top_pick = ranked[0]
        trace = ToolCallTrace(
            step=len(traces) + 1,
            phase=AgentPhase.AWAITING_SELECTION_APPROVAL,
            thought=f"Evaluating specs, seller trust, and coupon-adjusted value ({engine_used}).",
            tool_name="llm_evaluate_and_rank" if engine_used.startswith("llm:") else "evaluate_and_rank",
            tool_args={"candidate_count": len(candidates), "engine": engine_used},
            tool_output_summary=(
                f"Top pick: {top_pick.product.title} (Score {top_pick.match_score}/100, "
                f"Effective ₹{top_pick.effective_price_inr:,.0f})"
            ),
        )
        traces.append(trace)

        msg = (
            f"**Step 3 — Evaluation & Ranking (`{engine_used}`, Awaiting Your Selection):**\n"
            f"My #1 recommendation is **{top_pick.product.title}** (Score: **{top_pick.match_score}/100**, "
            f"Effective Price: **₹{top_pick.effective_price_inr:,.0f}**). "
            "Approve this selection or pick any alternative below to prepare checkout."
        )
        return AgentResponse(
            message=msg,
            phase=AgentPhase.AWAITING_SELECTION_APPROVAL,
            requirements=req,
            discovered_products=candidates,
            shortlisted=ranked[:4],
            traces=traces,
        )

    def step_4_prepare_checkout(
        self,
        req: UserRequirement,
        selected: ScoredProduct,
        shipping_address: str,
        payment_method: str,
        max_guardrail_inr: float = 100000.0,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        prior_traces: Optional[List[ToolCallTrace]] = None,
        shortlisted: Optional[List[ScoredProduct]] = None,
    ) -> AgentResponse:
        traces = list(prior_traces or [])
        effective_price = selected.effective_price_inr or selected.product.price_inr

        if effective_price > max_guardrail_inr:
            traces.append(
                ToolCallTrace(
                    step=len(traces) + 1,
                    phase=AgentPhase.AWAITING_SELECTION_APPROVAL,
                    thought="Checking selected item against user's max autonomous budget guardrail.",
                    tool_name="guardrail_check",
                    tool_args={"effective_price_inr": effective_price, "max_guardrail_inr": max_guardrail_inr},
                    tool_output_summary="Blocked checkout prep: exceeds configured max budget guardrail.",
                )
            )
            return AgentResponse(
                message=(
                    f"**Guardrail Alert:** **{selected.product.title}** (₹{effective_price:,.0f}) exceeds your "
                    f"budget cap of **₹{max_guardrail_inr:,.0f}**. Raise the cap in the sidebar or select a lower-priced option."
                ),
                phase=AgentPhase.AWAITING_SELECTION_APPROVAL,
                requirements=req,
                shortlisted=shortlisted or [selected],
                traces=traces,
            )

        rationale = self._build_rationale(selected, req, llm_provider=llm_provider, api_key=api_key)
        approval: PendingApproval = self.toolkit.create_checkout_approval(
            product_id=selected.product.id,
            shipping_address=shipping_address,
            payment_method=payment_method,
            rationale=rationale,
            quantity=1,
        )
        traces.append(
            ToolCallTrace(
                step=len(traces) + 1,
                phase=AgentPhase.AWAITING_CHECKOUT_APPROVAL,
                thought="Added approved product to cart, applied best coupon, and generated final HITL checkout token.",
                tool_name="create_checkout_approval",
                tool_args={
                    "product_id": selected.product.id,
                    "coupon": approval.coupon_code,
                    "final_total_inr": approval.final_total_inr,
                },
                tool_output_summary=f"Approval token {approval.approval_token} ready for final confirmation.",
            )
        )

        msg = (
            f"**Step 4 — Final Checkout Approval (Awaiting Your Confirmation):**\n"
            f"Prepared checkout for **{selected.product.title}** at **₹{approval.final_total_inr:,.0f}** "
            f"(Coupon `{approval.coupon_code or 'None'}` saved **₹{approval.discount_inr:,.0f}**). "
            "Click **Approve & Place Order** to complete the purchase."
        )
        return AgentResponse(
            message=msg,
            phase=AgentPhase.AWAITING_CHECKOUT_APPROVAL,
            requirements=req,
            shortlisted=shortlisted or [selected],
            traces=traces,
            pending_approval=approval,
        )

    def step_5_place_order(self, approval_token: str) -> Order:
        return self.toolkit.place_order(approval_token)

    def _build_rationale(
        self,
        selected: ScoredProduct,
        req: UserRequirement,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        if llm_provider and api_key:
            try:
                text = generate_checkout_rationale_llm(req, selected, llm_provider, api_key)
                if text:
                    return text
            except Exception:
                pass

        reasons = ", ".join(selected.matched_criteria[:3]) if selected.matched_criteria else "strong overall specs"
        tradeoff_note = f" Note: {selected.tradeoffs[0]}." if selected.tradeoffs else ""
        return (
            f"Match score {selected.match_score}/100 based on {reasons}. "
            f"Sold by {selected.product.seller} ({selected.product.seller_rating}/5 rating).{tradeoff_note}"
        )
