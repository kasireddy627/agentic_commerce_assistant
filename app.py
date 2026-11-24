from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import ShoppingAgent
from llm_client import resolve_llm_credentials
from models import AgentPhase, AgentResponse
from tools import CommerceToolkit

load_dotenv()

st.set_page_config(
    page_title="Agentic Commerce Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM STYLING ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        /* Left Sidebar Navigation Radio Styled as Tabs */
        div[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.45rem;
        }
        div[data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 0.65rem 0.9rem;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: rgba(128, 128, 128, 0.04);
            width: 100%;
            transition: all 0.15s ease;
        }
        div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: rgba(59, 130, 246, 0.5);
            background: rgba(59, 130, 246, 0.08);
        }
        /* Stepper Pills */
        .stepper-wrap {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin: 0.5rem 0 1.25rem 0;
        }
        .step-pill {
            padding: 0.65rem 0.9rem;
            border-radius: 10px;
            font-size: 0.875rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: rgba(128, 128, 128, 0.05);
            color: inherit;
        }
        .step-done {
            background: rgba(16, 185, 129, 0.12);
            border-color: rgba(16, 185, 129, 0.45);
            color: #10b981;
        }
        .step-active {
            background: rgba(59, 130, 246, 0.14);
            border-color: rgba(59, 130, 246, 0.55);
            color: #3b82f6;
            box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.2);
        }
        .step-pending {
            opacity: 0.55;
        }
        /* Product Badges & Cards */
        .badge-top {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            background: rgba(245, 158, 11, 0.16);
            color: #d97706;
            border: 1px solid rgba(245, 158, 11, 0.35);
            margin-bottom: 0.4rem;
        }
        .badge-alt {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(128, 128, 128, 0.12);
            margin-bottom: 0.4rem;
        }
        .price-main {
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0.2rem 0;
        }
        .price-strike {
            font-size: 0.9rem;
            text-decoration: line-through;
            opacity: 0.55;
            margin-left: 0.35rem;
            font-weight: 400;
        }
        .coupon-tag {
            display: inline-block;
            font-family: monospace;
            font-size: 0.75rem;
            padding: 0.12rem 0.45rem;
            border-radius: 4px;
            background: rgba(16, 185, 129, 0.14);
            color: #10b981;
            margin-left: 0.35rem;
        }
        /* Compact Status Pill in Sidebar */
        .status-chip {
            padding: 0.45rem 0.7rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 0.4rem;
        }
        .chip-ok {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #10b981;
        }
        .chip-warn {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #d97706;
        }
        .chip-info {
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.35);
            color: #3b82f6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- STATE & HELPERS ---
def init_state() -> None:
    if "toolkit" not in st.session_state:
        st.session_state.toolkit = CommerceToolkit()
    if "agent" not in st.session_state:
        st.session_state.agent = ShoppingAgent(st.session_state.toolkit)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Describe what you're looking for—including your budget and must-have specs "
                    "(e.g., *\"Laptop under ₹50k with 16GB RAM and 512GB SSD for coding\"*). "
                    "I will pause at **every stage** for your explicit sign-off."
                ),
            }
        ]
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "agent"


def start_workflow(query: str, llm_provider: str | None, api_key: str | None) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    resp = agent.step_1_parse_requirements(query, llm_provider=llm_provider, api_key=api_key)
    st.session_state.last_response = resp
    st.session_state.messages.append({"role": "assistant", "content": resp.message})


def reset_workflow(cancel_note: str) -> None:
    toolkit.clear_cart()
    toolkit.pending_approvals.clear()
    st.session_state.last_response = None
    st.session_state.messages.append({"role": "assistant", "content": cancel_note})


def render_stepper(phase: AgentPhase) -> None:
    steps = [
        ("1. Requirements", AgentPhase.AWAITING_REQUIREMENT_APPROVAL),
        ("2. Discovery", AgentPhase.AWAITING_DISCOVERY_APPROVAL),
        ("3. Selection", AgentPhase.AWAITING_SELECTION_APPROVAL),
        ("4. Checkout", AgentPhase.AWAITING_CHECKOUT_APPROVAL),
    ]
    phase_order = [p for _, p in steps]
    current_idx = (
        phase_order.index(phase)
        if phase in phase_order
        else (4 if phase == AgentPhase.ORDER_PLACED else -1)
    )

    pills_html = ['<div class="stepper-wrap">']
    for idx, (label, _) in enumerate(steps):
        if current_idx > idx:
            cls, icon = "step-pill step-done", "✓"
        elif current_idx == idx:
            cls, icon = "step-pill step-active", "●"
        else:
            cls, icon = "step-pill step-pending", "○"
        pills_html.append(f'<div class="{cls}"><span>{icon}</span><span>{label}</span></div>')
    pills_html.append("</div>")
    st.markdown("".join(pills_html), unsafe_allow_html=True)


def render_sidebar() -> tuple[str, str | None, str | None, int, str, str]:
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        nav_options = {
            "agent": "🤖 Agent",
            "catalog": f"🏪 Catalog ({len(toolkit.catalog)})",
            "order": f"📦 Order ({len(toolkit.orders)})",
        }
        selected_tab = st.radio(
            "Workspace",
            options=list(nav_options.keys()),
            format_func=lambda k: nav_options[k],
            key="active_tab",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### 🧠 Model Configuration")
        provider_label = st.selectbox(
            "LLM Engine",
            options=[
                "Google Gemini (gemini-2.5-flash)",
                "OpenAI (gpt-4o-mini)",
                "Offline Rule-Based (No Key)",
            ],
            index=0,
        )
        ui_api_key = ""
        if "Offline" not in provider_label:
            ui_api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="Paste key or load from .env",
            )

        llm_provider, api_key = resolve_llm_credentials(provider_label, ui_api_key)
        if llm_provider and api_key:
            st.markdown(
                f'<div class="status-chip chip-ok">● Live LLM: <code>{llm_provider}</code></div>',
                unsafe_allow_html=True,
            )
        elif llm_provider and not api_key:
            st.markdown(
                '<div class="status-chip chip-warn">▲ No API key — using deterministic fallback</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-chip chip-info">ℹ Offline Rule-Based Engine</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### 🛡️ Policy & Guardrails")
        max_guardrail_inr = st.slider(
            "Autonomous Spend Cap (₹)",
            min_value=10000,
            max_value=150000,
            value=60000,
            step=5000,
            help="Blocks checkout preparation if the selected product exceeds this threshold.",
        )
        shipping_address = st.text_area(
            "Default Delivery Address",
            value="Flat 402, Rushikonda Tech Park Rd, Visakhapatnam, AP 530045",
            height=78,
        )
        payment_method = st.selectbox(
            "Payment Method",
            options=["UPI (GPay / PhonePe)", "HDFC Credit Card", "ICICI NetBanking", "Cash on Delivery"],
            index=0,
        )

        st.divider()
        st.markdown("### 📊 Store Telemetry")
        m1, m2 = st.columns(2)
        m1.metric("In Stock", sum(1 for p in toolkit.catalog if p.in_stock))
        m2.metric("Orders", len(toolkit.orders))

        st.markdown("")
        if st.button("🔄 Reset Session & Store", use_container_width=True):
            for k in ("toolkit", "agent", "messages", "last_response", "active_tab"):
                st.session_state.pop(k, None)
            st.rerun()

    return selected_tab, llm_provider, api_key, max_guardrail_inr, shipping_address, payment_method


# --- INITIALIZE ---
init_state()
toolkit: CommerceToolkit = st.session_state.toolkit
agent: ShoppingAgent = st.session_state.agent
selected_tab, llm_provider, api_key, max_guardrail_inr, shipping_address, payment_method = render_sidebar()


# --- HEADER ---
hdr_left, hdr_right = st.columns([3.5, 1])
with hdr_left:
    st.title("🛒 Agentic Commerce Assistant")
    st.caption("Human-in-the-Loop (HITL) autonomous shopping workflow with auditable decision gates.")
with hdr_right:
    active_phase = st.session_state.last_response.phase if st.session_state.last_response else AgentPhase.IDLE
    st.metric("Workflow State", active_phase.value.replace("_", " ").title())

st.divider()


# ==========================================
# VIEW 1: AGENT
# ==========================================
if selected_tab == "agent":
    last_resp: AgentResponse | None = st.session_state.last_response
    phase = last_resp.phase if last_resp else AgentPhase.IDLE

    render_stepper(phase)

    if phase in (AgentPhase.IDLE, AgentPhase.ORDER_PLACED):
        st.caption("Quick-start prompts:")
        presets = [
            ("💻 Laptop < ₹50k", "I need a laptop under 50k with 16GB RAM and 512GB SSD for coding"),
            ("🎧 ANC Headphones", "Find wireless ANC headphones under 10k with good battery"),
            ("🖥️ 4K USB-C Monitor", "Need a 4K USB-C monitor under 30k for productivity"),
            ("📱 5G Smartphone < ₹30k", "Looking for a 5G smartphone under 30000 with fast charging"),
        ]
        pcols = st.columns(4)
        for col, (label, prompt_text) in zip(pcols, presets):
            if col.button(label, use_container_width=True):
                start_workflow(prompt_text, llm_provider, api_key)
                st.rerun()

    # Conversation Feed
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Execution Traces & Active HITL Gate
    if last_resp:
        if last_resp.traces:
            with st.expander(f"🔍 Execution Trace Log ({len(last_resp.traces)} step(s))", expanded=False):
                for t in last_resp.traces:
                    st.markdown(
                        f"**Step {t.step}** · `{t.phase.value}` · _{t.timestamp}_  \n"
                        f"**Reasoning:** {t.thought}  \n"
                        f"**Action:** `{t.tool_name}({t.tool_args})` → {t.tool_output_summary}"
                    )
                    if t != last_resp.traces[-1]:
                        st.divider()

        # --- GATE 1: REQUIREMENTS ---
        if last_resp.phase == AgentPhase.AWAITING_REQUIREMENT_APPROVAL and last_resp.requirements:
            req = last_resp.requirements
            with st.container(border=True):
                st.markdown("#### 🛡️ Gate 1 · Confirm Parsed Requirements")
                st.caption("Review or adjust extracted constraints before querying the catalog.")

                c1, c2, c3, c4 = st.columns(4)
                cat_opts = ["any", "laptop", "headphones", "monitor", "smartphone"]
                sel_cat = c1.selectbox(
                    "Category",
                    cat_opts,
                    index=cat_opts.index(req.category) if req.category in cat_opts else 0,
                )
                budget_val = c2.number_input(
                    "Max Budget (₹)",
                    min_value=0,
                    max_value=200000,
                    value=int(req.max_budget_inr or 50000),
                    step=1000,
                )
                ram_val = c3.number_input(
                    "Min RAM (GB)",
                    min_value=0,
                    max_value=64,
                    value=int(req.min_ram_gb or 0),
                    step=4,
                )
                ssd_val = c4.number_input(
                    "Min Storage (GB)",
                    min_value=0,
                    max_value=2048,
                    value=int(req.min_storage_gb or 0),
                    step=128,
                )

                st.markdown("")
                b1, b2, _ = st.columns([1.5, 1, 2.5])
                if b1.button("✅ Approve & Search Catalog", type="primary", use_container_width=True):
                    updated_req = req.model_copy(
                        update={
                            "category": None if sel_cat == "any" else sel_cat,
                            "max_budget_inr": float(budget_val) if budget_val > 0 else None,
                            "min_ram_gb": int(ram_val) if ram_val > 0 else None,
                            "min_storage_gb": int(ssd_val) if ssd_val > 0 else None,
                        }
                    )
                    next_resp = agent.step_2_discover_products(updated_req, prior_traces=last_resp.traces)
                    st.session_state.last_response = next_resp
                    st.session_state.messages.append({"role": "assistant", "content": next_resp.message})
                    st.rerun()
                if b2.button("Cancel", key="cancel_g1", use_container_width=True):
                    reset_workflow("Workflow cancelled at Gate 1. Enter a new request below whenever you're ready.")
                    st.rerun()

        # --- GATE 2: DISCOVERY ---
        elif last_resp.phase == AgentPhase.AWAITING_DISCOVERY_APPROVAL and last_resp.discovered_products:
            with st.container(border=True):
                st.markdown("#### 🛡️ Gate 2 · Review Discovered Candidates")
                st.caption(f"Found {len(last_resp.discovered_products)} matching product(s) in inventory.")

                disc_rows = [
                    {
                        "ID": p.id,
                        "Product": p.title,
                        "Price": f"₹{p.price_inr:,.0f}",
                        "Rating": f"{p.rating} ★ ({p.review_count})",
                        "Stock": f"{p.stock_count} units",
                        "ETA": f"{p.delivery_days}d",
                    }
                    for p in last_resp.discovered_products
                ]
                st.dataframe(pd.DataFrame(disc_rows), use_container_width=True, hide_index=True)

                b1, b2, _ = st.columns([1.5, 1, 2.5])
                if b1.button("✅ Approve & Rank Options", type="primary", use_container_width=True):
                    assert last_resp.requirements is not None
                    next_resp = agent.step_3_evaluate_and_rank(
                        last_resp.requirements,
                        last_resp.discovered_products,
                        prior_traces=last_resp.traces,
                        llm_provider=llm_provider,
                        api_key=api_key,
                    )
                    st.session_state.last_response = next_resp
                    st.session_state.messages.append({"role": "assistant", "content": next_resp.message})
                    st.rerun()
                if b2.button("Cancel", key="cancel_g2", use_container_width=True):
                    reset_workflow("Workflow cancelled at Gate 2.")
                    st.rerun()

        # --- GATE 3: SELECTION ---
        elif last_resp.phase == AgentPhase.AWAITING_SELECTION_APPROVAL and last_resp.shortlisted:
            st.markdown("#### 🛡️ Gate 3 · Select Product to Checkout")
            st.caption("Compare scored recommendations, applied coupons, and tradeoffs before locking your choice.")

            top_items = last_resp.shortlisted[:3]
            cols = st.columns(len(top_items))
            for idx, scored in enumerate(top_items):
                p = scored.product
                eff = scored.effective_price_inr or p.price_inr
                with cols[idx]:
                    with st.container(border=True):
                        if idx == 0:
                            st.markdown(
                                f'<span class="badge-top">🏆 #1 Recommended · {scored.match_score}/100</span>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<span class="badge-alt">Option #{idx + 1} · {scored.match_score}/100</span>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(f"**{p.title}**")
                        coupon_html = (
                            f'<span class="coupon-tag">{scored.best_coupon_code}</span>'
                            if scored.best_coupon_code
                            else ""
                        )
                        strike_html = (
                            f'<span class="price-strike">₹{p.price_inr:,.0f}</span>'
                            if eff < p.price_inr
                            else ""
                        )
                        st.markdown(
                            f'<div class="price-main">₹{eff:,.0f}{strike_html}{coupon_html}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"⭐ {p.rating} ({p.review_count}) · 🚚 {p.delivery_days}d · Stock: {p.stock_count}")

                        st.divider()
                        if scored.matched_criteria:
                            for crit in scored.matched_criteria[:3]:
                                st.markdown(f"✅ {crit}")
                        if scored.tradeoffs:
                            for trade in scored.tradeoffs[:2]:
                                st.markdown(f"⚠️ {trade}")

                        st.markdown("")
                        if st.button(
                            "Select & Prepare Checkout",
                            key=f"sel_{p.id}_{idx}",
                            type="primary" if idx == 0 else "secondary",
                            use_container_width=True,
                        ):
                            assert last_resp.requirements is not None
                            next_resp = agent.step_4_prepare_checkout(
                                req=last_resp.requirements,
                                selected=scored,
                                shipping_address=shipping_address,
                                payment_method=payment_method,
                                max_guardrail_inr=max_guardrail_inr,
                                llm_provider=llm_provider,
                                api_key=api_key,
                                prior_traces=last_resp.traces,
                                shortlisted=last_resp.shortlisted,
                            )
                            st.session_state.last_response = next_resp
                            st.session_state.messages.append({"role": "assistant", "content": next_resp.message})
                            st.rerun()

            if st.button("Cancel Workflow", key="cancel_g3"):
                reset_workflow("Workflow cancelled at Gate 3.")
                st.rerun()

        # --- GATE 4: FINAL CHECKOUT ---
        elif last_resp.phase == AgentPhase.AWAITING_CHECKOUT_APPROVAL and last_resp.pending_approval:
            pending = last_resp.pending_approval
            with st.container(border=True):
                st.markdown("#### 🛡️ Gate 4 · Final Checkout Authorization")
                st.caption("Verify order summary, delivery destination, and payment method prior to execution.")

                c1, c2 = st.columns([2.2, 1])
                with c1:
                    st.markdown(f"**Item:** {pending.recommended_product.title}")
                    st.markdown(
                        f"**Seller:** {pending.recommended_product.seller} "
                        f"({pending.recommended_product.seller_rating} ★)"
                    )
                    st.markdown(f"**Ship To:** {pending.shipping_address}")
                    st.markdown(f"**Payment:** {pending.payment_method}")
                    st.info(f"**Agent Rationale:** {pending.rationale}")
                with c2:
                    with st.container(border=True):
                        st.metric(
                            "Payable Total (incl. GST)",
                            f"₹{pending.final_total_inr:,.0f}",
                            delta=f"-₹{pending.discount_inr:,.0f} discount" if pending.discount_inr > 0 else None,
                        )
                        st.caption(f"Subtotal: ₹{pending.subtotal_inr:,.0f}")
                        st.caption(f"Coupon: `{pending.coupon_code or 'None'}`")

                b1, b2, _ = st.columns([1.5, 1, 2.5])
                if b1.button("✅ Authorize & Place Order", type="primary", use_container_width=True):
                    order = agent.step_5_place_order(pending.approval_token)
                    confirm_msg = (
                        f"🎉 **Order Confirmed!** (`{order.order_id}`)\n\n"
                        f"- **Product:** {order.items[0].product.title} (Remaining stock: **{order.items[0].product.stock_count}**)\n"
                        f"- **Total Paid:** ₹{order.total_inr:,.0f} (Saved ₹{order.discount_inr:,.0f} via `{order.coupon_code or 'None'}`)\n"
                        f"- **Delivery:** {order.estimated_delivery_days} day(s) to {order.shipping_address}"
                    )
                    st.session_state.last_response = last_resp.model_copy(
                        update={"phase": AgentPhase.ORDER_PLACED, "pending_approval": None, "placed_order": order}
                    )
                    st.session_state.messages.append({"role": "assistant", "content": confirm_msg})
                    st.rerun()

                if b2.button("Cancel Checkout", key="cancel_g4", use_container_width=True):
                    reset_workflow("Checkout cancelled at Gate 4 and cart cleared.")
                    st.rerun()

    # Persistent Chat Input
    if user_prompt := st.chat_input("What would you like to buy? (e.g., 'Laptop under 50k with 16GB RAM for coding')"):
        start_workflow(user_prompt, llm_provider, api_key)
        st.rerun()


# ==========================================
# VIEW 2: CATALOG
# ==========================================
elif selected_tab == "catalog":
    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.subheader("🏪 Store Inventory")
    with c_right:
        st.caption(f"Showing catalog items ({len(toolkit.catalog)} total)")

    f1, f2, f3 = st.columns([1.2, 2, 1])
    categories = ["all"] + sorted({p.category for p in toolkit.catalog})
    sel_cat = f1.selectbox("Category", categories, format_func=lambda x: x.capitalize())
    max_p = f2.slider("Max Price (₹)", 5000, 80000, 80000, step=1000)
    only_in_stock = f3.checkbox("In-Stock Only", value=False)

    filtered = [
        p
        for p in toolkit.catalog
        if (sel_cat == "all" or p.category == sel_cat)
        and p.price_inr <= max_p
        and (not only_in_stock or p.in_stock)
    ]

    rows = [
        {
            "ID": p.id,
            "Product": p.title,
            "Category": p.category.capitalize(),
            "Price": f"₹{p.price_inr:,.0f}",
            "Rating": f"{p.rating} ★ ({p.review_count})",
            "Stock": p.stock_count if p.in_stock else 0,
            "ETA": f"{p.delivery_days}d",
            "Key Specs": " · ".join(f"{k}: {v}" for k, v in list(p.specs.items())[:4]),
        }
        for p in filtered
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🎟️ Active Store Promotions")
    coupon_rows = [
        {
            "Code": c.code,
            "Offer": c.description,
            "Min Spend": f"₹{c.min_order_inr:,.0f}",
            "Eligible Categories": ", ".join(cat.capitalize() for cat in c.applicable_categories)
            if c.applicable_categories
            else "All Categories",
        }
        for c in toolkit.coupons
    ]
    st.dataframe(pd.DataFrame(coupon_rows), use_container_width=True, hide_index=True)


# ==========================================
# VIEW 3: ORDER
# ==========================================
elif selected_tab == "order":
    st.subheader("📦 Completed Orders")
    if not toolkit.orders:
        st.info("No orders placed yet. Complete a shopping workflow across all 4 approval gates to see orders here.")
    else:
        for o in reversed(toolkit.orders):
            with st.container(border=True):
                oc1, oc2 = st.columns([3, 1])
                with oc1:
                    st.markdown(f"**Order `{o.order_id}`** · `{o.status.value.upper()}` · {o.created_at}")
                    for item in o.items:
                        st.markdown(f"- **{item.quantity}× {item.product.title}** (₹{item.product.price_inr:,.0f})")
                    st.caption(
                        f"📍 {o.shipping_address}  |  💳 {o.payment_method}  |  🚚 ETA: {o.estimated_delivery_days} day(s)"
                    )
                with oc2:
                    st.metric("Total Paid", f"₹{o.total_inr:,.0f}")
                    st.caption(f"Saved: ₹{o.discount_inr:,.0f} · GST: ₹{o.gst_component_inr:,.0f}")