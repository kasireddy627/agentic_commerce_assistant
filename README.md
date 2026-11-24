# Agentic Commerce Assistant

A production-ready **Agentic Commerce Assistant** built with **Python**, **Pydantic v2**, and **Streamlit** using a complete dummy store backend. Instead of manually browsing and filtering, the user gives natural-language requirements (e.g., *"I need a laptop under 50k with 16GB RAM and 512GB SSD for coding"*), and the agent autonomously parses constraints, searches inventory, evaluates trade-offs, applies the best coupon, and prepares checkout for **Human-in-the-Loop (HITL)** approval.

## Project Structure

- `app.py` — Streamlit UI (Agentic Shopper chat, live execution trace, HITL checkout approval card, product comparison cards, dummy catalog explorer, order history).
- `agent.py` — `ShoppingAgent` orchestrator running the multi-step reasoning & tool-calling workflow (works 100% offline with zero API keys, with optional OpenAI `gpt-4o-mini` rationale support).
- `tools.py` — `CommerceToolkit` implementing requirement parsing, catalog search, weighted spec scoring, coupon optimization, cart management, and token-gated order placement.
- `catalog.py` — Dummy INR e-commerce catalog (laptops, headphones, monitors, smartphones) and promo coupon rules.
- `models.py` — Strict Pydantic v2 schemas for requirements, products, scored candidates, cart, pending approvals, orders, and tool traces.
- `tests/test_agentic_commerce.py` — Unit and end-to-end tests.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```bash
python3 -m unittest discover -s tests -v
```
