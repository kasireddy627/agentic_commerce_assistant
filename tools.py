from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from catalog import calculate_coupon_discount, get_active_coupons, get_dummy_catalog
from models import (
    Cart,
    CartItem,
    Coupon,
    Order,
    OrderStatus,
    PendingApproval,
    Product,
    ScoredProduct,
    UserRequirement,
)


class CommerceToolkit:
    """Stateful commerce backend & tool suite used by the shopping agent."""

    def __init__(self) -> None:
        self.catalog: List[Product] = get_dummy_catalog()
        self.coupons: List[Coupon] = get_active_coupons()
        self.cart: Cart = Cart()
        self.orders: List[Order] = []
        self.pending_approvals: Dict[str, PendingApproval] = {}

    def get_product(self, product_id: str) -> Optional[Product]:
        for p in self.catalog:
            if p.id == product_id:
                return p
        return None

    # 1. Requirement Parsing Tool
    def parse_requirements(self, query: str) -> UserRequirement:
        q = query.lower()

        # Category detection
        category = None
        category_map = {
            "laptop": ["laptop", "notebook", "ultrabook", "macbook"],
            "headphones": ["headphone", "headphones", "earphone", "anc", "headset"],
            "monitor": ["monitor", "display", "screen", "4k"],
            "smartphone": ["phone", "smartphone", "mobile", "5g"],
        }
        for cat, keywords in category_map.items():
            if any(kw in q for kw in keywords):
                category = cat
                break

        # Budget detection (handles "50k", "50,000", "₹50000", "under 50k", "below 15000")
        max_budget: Optional[float] = None
        budget_patterns = [
            r"(?:under|below|within|max|budget|upto|up to|less than)\s*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand)?",
            r"(?:rs\.?|inr|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand)?",
            r"(\d+(?:\.\d+)?)\s*k\b",
        ]
        for pat in budget_patterns:
            m = re.search(pat, q)
            if m:
                num_str = m.group(1).replace(",", "")
                val = float(num_str)
                suffix = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                if suffix in ("k", "thousand") or "k" in m.group(0).split()[-1]:
                    if val < 1000:
                        val *= 1000
                max_budget = val
                break

        # RAM detection
        min_ram_gb: Optional[int] = None
        ram_match = re.search(r"(\d+)\s*gb\s*(?:ram|memory)?", q)
        if ram_match:
            val = int(ram_match.group(1))
            if val in (4, 8, 12, 16, 24, 32, 64):
                min_ram_gb = val

        # Storage detection
        min_storage_gb: Optional[int] = None
        tb_match = re.search(r"(\d+)\s*tb\s*(?:ssd|hdd|storage)?", q)
        ssd_match = re.search(r"(\d+)\s*gb\s*(?:ssd|nvme|storage|rom)", q)
        if tb_match:
            min_storage_gb = int(tb_match.group(1)) * 1024
        elif ssd_match:
            min_storage_gb = int(ssd_match.group(1))

        # Processor preference
        processor_pref = None
        for proc in ["i7", "i5", "i3", "ryzen 7", "ryzen 5", "snapdragon"]:
            if proc in q:
                processor_pref = proc
                break

        # Use case detection
        use_case = None
        for uc in ["coding", "programming", "developer", "gaming", "office", "student", "travel", "design", "editing"]:
            if uc in q:
                use_case = "coding" if uc in ("programming", "developer") else uc
                break

        # Brand preference
        brands = ["lenovo", "asus", "hp", "dell", "acer", "msi", "sony", "jbl", "lg", "oneplus"]
        preferred_brands = [b.capitalize() if b not in ("hp", "lg", "msi") else b.upper() for b in brands if b in q]

        # Delivery constraint
        max_delivery_days = None
        if "express" in q or "fast delivery" in q or "tomorrow" in q or "1 day" in q:
            max_delivery_days = 1
        elif "2 days" in q:
            max_delivery_days = 2

        # Must-have keywords
        must_haves = []
        for kw in ["ips", "anc", "4k", "usb-c", "amoled", "gpu", "rtx", "ms office", "144hz", "lightweight"]:
            if kw in q:
                must_haves.append(kw)

        return UserRequirement(
            raw_query=query,
            category=category,
            max_budget_inr=max_budget,
            min_ram_gb=min_ram_gb,
            min_storage_gb=min_storage_gb,
            processor_preference=processor_pref,
            use_case=use_case,
            must_have_keywords=must_haves,
            preferred_brands=preferred_brands,
            max_delivery_days=max_delivery_days,
        )

    # 2. Catalog Search Tool
    def search_catalog(self, req: UserRequirement) -> List[Product]:
        results: List[Product] = []
        for p in self.catalog:
            if not p.in_stock or p.stock_count <= 0:
                continue
            if req.category and p.category != req.category:
                continue
            if req.max_budget_inr is not None and p.price_inr > req.max_budget_inr:
                continue
            if req.min_budget_inr is not None and p.price_inr < req.min_budget_inr:
                continue
            if req.min_rating and p.rating < req.min_rating:
                continue
            if req.max_delivery_days is not None and p.delivery_days > req.max_delivery_days:
                continue
            results.append(p)
        return results

    # 3. Best Coupon Finder Tool
    def find_best_coupon(self, product: Product, quantity: int = 1) -> Tuple[Optional[Coupon], float]:
        subtotal = product.price_inr * quantity
        best_coupon: Optional[Coupon] = None
        best_discount = 0.0
        for c in self.coupons:
            disc = calculate_coupon_discount(c, subtotal, category=product.category)
            if disc > best_discount:
                best_discount = disc
                best_coupon = c
        return best_coupon, best_discount

    # 4. Evaluation & Scoring Tool
    def evaluate_and_rank(self, candidates: List[Product], req: UserRequirement) -> List[ScoredProduct]:
        scored: List[ScoredProduct] = []
        for p in candidates:
            score = 50.0
            matched: List[str] = []
            tradeoffs: List[str] = []

            # RAM check
            p_ram = p.specs.get("ram_gb")
            if req.min_ram_gb and p_ram is not None:
                if p_ram >= req.min_ram_gb:
                    score += 15.0
                    matched.append(f"{p_ram}GB RAM meets {req.min_ram_gb}GB+ requirement")
                else:
                    score -= 20.0
                    tradeoffs.append(f"Only {p_ram}GB RAM (requested {req.min_ram_gb}GB)")
            elif p_ram and p_ram >= 16:
                score += 8.0
                matched.append(f"{p_ram}GB RAM")

            # Storage check
            p_storage = p.specs.get("storage_gb")
            if req.min_storage_gb and p_storage is not None:
                if p_storage >= req.min_storage_gb:
                    score += 10.0
                    matched.append(f"{p_storage}GB SSD meets requirement")
                else:
                    score -= 15.0
                    tradeoffs.append(f"Storage {p_storage}GB < {req.min_storage_gb}GB")

            # Processor check
            proc_str = str(p.specs.get("processor", "")).lower()
            if req.processor_preference:
                if req.processor_preference in proc_str:
                    score += 10.0
                    matched.append(f"Matches processor ({p.specs.get('processor')})")
                else:
                    tradeoffs.append(f"Uses {p.specs.get('processor')} instead of {req.processor_preference}")

            # Use case / tags
            if req.use_case:
                if req.use_case in p.tags:
                    score += 8.0
                    matched.append(f"Suited for {req.use_case}")
                else:
                    tradeoffs.append(f"Not specifically tuned for {req.use_case}")

            # Must-have keywords
            spec_blob = " ".join(str(v).lower() for v in p.specs.values()) + " " + " ".join(p.tags)
            for kw in req.must_have_keywords:
                if kw in spec_blob:
                    score += 6.0
                    matched.append(f"Has {kw.upper()}")
                else:
                    score -= 8.0
                    tradeoffs.append(f"Missing {kw.upper()}")

            # Brand preference
            if req.preferred_brands:
                if p.brand in req.preferred_brands:
                    score += 7.0
                    matched.append(f"Preferred brand ({p.brand})")

            # Rating & seller bonus
            score += (p.rating - 4.0) * 10.0
            if p.seller_rating >= 4.8:
                score += 3.0
                matched.append(f"Trusted seller ({p.seller_rating}/5)")

            # Delivery speed bonus
            if p.delivery_days <= 2:
                score += 2.0
                matched.append(f"Fast {p.delivery_days}-day delivery")

            # Coupon optimization
            best_coupon, discount = self.find_best_coupon(p, quantity=1)
            effective_price = round(p.price_inr - discount, 2)
            if req.max_budget_inr and effective_price <= req.max_budget_inr:
                headroom_pct = (req.max_budget_inr - effective_price) / req.max_budget_inr
                if headroom_pct >= 0.05:
                    score += 3.0

            # Display / battery trade-off hints
            if "TN" in str(p.specs.get("display", "")):
                tradeoffs.append("TN display (narrower viewing angles vs IPS)")

            final_score = max(0.0, min(100.0, round(score, 1)))
            scored.append(
                ScoredProduct(
                    product=p,
                    match_score=final_score,
                    matched_criteria=matched,
                    tradeoffs=tradeoffs,
                    effective_price_inr=effective_price,
                    best_coupon_code=best_coupon.code if best_coupon else None,
                )
            )

        scored.sort(key=lambda x: (-x.match_score, x.effective_price_inr or x.product.price_inr))
        return scored

    # 5. Cart Management Tool
    def add_to_cart(self, product_id: str, quantity: int = 1, auto_apply_coupon: bool = True) -> Cart:
        product = self.get_product(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found.")
        if not product.in_stock or product.stock_count < quantity:
            raise ValueError(f"Insufficient stock for {product.title}.")

        # Update existing item or append
        existing = next((i for i in self.cart.items if i.product.id == product_id), None)
        if existing:
            existing.quantity += quantity
        else:
            self.cart.items.append(CartItem(product=product, quantity=quantity))

        if auto_apply_coupon:
            best_coupon, discount = self.find_best_coupon(product, quantity=self.cart.items[0].quantity)
            # Re-evaluate against whole cart subtotal
            best_cart_coupon: Optional[Coupon] = None
            best_cart_disc = 0.0
            for c in self.coupons:
                d = calculate_coupon_discount(c, self.cart.subtotal_inr, category=product.category)
                if d > best_cart_disc:
                    best_cart_disc = d
                    best_cart_coupon = c
            self.cart.applied_coupon = best_cart_coupon
            self.cart.discount_inr = best_cart_disc

        return self.cart

    def clear_cart(self) -> None:
        self.cart = Cart()

    # 6. HITL Checkout Approval Guardrail Tool
    def create_checkout_approval(
        self,
        product_id: str,
        shipping_address: str,
        payment_method: str,
        rationale: str,
        quantity: int = 1,
    ) -> PendingApproval:
        product = self.get_product(product_id)
        if not product:
            raise ValueError(f"Product {product_id} not found.")

        self.clear_cart()
        self.add_to_cart(product_id, quantity=quantity, auto_apply_coupon=True)

        token = f"APR-{uuid.uuid4().hex[:8].upper()}"
        approval = PendingApproval(
            approval_token=token,
            recommended_product=product,
            quantity=quantity,
            subtotal_inr=self.cart.subtotal_inr,
            coupon_code=self.cart.applied_coupon.code if self.cart.applied_coupon else None,
            discount_inr=self.cart.discount_inr,
            final_total_inr=self.cart.total_inr,
            shipping_address=shipping_address,
            payment_method=payment_method,
            rationale=rationale,
        )
        self.pending_approvals[token] = approval
        return approval

    # 7. Order Execution Tool (Requires Approval Token)
    def place_order(self, approval_token: str) -> Order:
        approval = self.pending_approvals.pop(approval_token, None)
        if not approval:
            raise ValueError("Invalid or expired approval token. Human approval is required before placing an order.")

        product = self.get_product(approval.recommended_product.id)
        if not product or product.stock_count < approval.quantity:
            raise ValueError("Product is no longer in stock.")

        product.stock_count -= approval.quantity
        if product.stock_count <= 0:
            product.in_stock = False

        order = Order(
            order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
            items=list(self.cart.items) if self.cart.items else [CartItem(product=product, quantity=approval.quantity)],
            subtotal_inr=approval.subtotal_inr,
            coupon_code=approval.coupon_code,
            discount_inr=approval.discount_inr,
            gst_component_inr=self.cart.gst_inr,
            total_inr=approval.final_total_inr,
            shipping_address=approval.shipping_address,
            payment_method=approval.payment_method,
            status=OrderStatus.CONFIRMED,
            estimated_delivery_days=product.delivery_days,
        )
        self.orders.insert(0, order)
        self.clear_cart()
        return order
