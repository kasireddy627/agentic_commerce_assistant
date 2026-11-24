from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentPhase(str, Enum):
    IDLE = "IDLE"
    AWAITING_REQUIREMENT_APPROVAL = "AWAITING_REQUIREMENT_APPROVAL"
    AWAITING_DISCOVERY_APPROVAL = "AWAITING_DISCOVERY_APPROVAL"
    AWAITING_SELECTION_APPROVAL = "AWAITING_SELECTION_APPROVAL"
    AWAITING_CHECKOUT_APPROVAL = "AWAITING_CHECKOUT_APPROVAL"
    ORDER_PLACED = "ORDER_PLACED"


class OrderStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    PROCESSING = "PROCESSING"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


class UserRequirement(BaseModel):
    raw_query: str
    category: Optional[str] = None
    max_budget_inr: Optional[float] = None
    min_budget_inr: Optional[float] = None
    min_ram_gb: Optional[int] = None
    min_storage_gb: Optional[int] = None
    processor_preference: Optional[str] = None
    use_case: Optional[str] = None
    must_have_keywords: List[str] = Field(default_factory=list)
    preferred_brands: List[str] = Field(default_factory=list)
    min_rating: float = 4.0
    max_delivery_days: Optional[int] = None


class Product(BaseModel):
    id: str
    title: str
    brand: str
    category: str
    price_inr: float
    original_price_inr: float
    rating: float
    review_count: int
    seller: str
    seller_rating: float
    in_stock: bool
    stock_count: int
    delivery_days: int
    warranty: str
    return_policy_days: int
    specs: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list)
    review_summary: str = ""
    tags: List[str] = Field(default_factory=list)

    @property
    def discount_percent(self) -> float:
        if self.original_price_inr <= 0:
            return 0.0
        return round(((self.original_price_inr - self.price_inr) / self.original_price_inr) * 100, 1)


class ScoredProduct(BaseModel):
    product: Product
    match_score: float = Field(..., ge=0.0, le=100.0)
    matched_criteria: List[str] = Field(default_factory=list)
    tradeoffs: List[str] = Field(default_factory=list)
    effective_price_inr: Optional[float] = None
    best_coupon_code: Optional[str] = None


class Coupon(BaseModel):
    code: str
    description: str
    discount_type: str  # "flat" or "percent"
    discount_value: float
    min_order_inr: float = 0.0
    max_discount_inr: Optional[float] = None
    applicable_categories: List[str] = Field(default_factory=list)


class CartItem(BaseModel):
    product: Product
    quantity: int = Field(default=1, ge=1)

    @property
    def line_total_inr(self) -> float:
        return round(self.product.price_inr * self.quantity, 2)


class Cart(BaseModel):
    items: List[CartItem] = Field(default_factory=list)
    applied_coupon: Optional[Coupon] = None
    discount_inr: float = 0.0

    @property
    def subtotal_inr(self) -> float:
        return round(sum(item.line_total_inr for item in self.items), 2)

    @property
    def gst_inr(self) -> float:
        taxable = max(self.subtotal_inr - self.discount_inr, 0.0)
        return round(taxable - (taxable / 1.18), 2)

    @property
    def total_inr(self) -> float:
        return round(max(self.subtotal_inr - self.discount_inr, 0.0), 2)


class PendingApproval(BaseModel):
    approval_token: str
    recommended_product: Product
    quantity: int = 1
    subtotal_inr: float
    coupon_code: Optional[str] = None
    discount_inr: float = 0.0
    final_total_inr: float
    shipping_address: str
    payment_method: str
    rationale: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class Order(BaseModel):
    order_id: str
    items: List[CartItem]
    subtotal_inr: float
    coupon_code: Optional[str] = None
    discount_inr: float = 0.0
    gst_component_inr: float = 0.0
    total_inr: float
    shipping_address: str
    payment_method: str
    status: OrderStatus = OrderStatus.CONFIRMED
    estimated_delivery_days: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ToolCallTrace(BaseModel):
    step: int
    phase: AgentPhase
    thought: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_output_summary: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class AgentResponse(BaseModel):
    message: str
    phase: AgentPhase
    requirements: Optional[UserRequirement] = None
    discovered_products: List[Product] = Field(default_factory=list)
    shortlisted: List[ScoredProduct] = Field(default_factory=list)
    traces: List[ToolCallTrace] = Field(default_factory=list)
    pending_approval: Optional[PendingApproval] = None
    placed_order: Optional[Order] = None
