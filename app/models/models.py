from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class CalculationRequest(BaseModel):
    price: List[float]
    support_cost: List[float]
    daily_orders: List[List[float]]
    n_days: int

class CombinedMethodResponse(BaseModel):
    service_allocation: List[Dict[str, Any]]
    provider_analysis: List[Dict[str, Any]]
    financial_summary: Dict[str, float]
    pareto_front: List[List[float]]