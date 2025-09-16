from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CompanyInfo(BaseModel):
    name: str
    description: str
    website: str
    pricing_model: Optional[str] = "Unknown"
    integration_capabilities: Optional[str] = "Unknown"


class ResearchState(BaseModel):
    query: str
    extracted_tools: List[str] = Field(default_factory=list)
    companies: List[CompanyInfo] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    analysis: Optional[str] = None
