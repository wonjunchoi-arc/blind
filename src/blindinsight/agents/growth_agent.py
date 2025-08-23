"""
Growth & Stability Analysis Agent for BlindInsight AI.

Specializes in analyzing company growth prospects, financial stability,
and market position based on business intelligence and employee insights.
"""

import statistics
from typing import Any, Dict, List, Optional

from ..models.analysis import GrowthReport
from ..models.user import UserProfile
from .base import BaseAgent, AgentConfig, AgentResult


class GrowthStabilityAgent(BaseAgent):
    """
    Agent specialized in company growth and stability analysis.
    
    This agent analyzes growth by combining:
    - RAG-retrieved business intelligence and employee perspectives
    - External financial and market data
    - Industry trends and competitive analysis
    - Risk assessment and stability indicators
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__("growth_stability", config)
        
        # Growth-related keywords
        self.growth_keywords = {
            "positive": [
                "성장", "확장", "투자", "증가", "발전", "혁신", "신사업", 
                "글로벌", "시장점유율", "매출증가", "흑자", "수익성"
            ],
            "negative": [
                "감소", "축소", "구조조정", "적자", "손실", "위기", "불안정",
                "경영난", "부채", "리스크", "경쟁력하락", "시장축소"
            ],
            "stability": [
                "안정", "견고", "신뢰", "지속", "일관", "예측가능", "체계적",
                "시스템", "프로세스", "장기계획", "지속가능", "기반"
            ]
        }
        
        # Risk factors
        self.risk_indicators = [
            "구조조정", "정리해고", "사업철수", "적자", "부채",
            "법적분쟁", "규제위험", "경쟁심화", "시장변화"
        ]
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """Execute growth and stability analysis."""
        
        company = context.get("company") if context else query
        
        return await self._execute_with_error_handling(
            self.analyze_growth_stability,
            company,
            user_profile
        )
    
    async def analyze_growth_stability(
        self,
        company: str,
        user_profile: Optional[UserProfile] = None
    ) -> GrowthReport:
        """Analyze company growth prospects and stability."""
        
        # Retrieve growth-related documents
        documents = await self._retrieve_growth_documents(company)
        
        # Fetch external business data
        external_data = await self._fetch_external_business_data(company)
        
        # Analyze growth indicators
        growth_analysis = self._analyze_growth_indicators(documents, external_data)
        
        # Analyze stability indicators
        stability_analysis = self._analyze_stability_indicators(documents, external_data)
        
        # Assess risks
        risk_assessment = self._assess_risks(documents, external_data)
        
        # Calculate scores
        scores = self._calculate_growth_scores(growth_analysis, stability_analysis, risk_assessment)
        
        # Generate insights
        insights = self._generate_growth_insights(growth_analysis, stability_analysis, risk_assessment)
        
        # Calculate confidence
        confidence = self._calculate_confidence_score(documents, external_data)
        
        return GrowthReport(
            growth_score=scores["growth_score"],
            stability_score=scores["stability_score"],
            innovation_score=scores["innovation_score"],
            market_position_score=scores["market_position_score"],
            market_position=insights["market_position"],
            financial_health=insights["financial_health"],
            future_outlook=insights["future_outlook"],
            risk_factors=risk_assessment["factors"],
            risk_level=risk_assessment["level"],
            stability_indicators=insights["stability_indicators"],
            competitive_advantages=insights["advantages"],
            industry_challenges=insights["challenges"],
            data_points_count=len(documents),
            confidence_score=confidence
        )
    
    async def _retrieve_growth_documents(self, company: str) -> List[Dict[str, Any]]:
        """Retrieve growth-related documents."""
        
        queries = [
            f"{company} 성장 전망",
            f"{company} 사업 확장",
            f"{company} 재무 상태",
            f"{company} 시장 위치",
            f"{company} 경쟁력",
            f"{company} 미래 계획"
        ]
        
        all_documents = []
        
        for query in queries:
            documents = await self.retrieve_knowledge(
                query=query,
                category="business",
                k=3
            )
            
            for doc in documents:
                all_documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "query": query
                })
        
        return all_documents
    
    async def _fetch_external_business_data(self, company: str) -> Dict[str, Any]:
        """Fetch external business and financial data."""
        
        external_data = {}
        
        try:
            # Financial data
            financial_data = await self.call_mcp_service(
                "api_integration",
                "get_company_financials",
                {"company": company}
            )
            
            if financial_data:
                external_data["financial"] = financial_data
        
        except Exception as e:
            print(f"Failed to fetch financial data: {e}")
        
        try:
            # News and market data
            news_data = await self.call_mcp_service(
                "web_scraping",
                "scrape_company_news",
                {"company": company, "category": "business"}
            )
            
            if news_data and news_data.get("status") == "success":
                external_data["news"] = news_data.get("data", [])
        
        except Exception as e:
            print(f"Failed to fetch news data: {e}")
        
        return external_data
    
    def _analyze_growth_indicators(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze growth indicators from available data."""
        
        all_content = ""
        for doc in documents:
            all_content += doc.get("content", "") + " "
        
        # Add external content
        for source_data in external_data.values():
            if isinstance(source_data, list):
                for item in source_data:
                    if isinstance(item, dict):
                        all_content += item.get("content", "") + " "
        
        # Count growth-related keywords
        positive_growth = sum(1 for keyword in self.growth_keywords["positive"] if keyword in all_content)
        negative_growth = sum(1 for keyword in self.growth_keywords["negative"] if keyword in all_content)
        
        # Calculate growth sentiment
        total_growth_mentions = positive_growth + negative_growth
        growth_sentiment = 0.0
        if total_growth_mentions > 0:
            growth_sentiment = (positive_growth - negative_growth) / total_growth_mentions
        
        return {
            "positive_indicators": positive_growth,
            "negative_indicators": negative_growth,
            "growth_sentiment": growth_sentiment,
            "total_mentions": total_growth_mentions
        }
    
    def _analyze_stability_indicators(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze stability indicators."""
        
        all_content = ""
        for doc in documents:
            all_content += doc.get("content", "") + " "
        
        # Count stability keywords
        stability_mentions = sum(1 for keyword in self.growth_keywords["stability"] if keyword in all_content)
        
        # Look for specific stability indicators
        indicators = []
        if "장기계획" in all_content or "5년계획" in all_content:
            indicators.append("체계적 장기 계획")
        if "시스템" in all_content and "프로세스" in all_content:
            indicators.append("안정적 시스템 및 프로세스")
        if "지속가능" in all_content:
            indicators.append("지속가능 경영")
        
        return {
            "stability_mentions": stability_mentions,
            "stability_indicators": indicators,
            "stability_score": min(1.0, stability_mentions / 10)  # Normalize to 0-1
        }
    
    def _assess_risks(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess various risk factors."""
        
        all_content = ""
        for doc in documents:
            all_content += doc.get("content", "") + " "
        
        # Identify risk factors
        identified_risks = []
        for risk in self.risk_indicators:
            if risk in all_content:
                identified_risks.append(risk)
        
        # Calculate risk level
        risk_count = len(identified_risks)
        if risk_count == 0:
            risk_level = "low"
        elif risk_count <= 2:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "factors": identified_risks,
            "level": risk_level,
            "risk_score": min(1.0, risk_count / 5)  # Normalize to 0-1
        }
    
    def _calculate_growth_scores(
        self, 
        growth_analysis: Dict[str, Any], 
        stability_analysis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate various growth and stability scores."""
        
        # Base growth score from sentiment
        base_growth = max(0, min(100, (growth_analysis["growth_sentiment"] + 1) * 50))
        
        # Adjust for mentions and data quality
        mention_factor = min(1.2, 1 + (growth_analysis["total_mentions"] / 20))
        growth_score = base_growth * mention_factor
        
        # Stability score
        stability_score = stability_analysis["stability_score"] * 100
        
        # Innovation score (simplified based on growth indicators)
        innovation_score = min(100, growth_score * 0.8 + stability_score * 0.2)
        
        # Market position score (based on competitive keywords)
        market_position_score = min(100, (growth_score + stability_score) / 2)
        
        # Adjust all scores based on risk
        risk_penalty = risk_assessment["risk_score"] * 20
        
        return {
            "growth_score": max(0, min(100, growth_score - risk_penalty)),
            "stability_score": max(0, min(100, stability_score - risk_penalty)),
            "innovation_score": max(0, min(100, innovation_score - risk_penalty)),
            "market_position_score": max(0, min(100, market_position_score - risk_penalty))
        }
    
    def _generate_growth_insights(
        self, 
        growth_analysis: Dict[str, Any], 
        stability_analysis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate qualitative insights about growth and stability."""
        
        # Market position assessment
        if growth_analysis["growth_sentiment"] > 0.3:
            market_position = "선도기업"
        elif growth_analysis["growth_sentiment"] > 0:
            market_position = "안정적 성장"
        elif growth_analysis["growth_sentiment"] > -0.3:
            market_position = "보통"
        else:
            market_position = "도전적 상황"
        
        # Financial health assessment
        if risk_assessment["risk_score"] < 0.2:
            financial_health = "건전"
        elif risk_assessment["risk_score"] < 0.5:
            financial_health = "보통"
        else:
            financial_health = "주의필요"
        
        # Future outlook
        if growth_analysis["growth_sentiment"] > 0.2 and stability_analysis["stability_score"] > 0.5:
            future_outlook = "긍정적"
        elif growth_analysis["growth_sentiment"] > -0.2:
            future_outlook = "중립적"
        else:
            future_outlook = "부정적"
        
        return {
            "market_position": market_position,
            "financial_health": financial_health,
            "future_outlook": future_outlook,
            "stability_indicators": stability_analysis["stability_indicators"],
            "advantages": ["시장 인지도", "기술력"] if growth_analysis["positive_indicators"] > 3 else ["안정성"],
            "challenges": ["시장 경쟁 심화"] if growth_analysis["negative_indicators"] > 2 else []
        }
    
    def _calculate_confidence_score(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for growth analysis."""
        
        confidence_factors = []
        
        # Document availability
        doc_factor = min(1.0, len(documents) / 15)  # Optimal at 15+ documents
        confidence_factors.append(doc_factor)
        
        # External data availability
        external_factor = 0.8 if external_data else 0.4
        confidence_factors.append(external_factor)
        
        # Content quality (based on business-related content)
        if documents:
            business_keywords = self.growth_keywords["positive"] + self.growth_keywords["negative"] + self.growth_keywords["stability"]
            total_content = " ".join([doc.get("content", "") for doc in documents])
            keyword_density = sum(1 for keyword in business_keywords if keyword in total_content)
            quality_factor = min(1.0, keyword_density / 10)  # Optimal at 10+ keywords
            confidence_factors.append(quality_factor)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5