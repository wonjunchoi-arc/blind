"""
Compensation Analysis Agent for BlindInsight AI.

Specializes in analyzing salary, benefits, and compensation packages
based on employee reports and market data.
"""

import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..models.analysis import CompensationReport
from ..models.user import UserProfile
from .base import BaseAgent, AgentConfig, AgentResult


class CompensationAnalysisAgent(BaseAgent):
    """
    Agent specialized in compensation and benefits analysis.
    
    This agent analyzes compensation by combining:
    - RAG-retrieved salary information and discussions
    - External market data from job sites
    - Benefits analysis and comparison
    - Career progression and salary growth trends
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__("compensation_analysis", config)
        
        # Compensation-related keywords
        self.salary_keywords = [
            "연봉", "급여", "임금", "보너스", "인센티브", "스톡옵션",
            "복지", "휴가", "보험", "퇴직금", "수당"
        ]
        
        # Industry salary benchmarks (example data)
        self.industry_benchmarks = {
            "it_software": {
                "junior": (3500, 5500),
                "mid": (5500, 8000),  
                "senior": (8000, 12000),
                "lead": (12000, 18000)
            }
        }
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """Execute compensation analysis."""
        
        company = context.get("company") if context else query
        position = context.get("position") if context else None
        
        return await self._execute_with_error_handling(
            self.analyze_compensation,
            company,
            position,
            user_profile
        )
    
    async def analyze_compensation(
        self,
        company: str,
        position: Optional[str] = None,
        user_profile: Optional[UserProfile] = None
    ) -> CompensationReport:
        """Analyze compensation and benefits for a company."""
        
        # Retrieve compensation-related documents
        documents = await self._retrieve_compensation_documents(company, position)
        
        # Fetch external compensation data
        external_data = await self._fetch_external_compensation_data(company)
        
        # Extract salary information
        salary_data = self._extract_salary_data(documents, external_data)
        
        # Calculate compensation metrics
        salary_range = self._calculate_salary_range(salary_data)
        median_salary = self._calculate_median_salary(salary_data)
        percentile = self._calculate_salary_percentile(salary_data, user_profile)
        
        # Analyze benefits
        benefits_info = self._analyze_benefits(documents, external_data)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(documents, external_data, salary_data)
        
        return CompensationReport(
            salary_range=salary_range,
            median_salary=median_salary,
            salary_percentile=percentile,
            bonus_info=benefits_info.get("bonus", {}),
            stock_options=benefits_info.get("stock_options"),
            benefits_score=benefits_info.get("score", 50.0),
            total_compensation_estimate=int(median_salary * 1.2),  # Include benefits estimate
            vs_market_average=self._compare_to_market(salary_range, position),
            data_points_count=len(salary_data),
            confidence_score=confidence
        )
    
    async def _retrieve_compensation_documents(
        self, 
        company: str, 
        position: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve compensation-related documents."""
        
        queries = [
            f"{company} 연봉 정보",
            f"{company} 급여 후기",
            f"{company} 복지 혜택"
        ]
        
        if position:
            queries.append(f"{company} {position} 연봉")
        
        all_documents = []
        
        for query in queries:
            documents = await self.retrieve_knowledge(
                query=query,
                category="salary",
                k=5
            )
            
            for doc in documents:
                all_documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "query": query
                })
        
        return all_documents
    
    async def _fetch_external_compensation_data(self, company: str) -> Dict[str, Any]:
        """Fetch external compensation data."""
        
        external_data = {}
        
        try:
            # Salary information from job sites
            salary_data = await self.call_mcp_service(
                "api_integration",
                "get_salary_data",
                {"company": company}
            )
            
            if salary_data:
                external_data["salary_sites"] = salary_data
        
        except Exception as e:
            print(f"Failed to fetch external salary data: {e}")
        
        return external_data
    
    def _extract_salary_data(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract salary data from documents and external sources."""
        
        salary_data = []
        
        # Extract from documents
        for doc in documents:
            content = doc.get("content", "")
            
            # Simple regex-like extraction for Korean salary mentions
            # This is a simplified implementation
            if "만원" in content or "연봉" in content:
                # Extract numeric values (simplified)
                import re
                numbers = re.findall(r'(\d{3,5})', content)
                
                for num_str in numbers:
                    try:
                        salary = int(num_str)
                        if 2000 <= salary <= 50000:  # Reasonable salary range
                            salary_data.append({
                                "salary": salary,
                                "source": "internal_document",
                                "metadata": doc.get("metadata", {})
                            })
                    except ValueError:
                        continue
        
        # Extract from external data
        for source, data in external_data.items():
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "salary" in item:
                        salary_data.append({
                            "salary": item["salary"],
                            "source": source,
                            "metadata": item
                        })
        
        return salary_data
    
    def _calculate_salary_range(self, salary_data: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Calculate salary range from data."""
        
        if not salary_data:
            return (4000, 8000)  # Default range
        
        salaries = [item["salary"] for item in salary_data]
        
        if len(salaries) == 1:
            salary = salaries[0]
            return (int(salary * 0.9), int(salary * 1.1))
        
        # Use percentiles for range
        p25 = int(statistics.quantiles(salaries, n=4)[0])
        p75 = int(statistics.quantiles(salaries, n=4)[2])
        
        return (p25, p75)
    
    def _calculate_median_salary(self, salary_data: List[Dict[str, Any]]) -> int:
        """Calculate median salary."""
        
        if not salary_data:
            return 6000  # Default median
        
        salaries = [item["salary"] for item in salary_data]
        return int(statistics.median(salaries))
    
    def _calculate_salary_percentile(
        self, 
        salary_data: List[Dict[str, Any]], 
        user_profile: Optional[UserProfile]
    ) -> float:
        """Calculate salary percentile ranking."""
        
        if not salary_data:
            return 50.0
        
        salaries = [item["salary"] for item in salary_data]
        median_salary = statistics.median(salaries)
        
        # Calculate percentile based on median
        sorted_salaries = sorted(salaries)
        position = next((i for i, s in enumerate(sorted_salaries) if s >= median_salary), len(sorted_salaries))
        
        return (position / len(sorted_salaries)) * 100
    
    def _analyze_benefits(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze benefits and perks."""
        
        benefits_keywords = {
            "health": ["건강보험", "의료", "건강검진"],
            "vacation": ["휴가", "연차", "월차"],
            "bonus": ["보너스", "성과급", "인센티브"],
            "stock": ["스톡옵션", "주식", "RSU"],
            "education": ["교육비", "학습", "세미나"],
            "meal": ["식비", "점심", "저녁", "간식"]
        }
        
        benefits_score = 50.0  # Base score
        found_benefits = []
        
        all_content = ""
        for doc in documents:
            all_content += doc.get("content", "") + " "
        
        for category, keywords in benefits_keywords.items():
            for keyword in keywords:
                if keyword in all_content:
                    found_benefits.append(category)
                    benefits_score += 5
                    break
        
        # Cap benefits score
        benefits_score = min(100.0, benefits_score)
        
        return {
            "score": benefits_score,
            "categories": list(set(found_benefits)),
            "bonus": {"available": "보너스" in all_content},
            "stock_options": "스톡옵션" in all_content or "주식" in all_content
        }
    
    def _compare_to_market(
        self, 
        salary_range: Tuple[int, int], 
        position: Optional[str]
    ) -> float:
        """Compare salary to market average."""
        
        # Simplified market comparison
        market_avg = 6000  # Default market average
        
        if position:
            # Try to get industry benchmark
            level = self._infer_level_from_position(position)
            benchmark = self.industry_benchmarks.get("it_software", {}).get(level)
            
            if benchmark:
                market_avg = (benchmark[0] + benchmark[1]) / 2
        
        company_avg = (salary_range[0] + salary_range[1]) / 2
        
        return ((company_avg - market_avg) / market_avg) * 100
    
    def _infer_level_from_position(self, position: str) -> str:
        """Infer career level from position title."""
        
        position_lower = position.lower()
        
        if any(word in position_lower for word in ["주니어", "junior", "신입"]):
            return "junior"
        elif any(word in position_lower for word in ["시니어", "senior", "수석"]):
            return "senior"
        elif any(word in position_lower for word in ["리드", "lead", "팀장"]):
            return "lead"
        else:
            return "mid"
    
    def _calculate_confidence_score(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any],
        salary_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for compensation analysis."""
        
        confidence_factors = []
        
        # Data availability factor
        data_factor = min(1.0, len(salary_data) / 10)  # Optimal at 10+ data points
        confidence_factors.append(data_factor)
        
        # Source diversity factor
        sources = set(item.get("source", "unknown") for item in salary_data)
        source_factor = min(1.0, len(sources) / 3)  # Optimal at 3+ sources
        confidence_factors.append(source_factor)
        
        # External data factor
        external_factor = 0.8 if external_data else 0.4
        confidence_factors.append(external_factor)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5