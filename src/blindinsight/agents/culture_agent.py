"""
Culture Analysis Agent for BlindInsight AI.

Specializes in analyzing company culture, work-life balance,
and organizational climate based on employee feedback.
"""

import json
import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..models.analysis import CultureReport
from ..models.user import UserProfile
from .base import BaseAgent, AgentConfig, AgentResult


class CultureAnalysisAgent(BaseAgent):
    """
    Agent specialized in company culture analysis.
    
    This agent analyzes company culture by combining:
    - RAG-retrieved employee reviews and posts
    - External data from job review sites
    - Sentiment analysis and trend detection
    - Comparative analysis with industry benchmarks
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__("culture_analysis", config)
        
        # Culture-specific configuration
        self.culture_keywords = {
            "positive": [
                "좋은", "훌륭한", "만족", "추천", "성장", "자유로운", "수평적", 
                "혁신적", "도전적", "개방적", "소통", "배려", "복지", "워라밸"
            ],
            "negative": [
                "나쁜", "최악", "불만", "스트레스", "야근", "수직적", "경직된",
                "보수적", "폐쇄적", "갑질", "차별", "과로", "압박", "번아웃"
            ]
        }
        
        # Culture scoring weights
        self.scoring_weights = {
            "work_life_balance": 0.25,
            "growth_opportunity": 0.20,
            "management_quality": 0.20,
            "colleague_relationship": 0.15,
            "work_environment": 0.10,
            "company_stability": 0.10
        }
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """Execute culture analysis."""
        
        # Extract company from query or context
        company = context.get("company") if context else query
        position = context.get("position") if context else None
        
        return await self._execute_with_error_handling(
            self.analyze_company_culture,
            company,
            position,
            user_profile
        )
    
    async def analyze_company_culture(
        self,
        company: str,
        position: Optional[str] = None,
        user_profile: Optional[UserProfile] = None
    ) -> CultureReport:
        """
        Analyze company culture comprehensively.
        
        TODO(human): Implement the core culture analysis logic that:
        1. Retrieves culture-related documents from RAG system
        2. Fetches external data via MCP services (Glassdoor, JobPlanet)
        3. Performs sentiment analysis on collected data
        4. Calculates culture scores and metrics
        5. Generates insights, pros/cons summaries, and keywords
        6. Returns a complete CultureReport with confidence scoring
        
        Args:
            company: Company name to analyze
            position: Optional specific position
            user_profile: Optional user profile for personalized insights
            
        Returns:
            CultureReport with comprehensive culture analysis
        """
        
        # TODO(human): Implement the complete culture analysis method
        pass
    
    async def _retrieve_culture_documents(self, company: str) -> List[Dict[str, Any]]:
        """Retrieve culture-related documents from RAG system."""
        
        # Define culture-related search queries
        queries = [
            f"{company} 기업문화 리뷰",
            f"{company} 워라밸 후기",
            f"{company} 회사 분위기",
            f"{company} 조직문화 평가"
        ]
        
        all_documents = []
        
        for query in queries:
            documents = await self.retrieve_knowledge(
                query=query,
                category="culture",
                k=5
            )
            
            for doc in documents:
                all_documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "query": query
                })
        
        return all_documents
    
    async def _fetch_external_culture_data(self, company: str) -> Dict[str, Any]:
        """Fetch external culture data via MCP services."""
        
        external_data = {}
        
        try:
            # Glassdoor data
            glassdoor_data = await self.call_mcp_service(
                "web_scraping",
                "scrape_glassdoor",
                {"company": company, "category": "culture"}
            )
            
            if glassdoor_data and glassdoor_data.get("status") == "success":
                external_data["glassdoor"] = glassdoor_data.get("data", [])
        
        except Exception as e:
            print(f"Failed to fetch Glassdoor data: {e}")
        
        try:
            # JobPlanet data
            jobplanet_data = await self.call_mcp_service(
                "web_scraping", 
                "scrape_jobplanet",
                {"company": company}
            )
            
            if jobplanet_data and jobplanet_data.get("status") == "success":
                external_data["jobplanet"] = jobplanet_data.get("data", [])
        
        except Exception as e:
            print(f"Failed to fetch JobPlanet data: {e}")
        
        return external_data
    
    def _analyze_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Analyze sentiment of text data."""
        
        if not texts:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "overall": 0.0}
        
        sentiment_scores = []
        
        for text in texts:
            # Simple keyword-based sentiment analysis
            positive_count = sum(1 for word in self.culture_keywords["positive"] if word in text)
            negative_count = sum(1 for word in self.culture_keywords["negative"] if word in text)
            
            # Calculate sentiment score (-1 to 1)
            total_keywords = positive_count + negative_count
            if total_keywords > 0:
                score = (positive_count - negative_count) / total_keywords
            else:
                score = 0.0
            
            sentiment_scores.append(score)
        
        if not sentiment_scores:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "overall": 0.0}
        
        # Calculate overall sentiment distribution
        positive_ratio = sum(1 for s in sentiment_scores if s > 0.1) / len(sentiment_scores)
        negative_ratio = sum(1 for s in sentiment_scores if s < -0.1) / len(sentiment_scores)
        neutral_ratio = 1.0 - positive_ratio - negative_ratio
        
        overall_sentiment = statistics.mean(sentiment_scores)
        
        return {
            "positive": positive_ratio,
            "neutral": neutral_ratio,
            "negative": negative_ratio,
            "overall": overall_sentiment
        }
    
    def _extract_culture_keywords(self, documents: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Extract key culture-related keywords with weights."""
        
        keyword_counts = {}
        total_docs = len(documents)
        
        if total_docs == 0:
            return []
        
        # Count keyword occurrences
        for doc in documents:
            content = doc.get("content", "")
            
            for category_keywords in self.culture_keywords.values():
                for keyword in category_keywords:
                    if keyword in content:
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Calculate keyword weights
        keyword_weights = [
            (keyword, count / total_docs)
            for keyword, count in keyword_counts.items()
        ]
        
        # Sort by weight and return top keywords
        keyword_weights.sort(key=lambda x: x[1], reverse=True)
        return keyword_weights[:10]
    
    def _generate_pros_cons_summary(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Generate pros and cons summary from documents."""
        
        pros = []
        cons = []
        
        for doc in documents:
            content = doc.get("content", "")
            
            # Simple heuristic to identify pros and cons
            positive_score = sum(1 for word in self.culture_keywords["positive"] if word in content)
            negative_score = sum(1 for word in self.culture_keywords["negative"] if word in content)
            
            if positive_score > negative_score and positive_score > 0:
                # Extract positive aspects
                sentences = content.split('.')
                for sentence in sentences[:3]:  # Limit to first 3 sentences
                    if any(word in sentence for word in self.culture_keywords["positive"]):
                        pros.append(sentence.strip())
            
            elif negative_score > positive_score and negative_score > 0:
                # Extract negative aspects
                sentences = content.split('.')
                for sentence in sentences[:3]:  # Limit to first 3 sentences
                    if any(word in sentence for word in self.culture_keywords["negative"]):
                        cons.append(sentence.strip())
        
        # Remove duplicates and limit length
        pros = list(set(pros))[:5]
        cons = list(set(cons))[:5]
        
        return pros, cons
    
    def _calculate_culture_scores(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate various culture scores."""
        
        # Base scores from sentiment analysis
        all_texts = [doc.get("content", "") for doc in documents]
        
        # Add external review text
        for source_data in external_data.values():
            if isinstance(source_data, list):
                for item in source_data:
                    if isinstance(item, dict):
                        all_texts.extend([
                            item.get("pros", ""),
                            item.get("cons", ""),
                            item.get("title", "")
                        ])
        
        sentiment_analysis = self._analyze_sentiment(all_texts)
        
        # Calculate individual scores based on sentiment and keywords
        base_score = max(0, min(100, (sentiment_analysis["overall"] + 1) * 50))
        
        scores = {
            "culture_score": base_score,
            "work_life_balance": base_score + (sentiment_analysis["positive"] - sentiment_analysis["negative"]) * 20,
            "growth_opportunity": base_score + (sentiment_analysis["positive"] - sentiment_analysis["negative"]) * 15,
            "management_quality": base_score + (sentiment_analysis["positive"] - sentiment_analysis["negative"]) * 10,
            "colleague_relationship": base_score + (sentiment_analysis["positive"] - sentiment_analysis["negative"]) * 12,
            "work_environment": base_score + (sentiment_analysis["positive"] - sentiment_analysis["negative"]) * 8,
            "company_stability": base_score + (sentiment_analysis["positive"] - sentiment_analysis["negative"]) * 5
        }
        
        # Ensure scores are within valid range
        for key in scores:
            scores[key] = max(0, min(100, scores[key]))
        
        return scores
    
    def _calculate_confidence_score(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on data availability and quality."""
        
        confidence_factors = []
        
        # Document count factor
        doc_count = len(documents)
        doc_factor = min(1.0, doc_count / 20)  # Optimal at 20+ documents
        confidence_factors.append(doc_factor)
        
        # External data factor
        external_sources = len([source for source in external_data.values() if source])
        external_factor = min(1.0, external_sources / 2)  # Optimal at 2+ sources
        confidence_factors.append(external_factor)
        
        # Data recency factor (assume recent data if no timestamp)
        recency_factor = 0.8  # Default assumption of reasonably recent data
        confidence_factors.append(recency_factor)
        
        # Content quality factor (based on content length)
        if documents:
            avg_content_length = statistics.mean([
                len(doc.get("content", "")) for doc in documents
            ])
            quality_factor = min(1.0, avg_content_length / 500)  # Optimal at 500+ chars
            confidence_factors.append(quality_factor)
        
        # Calculate overall confidence
        return statistics.mean(confidence_factors) if confidence_factors else 0.5