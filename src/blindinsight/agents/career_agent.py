"""
Career Path Analysis Agent for BlindInsight AI.

Specializes in analyzing career opportunities, entry requirements,
and personalized career strategy based on user profiles and company data.
"""

import statistics
from typing import Any, Dict, List, Optional

from ..models.analysis import CareerReport, CultureReport, CompensationReport
from ..models.user import UserProfile, SkillCategory
from .base import BaseAgent, AgentConfig, AgentResult


class CareerPathAgent(BaseAgent):
    """
    Agent specialized in career path and entry strategy analysis.
    
    This agent analyzes career opportunities by combining:
    - RAG-retrieved hiring and career progression information
    - External job market and hiring data
    - User profile matching and skill gap analysis
    - Personalized entry strategy recommendations
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__("career_path", config)
        
        # Common skill requirements by role
        self.skill_requirements = {
            "software_engineer": {
                "required": ["Python", "Java", "JavaScript", "Git", "데이터베이스"],
                "preferred": ["AWS", "Docker", "Kubernetes", "React", "Spring"]
            },
            "ai_engineer": {
                "required": ["Python", "PyTorch", "TensorFlow", "머신러닝", "데이터분석"],
                "preferred": ["Kubernetes", "MLOps", "클라우드", "분산처리"]
            },
            "frontend_developer": {
                "required": ["JavaScript", "React", "HTML", "CSS", "TypeScript"],
                "preferred": ["Vue.js", "Angular", "웹팩", "반응형디자인"]
            },
            "backend_developer": {
                "required": ["Java", "Spring", "데이터베이스", "API", "서버"],
                "preferred": ["마이크로서비스", "Redis", "메시지큐", "도커"]
            }
        }
        
        # Career progression timelines
        self.progression_timelines = {
            "junior_to_mid": 24,      # months
            "mid_to_senior": 36,      # months
            "senior_to_lead": 48,     # months
            "lead_to_manager": 60     # months
        }
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """Execute career path analysis."""
        
        company = context.get("company") if context else query
        position = context.get("position") if context else None
        culture_context = context.get("culture_report") if context else None
        compensation_context = context.get("compensation_report") if context else None
        
        return await self._execute_with_error_handling(
            self.analyze_career_path,
            company,
            position,
            user_profile,
            culture_context,
            compensation_context
        )
    
    async def analyze_career_path(
        self,
        company: str,
        position: Optional[str] = None,
        user_profile: Optional[UserProfile] = None,
        culture_context: Optional[CultureReport] = None,
        compensation_context: Optional[CompensationReport] = None
    ) -> CareerReport:
        """Analyze career path and entry strategy."""
        
        # Retrieve career-related documents
        documents = await self._retrieve_career_documents(company, position)
        
        # Fetch external hiring data
        external_data = await self._fetch_external_hiring_data(company, position)
        
        # Analyze skill requirements
        skill_requirements = self._analyze_skill_requirements(documents, external_data, position)
        
        # Perform skill gap analysis if user profile provided
        skill_gap_analysis = {}
        if user_profile:
            skill_gap_analysis = self._perform_skill_gap_analysis(skill_requirements, user_profile)
        
        # Generate entry strategy
        entry_strategy = self._generate_entry_strategy(
            skill_requirements, skill_gap_analysis, user_profile
        )
        
        # Calculate success probability
        success_probability = self._calculate_success_probability(
            skill_gap_analysis, user_profile, culture_context, compensation_context
        )
        
        # Generate personalized recommendations
        personalized_advice = self._generate_personalized_advice(
            skill_gap_analysis, user_profile, culture_context
        )
        
        # Analyze interview process
        interview_insights = self._analyze_interview_process(documents, external_data)
        
        # Calculate confidence
        confidence = self._calculate_confidence_score(documents, external_data)
        
        return CareerReport(
            required_skills=skill_requirements.get("required", []),
            preferred_skills=skill_requirements.get("preferred", []),
            skill_gap_analysis=skill_gap_analysis,
            preparation_plan=entry_strategy["plan"],
            success_probability=success_probability["current"],
            timeline_estimate=entry_strategy["timeline"],
            improved_probability=success_probability["improved"],
            interview_process=interview_insights["process"],
            common_questions=interview_insights["questions"],
            hiring_criteria=interview_insights["criteria"],
            personalized_advice=personalized_advice["advice"],
            learning_resources=personalized_advice["resources"],
            networking_suggestions=personalized_advice["networking"],
            data_points_count=len(documents),
            confidence_score=confidence
        )
    
    async def _retrieve_career_documents(
        self, 
        company: str, 
        position: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve career and hiring-related documents."""
        
        queries = [
            f"{company} 채용 과정",
            f"{company} 면접 후기",
            f"{company} 입사 준비",
            f"{company} 커리어 패스"
        ]
        
        if position:
            queries.extend([
                f"{company} {position} 채용",
                f"{company} {position} 면접",
                f"{company} {position} 요구사항"
            ])
        
        all_documents = []
        
        for query in queries:
            documents = await self.retrieve_knowledge(
                query=query,
                category="career",
                k=3
            )
            
            for doc in documents:
                all_documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "query": query
                })
        
        return all_documents
    
    async def _fetch_external_hiring_data(
        self, 
        company: str, 
        position: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch external hiring and job posting data."""
        
        external_data = {}
        
        try:
            # Job posting data
            job_data = await self.call_mcp_service(
                "api_integration",
                "get_job_postings",
                {"company": company, "position": position}
            )
            
            if job_data:
                external_data["job_postings"] = job_data
        
        except Exception as e:
            print(f"Failed to fetch job posting data: {e}")
        
        return external_data
    
    def _analyze_skill_requirements(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any],
        position: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Analyze skill requirements from documents and external data."""
        
        # Start with position-based defaults
        role_key = self._map_position_to_role(position) if position else "software_engineer"
        default_requirements = self.skill_requirements.get(role_key, {
            "required": ["Programming", "Problem Solving"],
            "preferred": ["Communication", "Teamwork"]
        })
        
        required_skills = set(default_requirements["required"])
        preferred_skills = set(default_requirements["preferred"])
        
        # Extract skills from documents
        all_content = ""
        for doc in documents:
            all_content += doc.get("content", "") + " "
        
        # Add external job posting content
        for source_data in external_data.values():
            if isinstance(source_data, list):
                for item in source_data:
                    if isinstance(item, dict):
                        all_content += " ".join([
                            item.get("requirements", ""),
                            item.get("preferred_qualifications", ""),
                            item.get("description", "")
                        ])
        
        # Common technical skills to look for
        technical_skills = [
            "Python", "Java", "JavaScript", "React", "Vue.js", "Angular",
            "Spring", "Django", "Node.js", "AWS", "Docker", "Kubernetes",
            "MySQL", "PostgreSQL", "Redis", "Git", "Jenkins", "CI/CD",
            "PyTorch", "TensorFlow", "머신러닝", "데이터분석", "AI"
        ]
        
        # Look for skills in content
        for skill in technical_skills:
            if skill in all_content or skill.lower() in all_content.lower():
                if any(req_keyword in all_content for req_keyword in ["필수", "요구", "required"]):
                    required_skills.add(skill)
                else:
                    preferred_skills.add(skill)
        
        return {
            "required": list(required_skills),
            "preferred": list(preferred_skills)
        }
    
    def _perform_skill_gap_analysis(
        self, 
        skill_requirements: Dict[str, List[str]], 
        user_profile: UserProfile
    ) -> Dict[str, str]:
        """Perform skill gap analysis against user profile."""
        
        if not user_profile or not hasattr(user_profile, 'skills'):
            return {}
        
        skill_gap = {}
        
        # Check required skills
        user_skills = set(user_profile.skills.technical_skills.keys()) if user_profile.skills else set()
        
        for skill in skill_requirements.get("required", []):
            if skill not in user_skills:
                skill_gap[skill] = "필수 스킬 부족"
            else:
                # Check proficiency level
                proficiency = user_profile.skills.technical_skills.get(skill, 0)
                if proficiency < 3:  # Assuming 1-5 scale
                    skill_gap[skill] = "실력 향상 필요"
                else:
                    skill_gap[skill] = "보유"
        
        # Check preferred skills
        for skill in skill_requirements.get("preferred", []):
            if skill not in user_skills:
                skill_gap[skill] = "우대 스킬 추가 학습 권장"
            elif user_profile.skills.technical_skills.get(skill, 0) < 4:
                skill_gap[skill] = "우대 스킬 심화 학습 권장"
            else:
                skill_gap[skill] = "충분한 수준"
        
        return skill_gap
    
    def _generate_entry_strategy(
        self,
        skill_requirements: Dict[str, List[str]],
        skill_gap_analysis: Dict[str, str],
        user_profile: Optional[UserProfile]
    ) -> Dict[str, Any]:
        """Generate personalized entry strategy."""
        
        preparation_plan = []
        timeline_months = 6  # Default timeline
        
        # Analyze skill gaps and create learning plan
        missing_required = [
            skill for skill, status in skill_gap_analysis.items()
            if "부족" in status and skill in skill_requirements.get("required", [])
        ]
        
        improvement_needed = [
            skill for skill, status in skill_gap_analysis.items()
            if "향상 필요" in status
        ]
        
        # Create preparation steps
        if missing_required:
            preparation_plan.append({
                "step": "필수 스킬 습득",
                "skills": missing_required,
                "timeline": "2-3개월",
                "priority": "high"
            })
            timeline_months = max(timeline_months, 3)
        
        if improvement_needed:
            preparation_plan.append({
                "step": "기존 스킬 심화",
                "skills": improvement_needed,
                "timeline": "1-2개월",
                "priority": "medium"
            })
        
        # Add portfolio/project recommendations
        preparation_plan.append({
            "step": "포트폴리오 프로젝트",
            "description": "실무 경험을 보여줄 수 있는 프로젝트 개발",
            "timeline": "1-2개월",
            "priority": "high"
        })
        
        # Add networking and application preparation
        preparation_plan.append({
            "step": "네트워킹 및 지원 준비",
            "description": "업계 네트워킹 및 이력서/자기소개서 준비",
            "timeline": "2-4주",
            "priority": "medium"
        })
        
        return {
            "plan": preparation_plan,
            "timeline": timeline_months
        }
    
    def _calculate_success_probability(
        self,
        skill_gap_analysis: Dict[str, str],
        user_profile: Optional[UserProfile],
        culture_context: Optional[CultureReport],
        compensation_context: Optional[CompensationReport]
    ) -> Dict[str, float]:
        """Calculate success probability for job application."""
        
        base_probability = 50.0  # Base 50% chance
        
        # Skill matching factor
        if skill_gap_analysis:
            total_skills = len(skill_gap_analysis)
            if total_skills > 0:
                sufficient_skills = sum(1 for status in skill_gap_analysis.values() if "보유" in status or "충분" in status)
                skill_factor = (sufficient_skills / total_skills) * 30  # Up to 30% boost
                base_probability += skill_factor
        
        # Experience factor
        if user_profile and hasattr(user_profile, 'experience_years'):
            experience_boost = min(15, user_profile.experience_years * 3)  # Up to 15% boost
            base_probability += experience_boost
        
        # Culture fit factor
        if culture_context and culture_context.culture_score > 75:
            base_probability += 10  # Cultural fit bonus
        
        # Current probability
        current_probability = min(95.0, max(10.0, base_probability))
        
        # Improved probability after preparation
        preparation_boost = 20  # Assume 20% improvement after preparation
        improved_probability = min(95.0, current_probability + preparation_boost)
        
        return {
            "current": current_probability,
            "improved": improved_probability
        }
    
    def _generate_personalized_advice(
        self,
        skill_gap_analysis: Dict[str, str],
        user_profile: Optional[UserProfile],
        culture_context: Optional[CultureReport]
    ) -> Dict[str, Any]:
        """Generate personalized career advice."""
        
        advice = []
        resources = []
        networking = []
        
        # Skill-based advice
        if skill_gap_analysis:
            missing_skills = [skill for skill, status in skill_gap_analysis.items() if "부족" in status]
            if missing_skills:
                advice.append(f"우선적으로 {', '.join(missing_skills[:3])} 스킬을 집중 학습하세요.")
                
                # Add learning resources
                for skill in missing_skills[:3]:
                    if skill.lower() in ["python", "java", "javascript"]:
                        resources.append({
                            "skill": skill,
                            "resource": "온라인 코딩 부트캠프",
                            "url": "https://example.com"
                        })
        
        # Experience-based advice
        if user_profile and hasattr(user_profile, 'experience_years'):
            if user_profile.experience_years < 2:
                advice.append("신입 개발자로서 개인 프로젝트와 포트폴리오를 충실히 준비하세요.")
            elif user_profile.experience_years < 5:
                advice.append("중급 개발자로서 기술 심화와 리더십 경험을 쌓으세요.")
        
        # Culture-based advice
        if culture_context:
            if culture_context.work_life_balance > 80:
                advice.append("이 회사는 워라밸이 좋은 편이므로 장기 커리어 관점에서 고려해보세요.")
            if culture_context.growth_opportunity > 80:
                advice.append("성장 기회가 많은 회사이므로 적극적인 학습 의지를 어필하세요.")
        
        # Networking suggestions
        networking.extend([
            "LinkedIn에서 해당 회사 직원들과 네트워킹",
            "기술 커뮤니티와 컨퍼런스 참여",
            "회사 기술 블로그 및 발표 자료 연구"
        ])
        
        return {
            "advice": advice,
            "resources": resources,
            "networking": networking
        }
    
    def _analyze_interview_process(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze interview process and common questions."""
        
        all_content = ""
        for doc in documents:
            all_content += doc.get("content", "") + " "
        
        # Interview process steps
        process_steps = []
        if "서류전형" in all_content:
            process_steps.append("서류 전형")
        if "코딩테스트" in all_content or "온라인테스트" in all_content:
            process_steps.append("코딩 테스트")
        if "기술면접" in all_content:
            process_steps.append("기술 면접")
        if "임원면접" in all_content or "최종면접" in all_content:
            process_steps.append("최종 면접")
        
        # Common questions (simplified)
        common_questions = [
            "자기소개를 해주세요",
            "이 회사에 지원한 이유는 무엇인가요?",
            "가장 도전적이었던 프로젝트에 대해 설명해주세요",
            "기술적인 어려움을 어떻게 해결하시나요?"
        ]
        
        # Hiring criteria
        hiring_criteria = [
            "기술적 역량",
            "문제 해결 능력",
            "의사소통 능력",
            "팀워크",
            "학습 의지"
        ]
        
        return {
            "process": process_steps,
            "questions": common_questions,
            "criteria": hiring_criteria
        }
    
    def _map_position_to_role(self, position: str) -> str:
        """Map position title to standardized role key."""
        
        position_lower = position.lower()
        
        if any(keyword in position_lower for keyword in ["ai", "머신러닝", "딥러닝", "데이터사이언스"]):
            return "ai_engineer"
        elif any(keyword in position_lower for keyword in ["프론트엔드", "frontend", "react", "vue"]):
            return "frontend_developer"
        elif any(keyword in position_lower for keyword in ["백엔드", "backend", "서버", "api"]):
            return "backend_developer"
        else:
            return "software_engineer"
    
    def _calculate_confidence_score(
        self, 
        documents: List[Dict[str, Any]], 
        external_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for career analysis."""
        
        confidence_factors = []
        
        # Document availability
        doc_factor = min(1.0, len(documents) / 10)  # Optimal at 10+ documents
        confidence_factors.append(doc_factor)
        
        # External data availability
        external_factor = 0.8 if external_data else 0.5
        confidence_factors.append(external_factor)
        
        # Content relevance (based on career-related keywords)
        career_keywords = ["채용", "면접", "입사", "커리어", "스킬", "요구사항"]
        if documents:
            total_content = " ".join([doc.get("content", "") for doc in documents])
            relevance_score = sum(1 for keyword in career_keywords if keyword in total_content)
            relevance_factor = min(1.0, relevance_score / len(career_keywords))
            confidence_factors.append(relevance_factor)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.6