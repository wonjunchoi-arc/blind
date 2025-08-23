"""
Base agent classes and utilities for BlindInsight AI.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

from ..models.base import BaseModel, settings
from ..models.user import UserProfile


class AgentResult(BaseModel):
    """Result container for agent execution."""
    
    agent_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = {}
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "agent_name": "culture_analysis",
                "success": True,
                "result": {"culture_score": 85},
                "execution_time": 12.5,
                "confidence_score": 0.88
            }
        }


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    
    # LLM settings
    model_name: str = "gpt-4-turbo"
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 60
    
    # Retrieval settings
    max_retrievals: int = 20
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    
    # Quality settings
    min_confidence_threshold: float = 0.6
    require_sources: bool = True
    validate_outputs: bool = True
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "model_name": "gpt-4-turbo",
                "temperature": 0.3,
                "max_retrievals": 20,
                "enable_caching": True
            }
        }


class BaseAgent(ABC):
    """
    Abstract base class for all BlindInsight AI agents.
    
    This class provides common functionality for RAG-based retrieval,
    LLM interaction, MCP service calls, and result validation.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None
    ):
        self.name = name
        self.config = config or AgentConfig()
        self.tools = tools or []
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            api_key=settings.openai_api_key
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize RAG retriever (will be set by subclasses)
        self.rag_retriever = None
        
        # Initialize MCP clients (will be set by subclasses)
        self.mcp_clients = {}
        
        # Performance tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._success_count = 0
    
    @abstractmethod
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """
        Execute the agent's main functionality.
        
        Args:
            query: The main query or instruction for the agent
            context: Additional context information
            user_profile: User profile for personalized analysis
            
        Returns:
            AgentResult containing the execution results
        """
        pass
    
    async def retrieve_knowledge(
        self, 
        query: str, 
        category: Optional[str] = None,
        k: int = None
    ) -> List[Document]:
        """
        Retrieve relevant knowledge using RAG system.
        
        Args:
            query: Search query
            category: Optional category filter
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.rag_retriever:
            return []
        
        k = k or self.config.max_retrievals
        
        try:
            # Build search parameters
            search_kwargs = {"k": k}
            if category:
                search_kwargs["filter"] = {"category": category}
            
            # Perform retrieval
            documents = await self.rag_retriever.aget_relevant_documents(
                query, 
                **search_kwargs
            )
            
            # Filter by similarity threshold if available
            if hasattr(self.rag_retriever, "similarity_threshold"):
                documents = [
                    doc for doc in documents 
                    if getattr(doc, "metadata", {}).get("score", 1.0) >= self.config.similarity_threshold
                ]
            
            return documents[:k]
            
        except Exception as e:
            print(f"RAG retrieval failed for agent {self.name}: {str(e)}")
            return []
    
    async def call_mcp_service(
        self, 
        service_name: str, 
        method: str, 
        params: Dict[str, Any]
    ) -> Any:
        """
        Call an MCP service.
        
        Args:
            service_name: Name of the MCP service
            method: Method to call
            params: Parameters for the method
            
        Returns:
            Service response
        """
        if service_name not in self.mcp_clients:
            raise ValueError(f"MCP service '{service_name}' not configured for agent {self.name}")
        
        try:
            client = self.mcp_clients[service_name]
            return await client.call(method, params)
        except Exception as e:
            print(f"MCP service call failed for {service_name}.{method}: {str(e)}")
            return None
    
    async def generate_response(
        self, 
        prompt: str, 
        context_documents: List[Document] = None,
        **kwargs
    ) -> str:
        """
        Generate response using LLM with optional context.
        
        Args:
            prompt: Input prompt
            context_documents: Optional context documents
            **kwargs: Additional LLM parameters
            
        Returns:
            Generated response
        """
        # Build context if documents provided
        if context_documents:
            context_text = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_documents[:5])  # Limit context size
            ])
            
            full_prompt = f"""Context Information:
{context_text}

Query: {prompt}

Please provide a comprehensive analysis based on the context information above."""
        else:
            full_prompt = prompt
        
        try:
            # Generate response
            response = await self.llm.ainvoke(full_prompt, **kwargs)
            return response.content
        except Exception as e:
            print(f"LLM generation failed for agent {self.name}: {str(e)}")
            return ""
    
    async def validate_result(
        self, 
        result: Dict[str, Any], 
        required_fields: List[str] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate agent result.
        
        Args:
            result: Result to validate
            required_fields: List of required fields
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        errors = []
        
        if not self.config.validate_outputs:
            return True, errors
        
        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in result:
                    errors.append(f"Missing required field: {field}")
        
        # Check confidence threshold
        confidence = result.get("confidence_score", 0.0)
        if confidence < self.config.min_confidence_threshold:
            errors.append(f"Confidence score {confidence} below threshold {self.config.min_confidence_threshold}")
        
        # Check sources if required
        if self.config.require_sources and not result.get("sources"):
            errors.append("Sources are required but not provided")
        
        return len(errors) == 0, errors
    
    def update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """Update agent performance metrics."""
        self._execution_count += 1
        self._total_execution_time += execution_time
        if success:
            self._success_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        if self._execution_count == 0:
            return {
                "execution_count": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        return {
            "execution_count": self._execution_count,
            "success_rate": self._success_count / self._execution_count,
            "average_execution_time": self._total_execution_time / self._execution_count,
            "total_execution_time": self._total_execution_time
        }
    
    async def _execute_with_error_handling(
        self,
        execution_func,
        *args,
        **kwargs
    ) -> AgentResult:
        """
        Execute agent function with comprehensive error handling.
        
        Args:
            execution_func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            AgentResult with execution results
        """
        start_time = time.time()
        
        try:
            # Execute the main function
            result = await execution_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Validate result if it's a dictionary
            if isinstance(result, dict):
                is_valid, validation_errors = await self.validate_result(result)
                if not is_valid:
                    self.update_performance_metrics(execution_time, False)
                    return AgentResult(
                        agent_name=self.name,
                        success=False,
                        error_message=f"Validation failed: {', '.join(validation_errors)}",
                        execution_time=execution_time
                    )
            
            # Update metrics and return success
            self.update_performance_metrics(execution_time, True)
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                result=result if isinstance(result, dict) else {"output": result},
                execution_time=execution_time,
                confidence_score=result.get("confidence_score", 1.0) if isinstance(result, dict) else 1.0
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_performance_metrics(execution_time, False)
            
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


class AgentOrchestrator:
    """Orchestrates multiple agents for complex analysis tasks."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    async def execute_agents(
        self,
        agent_names: List[str],
        query: str,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> Dict[str, AgentResult]:
        """
        Execute multiple agents.
        
        Args:
            agent_names: List of agent names to execute
            query: Query to pass to agents
            context: Additional context
            parallel: Execute agents in parallel if True
            
        Returns:
            Dictionary mapping agent names to their results
        """
        
        # Filter valid agents
        valid_agents = [
            (name, self.agents[name]) 
            for name in agent_names 
            if name in self.agents
        ]
        
        if not valid_agents:
            return {}
        
        start_time = time.time()
        
        if parallel:
            # Execute agents in parallel
            tasks = [
                agent.execute(query, context)
                for _, agent in valid_agents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            agent_results = {}
            for (name, _), result in zip(valid_agents, results):
                if isinstance(result, Exception):
                    agent_results[name] = AgentResult(
                        agent_name=name,
                        success=False,
                        error_message=str(result),
                        execution_time=time.time() - start_time
                    )
                else:
                    agent_results[name] = result
        
        else:
            # Execute agents sequentially
            agent_results = {}
            for name, agent in valid_agents:
                result = await agent.execute(query, context)
                agent_results[name] = result
        
        # Record execution history
        total_time = time.time() - start_time
        self.execution_history.append({
            "timestamp": datetime.now(),
            "agent_names": agent_names,
            "parallel": parallel,
            "total_time": total_time,
            "success_count": sum(1 for r in agent_results.values() if r.success),
            "total_agents": len(agent_results)
        })
        
        return agent_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all registered agents."""
        agent_stats = {
            name: agent.get_performance_stats()
            for name, agent in self.agents.items()
        }
        
        orchestrator_stats = {
            "total_orchestrations": len(self.execution_history),
            "average_orchestration_time": (
                sum(h["total_time"] for h in self.execution_history) / len(self.execution_history)
                if self.execution_history else 0.0
            )
        }
        
        return {
            "agent_stats": agent_stats,
            "orchestrator_stats": orchestrator_stats
        }


# Agent factory functions
def create_agent(agent_type: str, config: Optional[AgentConfig] = None) -> BaseAgent:
    """
    Factory function to create agents by type.
    
    Args:
        agent_type: Type of agent to create
        config: Optional agent configuration
        
    Returns:
        Configured agent instance
    """
    
    # Import here to avoid circular imports
    from .culture_agent import CultureAnalysisAgent
    from .compensation_agent import CompensationAnalysisAgent
    from .growth_agent import GrowthStabilityAgent
    from .career_agent import CareerPathAgent
    
    agent_classes = {
        "culture": CultureAnalysisAgent,
        "compensation": CompensationAnalysisAgent,
        "growth": GrowthStabilityAgent,
        "career": CareerPathAgent
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_classes[agent_type](config=config)


def get_agent_by_type(agent_type: str) -> Type[BaseAgent]:
    """Get agent class by type."""
    
    # Import here to avoid circular imports
    from .culture_agent import CultureAnalysisAgent
    from .compensation_agent import CompensationAnalysisAgent
    from .growth_agent import GrowthStabilityAgent
    from .career_agent import CareerPathAgent
    
    agent_classes = {
        "culture": CultureAnalysisAgent,
        "compensation": CompensationAnalysisAgent,
        "growth": GrowthStabilityAgent,
        "career": CareerPathAgent
    }
    
    return agent_classes.get(agent_type)