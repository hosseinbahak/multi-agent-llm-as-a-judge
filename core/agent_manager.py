# core/agent_manager.py
"""Agent management and orchestration."""

import asyncio
import importlib
import inspect
from typing import Dict, List, Optional, Type, Any, Set
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from loguru import logger

from .data_models import AgentAnalysis, AgentType, EvaluationRequest
from .model_manager import ModelManager
from .cache_manager import CacheManager
from ..agents.base_agent import BaseAgent
from ..config.schemas import AgentConfig
from .exceptions import AgentExecutionError, AgentTimeoutError


class AgentManager:
    """Manages agent lifecycle and execution."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        cache_manager: CacheManager,
        max_concurrent: int = 5
    ):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.max_concurrent = max_concurrent
        
        self.agent_registry: Dict[AgentType, Type[BaseAgent]] = {}
        self.agent_instances: Dict[str, BaseAgent] = {}
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # Auto-discover agents
        self._discover_agents()
    
    def _discover_agents(self):
        """Auto-discover agent classes."""
        # Import all agent modules
        agent_modules = [
            "agents.analytical.chain_of_thought",
            "agents.analytical.adversary",
            "agents.analytical.challenger",
            "agents.creative.innovator",
            "agents.creative.synthesizer",
            "agents.verification.retrieval_verifier",
            "agents.verification.bias_auditor",
            "agents.meta.meta_qa",
            "agents.meta.assumption_grapher"
        ]
        
        for module_name in agent_modules:
            try:
                module = importlib.import_module(f"multi_agent_llm_judge.{module_name}")
                
                # Find BaseAgent subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseAgent) and obj != BaseAgent:
                        agent_type = obj.agent_type
                        if agent_type:
                            self.agent_registry[agent_type] = obj
                            logger.debug(f"Registered agent: {agent_type} -> {name}")
                            
            except Exception as e:
                logger.error(f"Failed to import {module_name}: {e}")
    
    def register_agent(self, agent_type: AgentType, agent_class: Type[BaseAgent]):
        """Manually register an agent class."""
        self.agent_registry[agent_type] = agent_class
    
    async def create_agent(
        self,
        agent_config: AgentConfig,
        agent_id: Optional[str] = None
    ) -> BaseAgent:
        """Create an agent instance."""
        if agent_config.type not in self.agent_registry:
            raise ValueError(f"Unknown agent type: {agent_config.type}")
        
        agent_class = self.agent_registry[agent_config.type]
        
        # Create unique ID
        if not agent_id:
            agent_id = f"{agent_config.type}_{datetime.now().timestamp()}"
        
        # Create agent instance
        agent = agent_class(
            config=agent_config,
            model_manager=self.model_manager,
            cache_manager=self.cache_manager
        )
        
        self.agent_instances[agent_id] = agent
        
        return agent
    
    async def execute_agent(
        self,
        agent: BaseAgent,
        request: EvaluationRequest,
        previous_analyses: Optional[List[AgentAnalysis]] = None,
        round_number: int = 1,
        timeout: Optional[int] = None
    ) -> AgentAnalysis:
        """Execute a single agent."""
        async with self._semaphore:
            try:
                # Set timeout
                timeout = timeout or agent.config.timeout
                
                # Execute with timeout
                analysis = await asyncio.wait_for(
                    agent.analyze(
                        question=request.question,
                        answer=request.answer,
                        context=request.context,
                        previous_analyses=previous_analyses,
                        round_number=round_number
                    ),
                    timeout=timeout
                )
                
                return analysis
                
            except asyncio.TimeoutError:
                logger.error(f"Agent {agent.name} timed out after {timeout}s")
                raise AgentTimeoutError(f"Agent {agent.name} timed out")
                
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                raise AgentExecutionError(f"Agent {agent.name} failed: {e}")
    
    async def execute_agents_parallel(
        self,
        agents: List[BaseAgent],
        request: EvaluationRequest,
        previous_analyses: Optional[List[AgentAnalysis]] = None,
        round_number: int = 1
    ) -> List[AgentAnalysis]:
        """Execute multiple agents in parallel."""
        tasks = [
            self.execute_agent(
                agent=agent,
                request=request,
                previous_analyses=previous_analyses,
                round_number=round_number
            )
            for agent in agents
        ]
        
        # Execute with error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agents[i].name} failed: {result}")
                # Could add retry logic here
            else:
                analyses.append(result)
        
        return analyses
    
    async def execute_agents_sequential(
        self,
        agents: List[BaseAgent],
        request: EvaluationRequest,
        previous_analyses: Optional[List[AgentAnalysis]] = None,
        round_number: int = 1
    ) -> List[AgentAnalysis]:
        """Execute agents sequentially (for dependencies)."""
        analyses = []
        
        for agent in agents:
            try:
                analysis = await self.execute_agent(
                    agent=agent,
                    request=request,
                    previous_analyses=previous_analyses + analyses,
                    round_number=round_number
                )
                analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Sequential execution failed at {agent.name}: {e}")
                # Continue with other agents
        
        return analyses
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent instance by ID."""
        return self.agent_instances.get(agent_id)
    
    def get_available_agent_types(self) -> List[AgentType]:
        """Get list of available agent types."""
        return list(self.agent_registry.keys())
    
    async def cleanup(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        
        # Clean up agent instances
        for agent in self.agent_instances.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
