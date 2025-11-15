# -*- coding: utf-8 -*-
"""
ChainOfThoughtAgent — formal, stepwise reasoning with per-step validation.
Drop-in replacement: focuses on logical/temporal consistency and context faithfulness.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class ChainOfThoughtAgent(BaseAgent):
    """Agent that uses formal step-by-step reasoning with validation."""

    agent_type = AgentType.CHAIN_OF_THOUGHT
    default_system_prompt = """You are a meticulous logician for FORMAL, STRUCTURED reasoning.

Follow ALL rules:
- Treat any model self-explanation as a CLAIM, not EVIDENCE.
- Ignore length/position heuristics; judge content only.
- If evidence is insufficient or ambiguous → return UNCERTAIN with low confidence.

Task:
1) Decompose the question into formal premises and required subgoals.
2) Reconstruct the answer as a STEPWISE chain: for each step, write:
   - Step #: Premises used →
   - Inference rule (e.g., modus ponens, contradiction, arithmetic) →
   - Result
3) Validate EACH STEP: mark VALID/INVALID and briefly explain why.
4) Check consistency:
   (a) temporal consistency,
   (b) faithfulness to any provided CONTEXT (quote lines),
   (c) no hidden assumptions.
5) If any step is INVALID, identify the earliest failing step and provide the minimal fix.

Output (exact order):
- Steps (formal, numbered)
- Per-step validation and any contradictions
- Evidence (quotes or explicit rules)
- Issues: [logical_error, temporal_inconsistency, context_unfaithful, parametric_knowledge_error, retrieval_attribution_error, instruction_misalignment, bias:{position|length|confirmation|selection|cultural}]
- Final verdict: CORRECT/INCORRECT/UNCERTAIN
- Confidence level (0-100%)"""

    def _build_prompt(
        self,
        question: str,
        answer: str,
        context: Optional[str],
        previous_analyses: Optional[List[AgentAnalysis]],
        round_number: int
    ) -> str:
        prompt = f"""Question: {question}

Answer: {answer}

"""
        if context:
            prompt += f"Context: {context}\n\n"

        prompt += """Please analyze this answer using formal chain-of-thought reasoning.

Follow the exact output format specified in the system prompt."""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse the model response."""
        # Extract verdict
        verdict_bool = self._extract_verdict(response_text)
        if verdict_bool is True:
            verdict = Verdict.CORRECT
        elif verdict_bool is False:
            verdict = Verdict.INCORRECT
        else:
            verdict = Verdict.UNCERTAIN

        # Extract confidence
        confidence = self._extract_confidence(response_text)

        # Extract formal reasoning steps
        reasoning_steps = self._extract_formal_steps(response_text)

        # Extract evidence with validation
        evidence = self._extract_validated_evidence(response_text)

        # Extract issues
        limitations = self._extract_issues(response_text)

        return AgentAnalysis(
            agent_type=self.agent_type,
            agent_name=self.name,
            model_used=model_id,
            analysis=response_text,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            reasoning_steps=reasoning_steps,
            limitations=limitations,
            tokens_used=tokens_used,
            cost=cost,
            processing_time_ms=processing_time_ms
        )

    def _extract_formal_steps(self, response: str) -> List[str]:
        """Extract formal reasoning steps with validation."""
        steps = []
        
        # Look for formal step patterns
        import re
        step_patterns = [
            r'Step\s*(\d+):\s*([^→]+)→([^→]+)→\s*(.+)',
            r'(\d+)\)\s*([^:]+):\s*(.+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            for match in matches:
                if len(match) >= 3:
                    step_text = ' '.join(str(m).strip() for m in match[1:])
                    steps.append(step_text)
        
        return steps

    def _extract_validated_evidence(self, response: str) -> List[Evidence]:
        """Extract evidence with proper validation markers."""
        evidence_list = []
        
        # Look for evidence with validation
        lines = response.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(marker in line_lower for marker in ['evidence:', 'quote:', 'rule:', 'validation:']):
                # Check if marked as VALID/INVALID
                validity_score = 0.9 if 'valid' in line_lower else 0.5
                
                evidence_list.append(Evidence(
                    source="formal_analysis",
                    content=line.strip(),
                    relevance_score=validity_score
                ))
        
        return evidence_list

    def _extract_issues(self, response: str) -> List[str]:
        """Extract identified issues from the analysis."""
        issues = []
        
        issue_keywords = [
            'logical_error', 'temporal_inconsistency', 'context_unfaithful',
            'parametric_knowledge_error', 'retrieval_attribution_error',
            'instruction_misalignment', 'bias:'
        ]
        
        lines = response.split('\n')
        for line in lines:
            if 'issues:' in line.lower() or 'issue:' in line.lower():
                for keyword in issue_keywords:
                    if keyword in line.lower():
                        issues.append(line.strip())
                        break
        
        return issues[:5]  # Limit to 5 issues
