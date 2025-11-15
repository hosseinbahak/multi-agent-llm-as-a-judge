# -*- coding: utf-8 -*-
"""
InnovatorAgent — proposes a distinct alternative solution path; checks agreement with original result.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class InnovatorAgent(BaseAgent):
    """Agent that explores alternative reasoning paths."""

    agent_type = AgentType.INNOVATOR
    default_system_prompt = """You explore alternative reasoning.

Tasks:
1) Propose a DISTINCT alternative solution path (not mere paraphrasing).
2) Compare the final conclusion with the original: SAME/DIFFERENT.
3) If DIFFERENT, explain which step diverges and why; prefer caution in the verdict.

Output:
- Alternative path (concise, stepwise)
- Agreement with original: SAME/DIFFERENT (+why)
- Issues: [logical_error, temporal_inconsistency, context_unfaithful, parametric_knowledge_error]
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

Original answer: {answer}

"""
        if context:
            prompt += f"Context: {context}\n\n"

        prompt += """Develop a DISTINCT alternative reasoning path and compare results."""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse the innovator response."""
        # Extract agreement status
        agreement = self._extract_agreement_status(response_text)
        
        # Extract verdict - be cautious if paths differ
        verdict_bool = self._extract_verdict(response_text)
        if agreement == "DIFFERENT":
            # Prefer UNCERTAIN when alternative path differs
            verdict = Verdict.UNCERTAIN
        elif verdict_bool is True:
            verdict = Verdict.CORRECT
        elif verdict_bool is False:
            verdict = Verdict.INCORRECT
        else:
            verdict = Verdict.UNCERTAIN

        # Extract confidence
        confidence = self._extract_confidence(response_text)
        
        # Extract alternative path steps
        alternative_steps = self._extract_alternative_path(response_text)
        
        # Extract divergence analysis
        evidence = self._extract_divergence_evidence(response_text)
        
        # Extract issues
        limitations = self._extract_innovation_issues(response_text)

        return AgentAnalysis(
            agent_type=self.agent_type,
            agent_name=self.name,
            model_used=model_id,
            analysis=response_text,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            reasoning_steps=alternative_steps,
            limitations=limitations,
            tokens_used=tokens_used,
            cost=cost,
            processing_time_ms=processing_time_ms
        )

    def _extract_agreement_status(self, response: str) -> str:
        """Extract whether alternative path agrees with original."""
        import re
        
        # Look for SAME/DIFFERENT markers
        agreement_pattern = r'agreement.*?:\s*(SAME|DIFFERENT)'
        match = re.search(agreement_pattern, response, re.IGNORECASE)
        
        if match:
            return match.group(1).upper()
        
        # Fallback: check for keywords
        if 'differ' in response.lower() or 'disagree' in response.lower():
            return "DIFFERENT"
        elif 'same' in response.lower() or 'agree' in response.lower():
            return "SAME"
        
        return "UNCERTAIN"

    def _extract_alternative_path(self, response: str) -> List[str]:
        """Extract the alternative reasoning path."""
        steps = []
        
        # Look for alternative path section
        import re
        path_section = self._extract_section(response, 'alternative path', 'agreement')
        
        if path_section:
            # Extract numbered steps
            step_pattern = r'(\d+)[.)\s]+(.+?)(?=\d+[.)\s]|$)'
            matches = re.findall(step_pattern, path_section, re.DOTALL)
            
            for _, step_content in matches:
                step_text = step_content.strip().replace('\n', ' ')
                if len(step_text) > 10:  # Filter out very short steps
                    steps.append(step_text)
        
        # Fallback: extract any stepwise content
        if not steps:
            lines = response.split('\n')
            for line in lines:
                if any(marker in line for marker in ['step', '→', 'then', 'therefore']):
                    steps.append(line.strip())
        
        return steps[:7]  # Limit to 7 steps

    def _extract_divergence_evidence(self, response: str) -> List[Evidence]:
        """Extract evidence about where paths diverge."""
        evidence_list = []
        
        divergence_keywords = ['diverge', 'differ', 'contrast', 'alternative', 'instead']
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in divergence_keywords):
                evidence_list.append(Evidence(
                    source="alternative_reasoning",
                    content=line.strip(),
                    relevance_score=0.85
                ))
        
        return evidence_list

    def _extract_innovation_issues(self, response: str) -> List[str]:
        """Extract identified issues from innovative analysis."""
        issues = []
        
        issue_keywords = [
            'logical_error', 'temporal_inconsistency',
            'context_unfaithful', 'parametric_knowledge_error'
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in issue_keywords:
                if keyword in line_lower:
                    issues.append(line.strip())
                    break
        
        return issues[:4]

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract a section between markers."""
        import re
        pattern = f'{start_marker}(.*?)(?:{end_marker}|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
