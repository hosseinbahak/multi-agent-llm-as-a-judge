# -*- coding: utf-8 -*-
"""
SynthesizerAgent — clusters claims; aggregates SUPPORT vs ATTACK; emits a consensus snapshot.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class SynthesizerAgent(BaseAgent):
    """Agent that synthesizes multiple analyses into consensus."""

    agent_type = AgentType.SYNTHESIZER
    default_system_prompt = """You are a synthesis expert.

Instructions:
1) Summarize prior analyses into SUPPORT vs ATTACK points tied to concrete steps/claims.
2) De-duplicate and cluster by claim.
3) Discount rhetorical fluency; prefer formally validated steps and sourced evidence.
4) Produce a concise decision rationale.

Output:
- Supporting points (claim → brief reason)
- Attacking points (claim → brief reason)
- Consensus snapshot: #support vs #attack, major point of contention
- Issues: [logical_error, context_unfaithful, parametric_knowledge_error, retrieval_attribution_error, bias:{position|length|confirmation|selection|cultural}]
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

        if previous_analyses:
            prompt += "Previous analyses to synthesize:\n\n"
            for i, analysis in enumerate(previous_analyses):
                prompt += f"Agent {i+1} ({analysis.agent_name}):\n"
                prompt += f"Verdict: {analysis.verdict.value}, Confidence: {analysis.confidence}%\n"
                prompt += f"Key points: {analysis.analysis[:200]}...\n\n"

        prompt += """Synthesize all analyses into SUPPORT vs ATTACK structure as specified."""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse the synthesis response."""
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

        # Extract support vs attack points
        evidence = self._extract_support_attack_evidence(response_text)

        # Extract consensus snapshot
        reasoning_steps = self._extract_consensus_summary(response_text)

        # Extract identified issues
        limitations = self._extract_synthesis_issues(response_text)

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

    def _extract_support_attack_evidence(self, response: str) -> List[Evidence]:
        """Extract supporting and attacking points as evidence."""
        evidence_list = []

        # Extract supporting points
        support_section = self._extract_section(response, 'supporting points', 'attacking points')
        if support_section:
            for line in support_section.split('\n'):
                if '→' in line or '-' in line:
                    evidence_list.append(Evidence(
                        source="synthesis_support",
                        content=line.strip(),
                        relevance_score=0.9
                    ))

        # Extract attacking points
        attack_section = self._extract_section(response, 'attacking points', 'consensus')
        if attack_section:
            for line in attack_section.split('\n'):
                if '→' in line or '-' in line:
                    evidence_list.append(Evidence(
                        source="synthesis_attack",
                        content=line.strip(),
                        relevance_score=0.8
                    ))

        return evidence_list

    def _extract_consensus_summary(self, response: str) -> List[str]:
        """Extract consensus snapshot."""
        consensus_items = []

        # Look for consensus section
        import re
        consensus_pattern = r'consensus.*?:(.*?)(?:issues:|final verdict:|$)'
        match = re.search(consensus_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            consensus_text = match.group(1)
            # Extract support vs attack counts
            count_pattern = r'(\d+)\s*support.*?(\d+)\s*attack'
            count_match = re.search(count_pattern, consensus_text, re.IGNORECASE)
            if count_match:
                consensus_items.append(f"Vote: {count_match.group(1)} support vs {count_match.group(2)} attack")
            
            # Extract major contention points
            if 'contention' in consensus_text.lower():
                lines = consensus_text.split('\n')
                for line in lines:
                    if 'contention' in line.lower():
                        consensus_items.append(line.strip())

        return consensus_items

    def _extract_synthesis_issues(self, response: str) -> List[str]:
        """Extract identified issues from synthesis."""
        issues = []
        
        issue_types = [
            'logical_error', 'context_unfaithful', 'parametric_knowledge_error',
            'retrieval_attribution_error', 'bias:'
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'issues:' in line_lower:
                # Extract the full issues line
                issues.append(line.strip())
            else:
                # Check for individual issue mentions
                for issue_type in issue_types:
                    if issue_type in line_lower:
                        issues.append(f"Issue: {line.strip()}")
                        break
        
        return issues[:5]

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract a section between two markers."""
        import re
        pattern = f'{start_marker}(.*?)(?:{end_marker}|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
