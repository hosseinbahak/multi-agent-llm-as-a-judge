# -*- coding: utf-8 -*-
"""
ChallengerAgent — elicits the minimum follow-ups to close evidence gaps; labels NEEDS-EVIDENCE.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class ChallengerAgent(BaseAgent):
    """Agent that identifies gaps and asks minimum follow-up questions."""

    agent_type = AgentType.CHALLENGER
    default_system_prompt = """You are a probing challenger.

Rules:
- Prefer UNKNOWN/NEEDS-EVIDENCE over guessing.
- Ignore answer length/position; judge content only.

Tasks:
1) Identify GAPS/Ambiguities that prevent a reliable judgment.
2) Ask the MINIMUM SET of follow-up QUESTIONS that, if answered, would close each gap.
3) For each gap, label the required EVIDENCE TYPE (citation, numeric derivation, definitional rule, temporal fact).
4) State whether the current answer should be treated as UNKNOWN/NEEDS-EVIDENCE.

Output:
- Gaps → Follow-up questions → Evidence type
- Limitations (concise)
- Issues: [instruction_misalignment, context_unfaithful, parametric_knowledge_error]
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

Answer to challenge: {answer}

"""
        if context:
            prompt += f"Context: {context}\n\n"

        prompt += """Challenge this answer by identifying gaps and required evidence.

Follow the exact output format specified."""

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
        
        # Challenger prefers UNCERTAIN when evidence is lacking
        if 'needs-evidence' in response_text.lower() or 'unknown' in response_text.lower():
            verdict = Verdict.UNCERTAIN
        elif verdict_bool is True:
            verdict = Verdict.CORRECT
        elif verdict_bool is False:
            verdict = Verdict.INCORRECT
        else:
            verdict = Verdict.UNCERTAIN

        # Extract confidence (lower for gaps)
        confidence = self._extract_confidence(response_text)
        
        # Extract gaps with evidence requirements
        gaps_and_questions = self._extract_gaps_with_evidence_types(response_text)
        
        # Extract follow-up questions
        questions = self._extract_minimum_questions(response_text)
        
        # Extract limitations
        limitations = self._extract_limitations(response_text)

        return AgentAnalysis(
            agent_type=self.agent_type,
            agent_name=self.name,
            model_used=model_id,
            analysis=response_text,
            verdict=verdict,
            confidence=confidence,
            reasoning_steps=questions,
            limitations=limitations + gaps_and_questions,
            tokens_used=tokens_used,
            cost=cost,
            processing_time_ms=processing_time_ms
        )

    def _extract_gaps_with_evidence_types(self, response: str) -> List[str]:
        """Extract gaps with their required evidence types."""
        gaps = []
        
        import re
        # Pattern for gap → question → evidence type
        pattern = r'gap[s]?.*?→.*?→\s*(\w+\s*\w*)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            gaps.append(f"Requires: {match.strip()}")
        
        # Also look for explicit evidence type mentions
        evidence_types = ['citation', 'numeric derivation', 'definitional rule', 'temporal fact']
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            for ev_type in evidence_types:
                if ev_type in line_lower:
                    gaps.append(line.strip())
                    break
        
        return gaps[:5]

    def _extract_minimum_questions(self, response: str) -> List[str]:
        """Extract the minimum set of follow-up questions."""
        questions = []
        
        lines = response.split('\n')
        for line in lines:
            # Look for question marks and ensure it's substantive
            if '?' in line and len(line.strip()) > 15:
                # Filter out rhetorical questions
                if not any(rhet in line.lower() for rhet in ['wouldn\'t', 'isn\'t it', 'right?']):
                    questions.append(line.strip())
        
        # Also extract from structured format
        import re
        structured_pattern = r'(?:follow-up|question)[\s:]+(.*?\?)'
        matches = re.findall(structured_pattern, response, re.IGNORECASE)
        questions.extend(matches)
        
        # Deduplicate and return minimum set
        seen = set()
        unique_questions = []
        for q in questions:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_questions.append(q)
        
        return unique_questions[:5]

    def _extract_limitations(self, response: str) -> List[str]:
        """Extract identified limitations."""
        limitations = []
        
        limit_keywords = ['limitation', 'insufficient', 'unclear', 'ambiguous', 'missing', 'gap']
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in limit_keywords):
                limitations.append(line.strip())
        
        return limitations[:3]
