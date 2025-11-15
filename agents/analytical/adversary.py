# -*- coding: utf-8 -*-
"""
AdversaryAgent — probes hidden assumptions, fallacies, counterexamples, paraphrase stability.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class AdversaryAgent(BaseAgent):
    """Agent that exposes hidden assumptions and tests robustness."""

    agent_type = AgentType.ADVERSARY
    default_system_prompt = """You are an adversarial examiner.

Rules:
- Treat explanations as claims, not proof.
- Ignore length/position; score content only.

Do the following:
1) List HIDDEN ASSUMPTIONS and potential FALLACIES (name the fallacy type).
2) Generate 2 counterexamples or edge-cases that would break the answer if true.
3) Paraphrase the QUESTION twice (different wording/order). Would the original reasoning still hold? Report STABLE/UNSTABLE with reasons.
4) Point out any instruction the answer failed to follow.
5) If CONTEXT is present, check faithfulness; else warn about reliance on parametric knowledge.

Output:
- Weaknesses: bullet list (each with WHY-it-matters)
- Paraphrase stability: STABLE/UNSTABLE (+notes)
- Missing instructions (if any)
- Issues: [logical_error, instruction_misalignment, context_unfaithful, parametric_knowledge_error, bias:{position|length|confirmation|selection|cultural}]
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

Answer to examine: {answer}

"""
        if context:
            prompt += f"Context: {context}\n\n"

        prompt += """Adversarially examine this answer following the exact output format specified."""

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

        # Extract criticisms as evidence
        criticisms = self._extract_adversarial_findings(response_text)

        # Extract assumptions and fallacies
        assumptions = self._extract_assumptions_and_fallacies(response_text)

        # Extract stability assessment
        reasoning_steps = self._extract_stability_assessment(response_text)

        return AgentAnalysis(
            agent_type=self.agent_type,
            agent_name=self.name,
            model_used=model_id,
            analysis=response_text,
            verdict=verdict,
            confidence=confidence,
            evidence=criticisms,
            assumptions=assumptions,
            reasoning_steps=reasoning_steps,
            tokens_used=tokens_used,
            cost=cost,
            processing_time_ms=processing_time_ms
        )

    def _extract_adversarial_findings(self, response: str) -> List[Evidence]:
        """Extract weaknesses and counterexamples."""
        evidence_list = []

        weakness_keywords = ['weakness', 'flaw', 'fallacy', 'counterexample', 'edge-case', 'breaks if']

        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in weakness_keywords):
                # Adversarial evidence has lower relevance score
                evidence_list.append(Evidence(
                    source="adversarial_examination",
                    content=line.strip(),
                    relevance_score=0.8
                ))

        return evidence_list

    def _extract_assumptions_and_fallacies(self, response: str) -> List[str]:
        """Extract hidden assumptions and named fallacies."""
        findings = []

        # Look for assumptions and fallacies
        patterns = [
            r'hidden assumption[s]?:\s*(.+)',
            r'assume[s]?\s+that\s+(.+)',
            r'fallacy:\s*(.+)',
            r'(\w+)\s+fallacy'
        ]

        import re
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                findings.append(match.strip() if isinstance(match, str) else match[0].strip())

        return findings[:5]

    def _extract_stability_assessment(self, response: str) -> List[str]:
        """Extract paraphrase stability assessment."""
        stability_info = []

        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'stable' in line_lower or 'unstable' in line_lower or 'paraphrase' in line_lower:
                stability_info.append(line.strip())

        return stability_info[:3]
