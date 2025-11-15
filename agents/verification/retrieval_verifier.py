# -*- coding: utf-8 -*-
"""
RetrievalVerifierAgent — atomic claim checks vs context; SUPPORTED/CONTRADICTED/NOT_FOUND + attribution.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class RetrievalVerifierAgent(BaseAgent):
    """Agent that verifies retrieval faithfulness and attribution."""

    agent_type = AgentType.RETRIEVAL_VERIFIER
    default_system_prompt = """You verify retrieval faithfulness.

Behaviors:
- If CONTEXT is present, treat it as ground truth for claims.
- If no CONTEXT, flag dependence on PARAMETRIC KNOWLEDGE; avoid certainty unless universally true.

Procedure:
1) Extract ATOMIC claims from the answer.
2) For each claim, check against CONTEXT and label:
   {SUPPORTED, CONTRADICTED, NOT_FOUND} with a quote + line/span reference.
3) If the answer cites external facts without explicit attribution, flag retrieval_attribution_error.
4) Summarize overall faithfulness.

Output:
- Claim checks: [claim → status → evidence/ref]
- Attribution notes (if any)
- Issues: [context_unfaithful, hallucination, retrieval_attribution_error, parametric_knowledge_error]
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

Answer to verify: {answer}

"""
        if context:
            prompt += f"Context (ground truth):\n{context}\n\n"
        else:
            prompt += "Context: NOT PROVIDED - Flag any parametric knowledge dependencies.\n\n"

        prompt += """Verify retrieval faithfulness following the exact procedure specified."""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse the verification response."""
        # Extract verdict based on faithfulness
        verdict = self._determine_faithfulness_verdict(response_text)
        
        # Extract confidence
        confidence = self._extract_confidence(response_text)
        
        # Extract claim verification results
        evidence = self._extract_claim_verifications(response_text)
        
        # Extract attribution issues
        limitations = self._extract_attribution_issues(response_text)
        
        # Extract overall faithfulness summary
        reasoning_steps = self._extract_faithfulness_summary(response_text)

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

    def _determine_faithfulness_verdict(self, response: str) -> Verdict:
        """Determine verdict based on faithfulness checking."""
        response_lower = response.lower()
        
        # Count verification statuses
        supported_count = response_lower.count('supported')
        contradicted_count = response_lower.count('contradicted')
        not_found_count = response_lower.count('not_found') + response_lower.count('not found')
        
        # Check for major issues
        if 'hallucination' in response_lower or contradicted_count > 0:
            return Verdict.INCORRECT
        elif 'context_unfaithful' in response_lower or not_found_count > 2:
            return Verdict.UNCERTAIN
        elif supported_count > 3 and contradicted_count == 0:
            return Verdict.CORRECT
        else:
            # Default extraction
            verdict_bool = self._extract_verdict(response)
            if verdict_bool is True:
                return Verdict.CORRECT
            elif verdict_bool is False:
                return Verdict.INCORRECT
            else:
                return Verdict.UNCERTAIN

    def _extract_claim_verifications(self, response: str) -> List[Evidence]:
        """Extract atomic claim verification results."""
        evidence_list = []
        
        import re
        # Pattern for claim → status → evidence
        claim_pattern = r'(?:claim:|-)?\s*(.+?)\s*→\s*(SUPPORTED|CONTRADICTED|NOT_FOUND|NOT FOUND)\s*(?:→\s*(.+))?'
        matches = re.findall(claim_pattern, response, re.IGNORECASE)
        
        for claim, status, evidence_text in matches:
            relevance = 0.9 if 'SUPPORTED' in status.upper() else 0.5
            evidence_content = f"{claim.strip()} - {status.upper()}"
            if evidence_text:
                evidence_content += f": {evidence_text.strip()}"
            
            evidence_list.append(Evidence(
                source=f"verification_{status.lower()}",
                content=evidence_content,
                relevance_score=relevance
            ))
        
        # Also look for quoted evidence
        quote_pattern = r'"([^"]+)".*?(?:line|span|ref)[:\s]*(\d+|[\d-]+)'
        quote_matches = re.findall(quote_pattern, response)
        for quote, ref in quote_matches:
            evidence_list.append(Evidence(
                source=f"context_quote_line_{ref}",
                content=quote,
                relevance_score=0.95
            ))
        
        return evidence_list

    def _extract_attribution_issues(self, response: str) -> List[str]:
        """Extract attribution and parametric knowledge issues."""
        issues = []
        
        issue_keywords = [
            'retrieval_attribution_error', 'parametric_knowledge_error',
            'context_unfaithful', 'hallucination', 'no attribution',
            'missing citation', 'unsourced claim'
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in issue_keywords:
                if keyword in line_lower:
                    issues.append(line.strip())
                    break
        
        return issues[:5]

    def _extract_faithfulness_summary(self, response: str) -> List[str]:
        """Extract overall faithfulness assessment."""
        summary_items = []
        
        # Look for summary section
        summary_keywords = ['overall', 'summary', 'faithfulness', 'conclusion']
        
        lines = response.split('\n')
        capturing = False
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in summary_keywords):
                capturing = True
            elif capturing and line.strip():
                summary_items.append(line.strip())
                if len(summary_items) >= 3:
                    break
        
        return summary_items
