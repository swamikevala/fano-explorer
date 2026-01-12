"""
DeepSeek V2 Math Prover integration for mathematical claim verification.

Provides rigorous mathematical verification that complements the
general LLM review panel. DeepSeek ONLY evaluates mathematical
correctness - not structure, naturalness, or insight quality.
"""

import asyncio
import os
import time
from typing import Optional

from shared.logging import get_logger

from .models import VerificationResult

log = get_logger("explorer", "review_panel.deepseek")


# Prompt for extracting mathematical claims
CLAIM_EXTRACTION_PROMPT = """You are a mathematical claim extractor. Your task is to identify
specific, verifiable mathematical claims from the following text.

TEXT:
{insight}

TASK:
Extract each distinct mathematical claim that could be proven or refuted.
Focus on:
- Numerical equalities or relationships (e.g., "72 + 12 = 84")
- Cardinality claims (e.g., "exactly 7 points")
- Structural claims (e.g., "isomorphic to PSL(2,7)")
- Properties (e.g., "forms a group under composition")
- Existence/uniqueness claims (e.g., "there exists exactly one...")

Do NOT extract:
- Vague statements ("this is elegant")
- Interpretive claims ("this explains why...")
- Metaphorical statements

For each claim, state it precisely in a form that could be verified.

FORMAT:
CLAIM: [precise mathematical statement]
CLAIM: [precise mathematical statement]
...

If no mathematical claims found, respond:
NO_CLAIMS: [explanation]
"""


# Prompt for verifying a single claim
VERIFICATION_PROMPT = """You are a rigorous mathematical proof assistant. Your task is to
VERIFY or REFUTE the following mathematical claim.

CLAIM TO VERIFY:
{claim}

CONTEXT (definitions and axioms that may be assumed):
{context}

INSTRUCTIONS:
1. Restate the claim in precise mathematical notation if possible
2. Attempt to PROVE the claim, OR
3. Find a specific COUNTEREXAMPLE that refutes it
4. If the claim involves arithmetic, verify all calculations
5. If the claim references known theorems or structures, verify accuracy

Be rigorous. Do not accept claims without proof. Do not reject claims without counterexample.

RESPOND IN THIS EXACT FORMAT:

PRECISE_STATEMENT:
[Restate claim precisely, using mathematical notation where appropriate]

VERIFICATION_APPROACH:
[1-2 sentences describing your proof strategy or search for counterexample]

PROOF_OR_COUNTEREXAMPLE:
[If VERIFIED: Complete proof with clear logical steps]
[If REFUTED: Specific counterexample with verification]
[If UNCLEAR: Explanation of what's missing or ambiguous]

VERDICT: VERIFIED | REFUTED | UNCLEAR | NOT_MATHEMATICAL

CONFIDENCE: [0.0 to 1.0 - how certain you are of this verdict]

CONCERNS:
[Any caveats, assumptions required, or edge cases]
"""


# Prompt for generating formal proofs (post-blessing augmentation)
FORMAL_PROOF_PROMPT = """You are a mathematical proof writer. Generate a detailed,
formal proof for the following claim.

CLAIM:
{claim}

CONTEXT:
{context}

REQUIREMENTS:
- Write a complete, rigorous proof
- State all assumptions and definitions used
- Number each logical step
- Cite known theorems when used
- Conclude with QED or ∎

FORMAT:

THEOREM:
[Precise statement of what is being proved]

DEFINITIONS:
[Any definitions needed]

ASSUMPTIONS:
[Any assumptions made]

FORMAL_PROOF:
1. [First step] - [Justification]
2. [Second step] - [Justification]
...
n. [Conclusion] ∎

NOTES:
[Any additional remarks, alternative approaches, or generalizations]
"""


class DeepSeekVerifier:
    """
    Mathematical claim verifier using DeepSeek Prover V2 via OpenRouter.

    Responsibilities:
    - Extract mathematical claims from insights
    - Verify claims with formal proofs or counterexamples
    - Report verification results for review panel
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = "deepseek/deepseek-prover-v2",
    ):
        """
        Initialize DeepSeek verifier via OpenRouter.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            endpoint: API endpoint URL (defaults to OpenRouter)
            model: Model identifier to use (OpenRouter model path)
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.endpoint = endpoint or "https://openrouter.ai/api/v1"
        self.model = model
        self.client = None
        self._initialized = False

    def _ensure_client(self):
        """Lazily initialize the API client."""
        if self._initialized:
            return

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key to DeepSeekVerifier."
            )

        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.endpoint,
                default_headers={
                    "HTTP-Referer": "https://github.com/fano-explorer",
                    "X-Title": "Fano Explorer Math Verification",
                }
            )
            self._initialized = True
            log.info(f"[deepseek] Initialized via OpenRouter with model {self.model}")
        except ImportError:
            raise ImportError(
                "openai package required for DeepSeek integration. "
                "Install with: pip install openai"
            )

    def is_available(self) -> bool:
        """Check if DeepSeek API is available."""
        try:
            self._ensure_client()
            return True
        except (ValueError, ImportError) as e:
            log.warning(f"[deepseek] Not available: {e}")
            return False

    async def verify_insight(
        self,
        insight: str,
        context: str = "",
        extract_claims: bool = True,
    ) -> VerificationResult:
        """
        Verify mathematical claims in an insight.

        Args:
            insight: The insight text to verify
            context: Background axioms/definitions for context
            extract_claims: Whether to first extract individual claims

        Returns:
            VerificationResult with verdict and evidence
        """
        self._ensure_client()
        start_time = time.time()

        log.info(f"[deepseek] Verifying insight ({len(insight)} chars)")

        try:
            # Step 1: Extract mathematical claims
            if extract_claims:
                claims = await self._extract_claims(insight)
                log.info(f"[deepseek] Extracted {len(claims)} claims")
            else:
                claims = [insight]

            if not claims:
                return VerificationResult(
                    verdict="not_applicable",
                    precise_statement="",
                    confidence=1.0,
                    concerns=["No mathematical claims found to verify"],
                    model_used=self.model,
                    verification_time_seconds=time.time() - start_time,
                )

            # Step 2: Verify each claim
            verified = []
            refuted = []
            unclear = []
            all_proofs = []
            all_counterexamples = []
            all_concerns = []

            for claim in claims:
                result = await self._verify_single_claim(claim, context)

                if result["verdict"] == "verified":
                    verified.append(claim)
                    if result.get("proof"):
                        all_proofs.append(f"### {claim}\n\n{result['proof']}")
                elif result["verdict"] == "refuted":
                    refuted.append(claim)
                    if result.get("counterexample"):
                        all_counterexamples.append(f"### {claim}\n\n{result['counterexample']}")
                else:
                    unclear.append(claim)
                    if result.get("concerns"):
                        all_concerns.extend(result["concerns"])

            # Step 3: Determine overall verdict
            if refuted:
                verdict = "refuted"
                confidence = 0.9  # High confidence if any claim refuted
            elif verified and not unclear:
                verdict = "verified"
                confidence = 0.85
            elif verified:
                verdict = "verified"  # Some verified, some unclear
                confidence = 0.6
            elif unclear:
                verdict = "unclear"
                confidence = 0.3
            else:
                verdict = "not_applicable"
                confidence = 1.0

            return VerificationResult(
                verdict=verdict,
                precise_statement="; ".join(claims),
                formal_proof="\n\n".join(all_proofs) if all_proofs else None,
                counterexample="\n\n".join(all_counterexamples) if all_counterexamples else None,
                confidence=confidence,
                concerns=all_concerns,
                claims_extracted=claims,
                claims_verified=verified,
                claims_refuted=refuted,
                claims_unclear=unclear,
                model_used=self.model,
                verification_time_seconds=time.time() - start_time,
            )

        except Exception as e:
            log.error(f"[deepseek] Verification failed: {e}")
            return VerificationResult(
                verdict="unclear",
                precise_statement=insight[:200],
                confidence=0.0,
                concerns=[f"Verification error: {str(e)}"],
                model_used=self.model,
                verification_time_seconds=time.time() - start_time,
            )

    async def _extract_claims(self, insight: str) -> list[str]:
        """
        Extract individual mathematical claims from insight.

        Uses DeepSeek to identify specific verifiable claims.
        """
        prompt = CLAIM_EXTRACTION_PROMPT.format(insight=insight)

        response = await self._send_request(prompt)

        # Parse claims from response
        claims = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("CLAIM:"):
                claim = line[6:].strip()
                if claim:
                    claims.append(claim)
            elif line.startswith("- ") or line.startswith("• "):
                claim = line[2:].strip()
                if claim and len(claim) > 10:  # Skip trivial entries
                    claims.append(claim)

        return claims[:10]  # Limit to 10 claims per insight

    async def _verify_single_claim(
        self,
        claim: str,
        context: str,
    ) -> dict:
        """
        Verify a single mathematical claim.

        Returns dict with verdict, proof/counterexample, concerns.
        """
        prompt = VERIFICATION_PROMPT.format(
            claim=claim,
            context=context or "(no additional context)",
        )

        response = await self._send_request(prompt)

        # Parse response
        result = {
            "verdict": "unclear",
            "proof": None,
            "counterexample": None,
            "concerns": [],
        }

        # Extract verdict
        if "VERDICT:" in response:
            verdict_line = response.split("VERDICT:")[1].split("\n")[0].strip().upper()
            if "VERIFIED" in verdict_line:
                result["verdict"] = "verified"
            elif "REFUTED" in verdict_line:
                result["verdict"] = "refuted"
            elif "NOT_MATHEMATICAL" in verdict_line or "NOT MATHEMATICAL" in verdict_line:
                result["verdict"] = "not_applicable"
            else:
                result["verdict"] = "unclear"

        # Extract proof
        if "PROOF:" in response or "PROOF_OR_COUNTEREXAMPLE:" in response:
            start_marker = "PROOF:" if "PROOF:" in response else "PROOF_OR_COUNTEREXAMPLE:"
            start = response.find(start_marker) + len(start_marker)
            # Find next section or end
            end_markers = ["VERDICT:", "CONFIDENCE:", "CONCERNS:"]
            end = len(response)
            for marker in end_markers:
                pos = response.find(marker, start)
                if pos != -1 and pos < end:
                    end = pos
            proof_text = response[start:end].strip()
            if result["verdict"] == "verified":
                result["proof"] = proof_text
            elif result["verdict"] == "refuted":
                result["counterexample"] = proof_text

        # Extract concerns
        if "CONCERNS:" in response:
            concerns_start = response.find("CONCERNS:") + 9
            concerns_text = response[concerns_start:].strip()
            for line in concerns_text.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("• "):
                    result["concerns"].append(line[2:])
                elif line and not line.startswith("[") and len(line) > 5:
                    result["concerns"].append(line)
                    break  # Usually just one line if not bulleted

        return result

    async def _send_request(self, prompt: str) -> str:
        """
        Send request to DeepSeek API.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Deterministic for math
            max_tokens=4096,
        )
        return response.choices[0].message.content

    async def generate_formal_proof(
        self,
        claim: str,
        context: str = "",
    ) -> Optional[str]:
        """
        Generate a detailed formal proof for augmentation.

        Used post-blessing to create proof artifacts.
        More detailed than verification proofs.
        """
        self._ensure_client()
        prompt = FORMAL_PROOF_PROMPT.format(
            claim=claim,
            context=context or "(no additional context)",
        )

        try:
            response = await self._send_request(prompt)

            # Extract the proof section
            if "FORMAL_PROOF:" in response:
                start = response.find("FORMAL_PROOF:") + 13
                end = response.find("QED", start)
                if end == -1:
                    end = response.find("∎", start)
                if end == -1:
                    end = len(response)
                else:
                    end += 3  # Include "QED" or "∎"
                return response[start:end].strip()

            return response

        except Exception as e:
            log.error(f"[deepseek] Proof generation failed: {e}")
            return None


def get_deepseek_verifier(config: dict = None) -> Optional[DeepSeekVerifier]:
    """
    Create a DeepSeekVerifier from config.

    Args:
        config: math_verification config section

    Returns:
        DeepSeekVerifier or None if not configured
    """
    config = config or {}
    math_config = config.get("math_verification", {})

    if not math_config.get("enabled", False):
        log.info("[deepseek] Math verification disabled in config")
        return None

    api_key_env = math_config.get("openrouter_api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env)

    if not api_key:
        log.warning(f"[deepseek] {api_key_env} not set")
        return None

    return DeepSeekVerifier(
        api_key=api_key,
        endpoint=math_config.get("endpoint", "https://openrouter.ai/api/v1"),
        model=math_config.get("model", "deepseek/deepseek-prover-v2"),
    )
