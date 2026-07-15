"""LLM risk-scoring for each archetype: one call per archetype (offline, not in the RL hot
loop), constrained to JSON via pydantic validation, disk-cached by prompt hash so re-runs
are free and deterministic. Malformed output retries once, then falls back to a hardcoded
default so a bad LLM response can never silently corrupt a training run.
"""

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from llm.llm_client import LocalLLMClient

CACHE_PATH = Path("/home/user5/Desktop/MS THESIS/data/archetypes/llm_risk_cache.json")
ARCHETYPES_PATH = Path("/home/user5/Desktop/MS THESIS/data/archetypes/archetypes.json")

DEFAULT_RISK = {"risk": 5, "rationale": "default fallback: LLM response could not be parsed"}


class RiskScore(BaseModel):
    risk: int = Field(ge=0, le=10)
    rationale: str


PROMPT_TEMPLATE = """You are an ICS (industrial control system) network security analyst. \
Given the statistical summary of a cluster of network flows below, assess how risky this \
traffic pattern is for an industrial control system (PLC/SCADA) environment.

Cluster statistics ({n_samples} flows, {attack_rate:.1%} historically confirmed as attack traffic):
- SYN rate: {sSynRate}, ACK rate: {sAckRate}, FIN rate: {sFinRate}, RST rate: {sRstRate}
- Fragment rate: {sFragmentRate}
- Duration: {duration}s, packets/flow: {sPackets}, avg bytes/packet: {sBytesAvg}, load: {sLoad}

Respond with ONLY a JSON object: {{"risk": <integer 0-10, where 10 is the most dangerous \
to physical process safety>, "rationale": "<one sentence>"}}."""


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def score_archetype(client: LocalLLMClient, archetype: dict, cache: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        n_samples=archetype["n_samples"], attack_rate=archetype["attack_rate"],
        **archetype["descriptive_stats"],
    )
    key = _prompt_hash(prompt)
    if key in cache:
        return cache[key]

    result = None
    for attempt in range(2):
        raw = client.generate(prompt, max_new_tokens=150)
        parsed = _extract_json(raw)
        if parsed is not None:
            try:
                result = RiskScore(**parsed).model_dump()
                break
            except ValidationError:
                continue
    if result is None:
        result = DEFAULT_RISK

    cache[key] = result
    return result


def score_all_archetypes():
    archetypes = json.loads(ARCHETYPES_PATH.read_text())
    cache = _load_cache()
    client = LocalLLMClient()

    for a in archetypes:
        score = score_archetype(client, a, cache)
        a["llm_risk"] = score["risk"]
        a["llm_rationale"] = score["rationale"]
        print(f"cluster {a['cluster_id']:2d} (attack_rate={a['attack_rate']:.2f}): "
              f"risk={score['risk']}  {score['rationale'][:80]}")

    _save_cache(cache)
    ARCHETYPES_PATH.write_text(json.dumps(archetypes, indent=2))
    return archetypes


if __name__ == "__main__":
    score_all_archetypes()
