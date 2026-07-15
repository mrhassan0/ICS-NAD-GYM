"""Base reward table for ICSNADGym (no LLM shaping — that's layered on later as a
potential-based term). Actions: 0=ALLOW, 1=ALERT, 2=BLOCK.

Asymmetric by design: missing a real attack (Attack+ALLOW) is the worst outcome, reflecting
the physical-process risk of undetected ICS intrusions; false alarms are penalized but far
less severely.
"""

ALLOW, ALERT, BLOCK = 0, 1, 2

BASE_TABLE = {
    (False, ALLOW): 1.0,
    (False, ALERT): -5.0,
    (False, BLOCK): -10.0,
    (True, ALLOW): -20.0,
    (True, ALERT): 10.0,
    (True, BLOCK): 15.0,
}

QUARANTINE_TRUE_ONGOING_BONUS = 2.0    # per-step bonus while correctly holding an attacker
QUARANTINE_FALSE_ONGOING_PENALTY = -3.0  # per-step cost while wrongly holding a benign source


def base_reward(is_attack: bool, action: int) -> float:
    return BASE_TABLE[(is_attack, action)]


def detection_latency_bonus(flows_since_onset: int) -> float:
    """Extra reward for detecting an attack run quickly after its true onset."""
    return 5.0 / (1 + flows_since_onset)


def quarantine_ongoing_reward(was_correct_block: bool) -> float:
    return QUARANTINE_TRUE_ONGOING_BONUS if was_correct_block else QUARANTINE_FALSE_ONGOING_PENALTY


SHAPING_GAMMA = 0.99  # matches ICSNADGym/PPO's discount factor, per Ng/Harada/Russell 1999


def potential_based_shaping(phi_s: float, phi_s_next: float) -> float:
    """F(s,s') = γΦ(s') − Φ(s). Policy-invariant by construction: adding this term to any
    reward function never changes the optimal policy, only the speed/ease of finding it —
    so any measured training difference is attributable to the LLM's archetype guidance,
    not a silently altered objective."""
    return SHAPING_GAMMA * phi_s_next - phi_s
