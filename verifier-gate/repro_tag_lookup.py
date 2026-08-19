"""Minimal repro: _find_tag_logprobs returns the WRONG position when the
constrained sample strips to an empty string. No API key, no model, no network.

    python repro_tag_lookup.py
"""
from llm_verifier.fine_grained_reward import extract_score, _find_tag_logprobs

# A real top-20 from a grammar-constrained prefill call (DeepSeek-V4-Flash,
# vLLM). The distribution is healthy: 19 of 20 alternatives are score letters.
ALTS = [(" ", -0.116), (" A", -2.616), (" B", -4.366),
        (" C", -5.054), (" P", -5.054), (" R", -5.491)]

TAG = "<score_A>"
CLOSING = "</" + TAG[1:]


def build(sampled_content):
    """Exactly what _score_tags_by_prefill constructs."""
    letter = sampled_content.strip()          # ' ' -> ''
    text = "analysis" + f"\n{TAG}" + letter + CLOSING
    tokens = [f"\n{TAG}", letter, CLOSING]
    plp = [[(f"\n{TAG}", 0.0)], ALTS, [(CLOSING, 0.0)]]
    return text, tokens, plp


print("case 1 — model sampled 'A' (letter survives .strip())")
t, tok, plp = build("A")
print("   _find_tag_logprobs ->", _find_tag_logprobs(tok, plp, TAG)[:2])
s_ok = extract_score(t, tok, plp, TAG)
print(f"   extract_score      -> {s_ok:.4f}")

print()
print("case 2 — model sampled ' ' (a legal prefix of ' A'; strips to '')")
t, tok, plp = build(" ")
print("   tokens             ->", tok)
print("   _find_tag_logprobs ->", _find_tag_logprobs(tok, plp, TAG))
s_bad = extract_score(t, tok, plp, TAG)
print(f"   extract_score      -> {s_bad:.4f}")

print()
print(f"same distribution, different result: {s_ok:.4f} vs {s_bad:.4f}")
assert s_ok != 0.5, "case 1 should be a real score"
if s_bad == 0.5:
    print("REPRODUCED: the identical distribution is discarded and scored 0.5.")
else:
    print("FIXED: both cases now use the distribution.")
