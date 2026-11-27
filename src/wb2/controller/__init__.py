"""WB2 stopping policies over saved MC-Dropout logit banks (implementation.md
§3 C1–C4, Member C). Inference-time inputs are logits and a frozen
temperature only — no module here reads a true label to make a decision.
"""
