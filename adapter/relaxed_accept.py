"""Relaxed greedy acceptance for DSpark (N1). Default OFF. NOT EXACT: greedy output changes.

The idea is from docs/history.md (accepting any draft token inside the target's top-p gave +20 %
accepted tokens on V4.1). README "Fast mode (not exact)" has the measured speed and fidelity.

The stock greedy rule accepts draft i while candidates[:, i+1] == argmax(target row i)
(dflash_utils.compute_dflash_correct_drafts_and_bonus). Here the target-predict table is rebuilt
first: at row i the draft token replaces the argmax when it passes the rule below, so the engine's
own kernels then count it as a match, cap it (verify cap) and take the bonus from the first row
that fails. Only all-greedy batches use this path; sampled rows are unchanged.

DSV41_RELAXED_ACCEPT (comma-separated, each optional; a draft must pass every rule given):
  ratio:R   p(draft) >= R * p(argmax)        (R in (0, 1]; logit gap <= ln(1/R))
  topp:P    target mass strictly above p(draft) < P   (draft is in the top-p set; P in (0, 1))
Example: "ratio:0.3" or "topp:0.5,ratio:0.1".

Needs full-vocab logits: refuses to start with DSV41_CERT_HEAD on (the certified head writes -inf
over most of the vocabulary in greedy verify steps). Graph-capturable: torch ops only, no syncs.
"""
import math
import os

import torch

SPEC = os.environ.get("DSV41_RELAXED_ACCEPT", "").strip()
ENABLED = SPEC not in ("", "0", "off", "false")
_cfg = {"ratio": None, "topp": None, "calls": 0}


def parse(spec):
    cfg = {"ratio": None, "topp": None}
    for part in spec.split(","):
        k, _, v = part.strip().partition(":")
        if k not in cfg or not v:
            raise ValueError(f"DSV41_RELAXED_ACCEPT: bad term {part!r}")
        cfg[k] = float(v)
    if cfg["ratio"] is None and cfg["topp"] is None:
        raise ValueError("DSV41_RELAXED_ACCEPT: no rule")
    if cfg["ratio"] is not None and not 0.0 < cfg["ratio"] <= 1.0:
        raise ValueError("DSV41_RELAXED_ACCEPT: ratio must be in (0, 1]")
    if cfg["topp"] is not None and not 0.0 < cfg["topp"] < 1.0:
        raise ValueError("DSV41_RELAXED_ACCEPT: topp must be in (0, 1)")
    return cfg


def relaxed_predict(candidates, target_logits, stride, target_predict):
    """candidates [bs, stride], target_logits [bs*stride, V], target_predict [bs, stride] (argmax).
    Returns a new [bs, stride] table: draft token where it passes the rule, else the argmax."""
    bs = candidates.shape[0]
    g = stride - 1
    lg = target_logits.view(bs, stride, -1)[:, :g].float()            # rows that verify drafts
    drafts = candidates[:, 1:].to(torch.int64)                          # [bs, g]
    l_d = lg.gather(-1, drafts.unsqueeze(-1)).squeeze(-1)
    l_max = lg.gather(-1, target_predict[:, :g].to(torch.int64).unsqueeze(-1)).squeeze(-1)
    ok = torch.isfinite(l_d)
    if _cfg["ratio"] is not None:
        ok = ok & ((l_d - l_max) >= math.log(_cfg["ratio"]))
    if _cfg["topp"] is not None:
        lse = torch.logsumexp(lg, dim=-1, keepdim=True)
        probs = torch.exp(lg - lse)
        above = torch.where(lg > l_d.unsqueeze(-1), probs, torch.zeros_like(probs)).sum(-1)
        ok = ok & (above < _cfg["topp"])
    out = target_predict.clone()
    out[:, :g] = torch.where(ok, drafts.to(out.dtype), target_predict[:, :g])
    return out


def _make(accept_mod, triton_variant):
    row_argmax = accept_mod._row_argmax
    correct_and_bonus = accept_mod.compute_dflash_correct_drafts_and_bonus
    cap = accept_mod.CapCorrectLen
    gather_bonus = accept_mod.gather_row_bonus_triton

    def accept(*, candidates, target_logits, verify_num_draft_tokens, cutoff_verify_lens=None):
        bs = candidates.shape[0]
        pred = row_argmax(target_logits).view(bs, verify_num_draft_tokens)
        pred = relaxed_predict(candidates, target_logits, verify_num_draft_tokens, pred)
        correct_len, bonus = correct_and_bonus(candidates=candidates, target_predict=pred)
        cap_trim_lens = torch.zeros_like(correct_len)
        if cutoff_verify_lens is not None:
            correct_len, cap_trim_lens = cap.execute(correct_len=correct_len,
                                                     verify_lens=cutoff_verify_lens)
            if triton_variant:
                bonus = gather_bonus(table=pred, idx=correct_len)
            else:
                rows = torch.arange(bs, device=pred.device)
                bonus = pred[rows, correct_len.to(torch.long)].to(torch.int64)
        return correct_len, bonus, cap_trim_lens

    return accept


def install_accept(accept_mod):
    """sglang.kernels.ops.speculative.dspark.dspark_accept: AcceptGreedy's two back ends."""
    if not ENABLED:
        return
    if os.environ.get("DSV41_CERT_HEAD", "0").strip().lower() not in ("0", "off", "false", "", "no"):
        raise RuntimeError("DSV41_RELAXED_ACCEPT needs full logits: set DSV41_CERT_HEAD=0")
    _cfg.update(parse(SPEC))
    for name in ("_row_argmax", "compute_dflash_correct_drafts_and_bonus", "CapCorrectLen",
                 "gather_row_bonus_triton", "accept_greedy", "accept_greedy_triton"):
        if not hasattr(accept_mod, name):
            raise RuntimeError(f"DSV41_RELAXED_ACCEPT: engine symbol {name} drifted")
    accept_mod.accept_greedy = _make(accept_mod, triton_variant=False)
    accept_mod.accept_greedy_triton = _make(accept_mod, triton_variant=True)
    print(f"[relaxed_accept] ON {_cfg} (NOT EXACT)", flush=True)


def install_verify(verify_mod):
    """sglang.srt.speculative.dspark_components.dspark_verify imported accept_greedy_triton by name
    (the in-graph epilogue's accept): rebind it to the relaxed one."""
    if not ENABLED:
        return
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    if dspark_accept.accept_greedy_triton.__module__ != __name__:
        raise RuntimeError("DSV41_RELAXED_ACCEPT: dspark_accept not patched before dspark_verify")
    verify_mod.accept_greedy_triton = dspark_accept.accept_greedy_triton
    print("[relaxed_accept] dspark_verify epilogue rebound", flush=True)
