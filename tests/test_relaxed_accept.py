"""CPU test for relaxed_accept.py against a torch copy of the engine's greedy rule.
CPU only, no GPU or engine needed. Run: PYTHONPATH=adapter python3 tests/test_relaxed_accept.py
(the test sets DSV41_RELAXED_ACCEPT and DSV41_CERT_HEAD itself)."""
import importlib
import math
import os
import types

import torch


def stock_correct_and_bonus(*, candidates, target_predict):
    match = candidates[:, 1:] == target_predict[:, :-1]
    correct = match.to(torch.int32).cumprod(dim=1).sum(dim=1).to(torch.int32)
    bonus = target_predict[torch.arange(candidates.shape[0]), correct.long()].to(torch.int64)
    return correct, bonus


class Cap:
    @staticmethod
    def execute(*, correct_len, verify_lens):
        capped = torch.minimum(correct_len, (verify_lens - 1).to(correct_len.dtype))
        return capped, correct_len - capped


def fake_accept_mod():
    m = types.ModuleType("fake_dspark_accept")
    m._row_argmax = lambda x: x.argmax(-1)
    m.compute_dflash_correct_drafts_and_bonus = stock_correct_and_bonus
    m.CapCorrectLen = Cap
    m.gather_row_bonus_triton = lambda *, table, idx: table[torch.arange(table.shape[0]), idx.long()]
    m.accept_greedy = m.accept_greedy_triton = None
    return m


def load(spec, cert="0"):
    os.environ["DSV41_RELAXED_ACCEPT"] = spec
    os.environ["DSV41_CERT_HEAD"] = cert
    ra = importlib.reload(importlib.import_module("relaxed_accept"))
    m = fake_accept_mod()
    ra.install_accept(m)
    return m


def stock(cands, logits, cut=None):
    pred = logits.argmax(-1).view(cands.shape)
    c, b = stock_correct_and_bonus(candidates=cands, target_predict=pred)
    if cut is not None:
        c2, _ = Cap.execute(correct_len=c, verify_lens=cut)
        b = pred[torch.arange(cands.shape[0]), c2.long()]
        c = c2
    return c, b


def main():
    torch.manual_seed(0)
    bs, stride, V = 3, 6, 50
    logits = torch.randn(bs * stride, V) * 3
    pred = logits.argmax(-1).view(bs, stride)
    second = logits.topk(2, -1).indices[:, 1].view(bs, stride)
    # request 0: all drafts are the argmax; request 1: drafts are the 2nd-best; request 2: random
    cands = torch.zeros(bs, stride, dtype=torch.int64)
    cands[0, 1:] = pred[0, :-1]
    cands[1, 1:] = second[1, :-1]
    cands[2, 1:] = torch.randint(0, V, (stride - 1,))
    n = 0
    # 1) ratio:1.0 == stock rule exactly (no ties in random floats), both back ends, with and without cut
    m = load("ratio:1.0")
    for fn in (m.accept_greedy, m.accept_greedy_triton):
        for cut in (None, torch.tensor([3, 6, 2])):
            c, b, _ = fn(candidates=cands, target_logits=logits, verify_num_draft_tokens=stride,
                         cutoff_verify_lens=cut)
            sc, sb = stock(cands, logits, cut)
            assert torch.equal(c, sc) and torch.equal(b, sb), (c, sc, b, sb)
            n += 1
    # 2) ratio:R accepts the 2nd-best exactly when its prob ratio >= R
    lg = logits.view(bs, stride, V)
    r = torch.exp(lg[1, :-1].gather(-1, second[1, :-1, None]).squeeze(-1) - lg[1, :-1].max(-1).values)
    R = float(r.min()) * 0.999
    m = load(f"ratio:{R}")
    c, b, _ = m.accept_greedy(candidates=cands, target_logits=logits, verify_num_draft_tokens=stride)
    assert int(c[0]) == 5 and int(c[1]) == 5, c
    assert int(b[1]) == int(pred[1, 5]), "bonus must be the argmax of the row after the last draft"
    n += 1
    # 3) topp:P: draft accepted iff mass strictly above it < P
    m = load("topp:0.999999")
    c, _, _ = m.accept_greedy(candidates=cands, target_logits=logits, verify_num_draft_tokens=stride)
    assert int(c[1]) == 5, c
    p = torch.softmax(lg[1, 0], -1)
    above = float(p[p > p[second[1, 0]]].sum())
    m = load(f"topp:{above * 0.999}")
    c, _, _ = m.accept_greedy(candidates=cands, target_logits=logits, verify_num_draft_tokens=stride)
    assert int(c[1]) == 0, c
    m = load(f"topp:{min(0.999, above * 1.001 + 1e-6)}")
    c, _, _ = m.accept_greedy(candidates=cands, target_logits=logits, verify_num_draft_tokens=stride)
    assert int(c[1]) >= 1, c
    n += 3
    # 4) -inf draft logit is never accepted; refusal with cert head on; bad specs refused
    lg2 = logits.clone().view(bs, stride, V)
    lg2[0, 0, cands[0, 1]] = -math.inf
    lg2[0, 0, (cands[0, 1] + 1) % V] = 50.0
    m = load("ratio:0.000001")
    c, _, _ = m.accept_greedy(candidates=cands, target_logits=lg2.view(bs * stride, V),
                              verify_num_draft_tokens=stride)
    assert int(c[0]) == 0, c
    for spec, cert in (("ratio:0.5", "1"), ("ratio:0", "0"), ("topp:1.0", "0"), ("foo:1", "0")):
        try:
            load(spec, cert)
        except (RuntimeError, ValueError):
            n += 1
            continue
        raise AssertionError(f"accepted {spec} cert={cert}")
    # 5) disabled = no patch
    os.environ["DSV41_RELAXED_ACCEPT"] = ""
    ra = importlib.reload(importlib.import_module("relaxed_accept"))
    m = fake_accept_mod()
    ra.install_accept(m)
    assert m.accept_greedy is None
    n += 2
    print(f"OK {n} checks")


if __name__ == "__main__":
    main()
