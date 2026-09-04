"""Whether the pruning surgery is correct, rather than merely plausible.

The mechanism narrows an attention mask, a rotary embedding and a cache position
along axes that are easy to confuse, and every way of getting it wrong produces
output that still looks like language. The decisive check is therefore an
identity: keep *every* visual token and the pruned path must reproduce the
unpruned generation exactly. A mask sliced along the query axis instead of the
key axis, or positions renumbered instead of preserved, fails it immediately.

The identity test needs a real checkpoint and a GPU. A CUDA device alone is not
enough of a signal to run it: on a laptop with a GPU but no warm cache it would
start a 14 GB download from inside a unit test. It therefore requires an
explicit opt-in, ``XMODAL_GPU_TESTS=1``, which the cluster job sets and nothing
else does. The slicing tests below run anywhere and cover the index arithmetic
on its own.
"""

from __future__ import annotations

import os

import pytest
import torch

from xmodal.prune import _slice_kwargs


def test_prefill_slices_both_mask_axes():
    """During prefill queries and keys are the same positions, so both move."""
    L = 8
    keep = torch.tensor([0, 2, 5, 7])
    am = torch.arange(L * L, dtype=torch.float32).view(1, 1, L, L)
    cos = torch.arange(L, dtype=torch.float32).view(1, L, 1).repeat(1, 1, 4)
    kw = {"attention_mask": am, "position_embeddings": (cos, cos.clone()),
          "cache_position": torch.arange(L),
          "position_ids": torch.arange(L).view(1, L)}

    out = _slice_kwargs(kw, keep, q_len=L, prompt_len=L)
    assert out["attention_mask"].shape == (1, 1, 4, 4)
    assert torch.equal(out["attention_mask"][0, 0], am[0, 0][keep][:, keep])
    assert torch.equal(out["cache_position"], keep)
    assert torch.equal(out["position_ids"][0], keep)
    # Positions are preserved, not renumbered: the kept tokens keep the rotary
    # phases they had in the full sequence.
    assert torch.equal(out["position_embeddings"][0][0, :, 0], keep.float())


def test_decode_slices_only_the_key_axis_and_keeps_the_tail():
    """One new query, keys are the shortened cache plus everything generated since."""
    prompt_len, total = 8, 11        # three tokens generated so far
    keep = torch.tensor([0, 2, 5, 7])
    am = torch.arange(total, dtype=torch.float32).view(1, 1, 1, total)
    out = _slice_kwargs({"attention_mask": am}, keep, q_len=1, prompt_len=prompt_len)
    assert out["attention_mask"].shape == (1, 1, 1, 7)
    assert out["attention_mask"][0, 0, 0].tolist() == [0.0, 2.0, 5.0, 7.0, 8.0, 9.0, 10.0]


def test_decode_leaves_rotary_and_cache_position_alone():
    """The new token is a single position; narrowing it would delete it."""
    keep = torch.tensor([0, 2])
    cos = torch.zeros(1, 1, 4)
    kw = {"position_embeddings": (cos, cos.clone()), "cache_position": torch.tensor([9])}
    out = _slice_kwargs(kw, keep, q_len=1, prompt_len=8)
    assert out["position_embeddings"][0].shape == (1, 1, 4)
    assert out["cache_position"].tolist() == [9]


def test_absent_kwargs_are_not_invented():
    out = _slice_kwargs({}, torch.tensor([0, 1]), q_len=4, prompt_len=4)
    assert out == {}


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("XMODAL_GPU_TESTS") != "1",
                    reason="set XMODAL_GPU_TESTS=1 (needs a GPU and a warm cache)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_full_budget_reproduces_unpruned_generation():
    """The identity: keeping every visual token must change nothing at all.

    This is the test that licenses every accuracy number in the project. If the
    surgery perturbs the sequence even slightly, a criterion that looks good is
    indistinguishable from a bug that happens to help.
    """
    from PIL import Image

    from xmodal.calib import Sample
    from xmodal.models import describe, load_vlm
    from xmodal.pipeline import generate_pruned

    name = os.environ.get("XMODAL_TEST_MODEL", "llava-hf/llava-1.5-7b-hf")
    model, processor = load_vlm(name, dtype=torch.float16, device_map={"": 0},
                                attn_implementation="sdpa")
    info = describe(model, name)
    sample = Sample(
        image=Image.new("RGB", (336, 336), (90, 140, 200)),
        question="Describe the colour of this image.",
        answer="blue",
        meta={"task": "pope", "gold": "blue"},
    )

    from xmodal.prune import PrunableModel

    with PrunableModel(model, layer=15) as pm:
        base, _ = generate_pruned(pm, processor, sample, info.image_token_id,
                                  "cuda", criterion="none", budget=0,
                                  max_new_tokens=24)
        # Budget equal to the visual token count: every token is kept, so the
        # only difference from the reference is the machinery itself.
        from xmodal.calib import make_batch
        from xmodal.pipeline import selection_inputs

        batch = make_batch(sample, processor, info.image_token_id, "cuda",
                           with_answer=False)
        n_visual = int(selection_inputs(pm, batch).visual_pos.numel())
        for criterion in ("leverage", "pivot", "attention", "residual", "greedy",
                          "random"):
            got, info_d = generate_pruned(pm, processor, sample, info.image_token_id,
                                          "cuda", criterion=criterion,
                                          budget=n_visual, max_new_tokens=24)
            assert info_d["kept_len"] == info_d["prompt_len"], criterion
            assert got == base, f"{criterion}: {got!r} != {base!r}"
