import sys

import torch


sys.path.insert(0, "/home/DIFFER/github_latest_test2_lXaBlM/scripts")
import ndp_v5_prcc_runner as ndp  # noqa: E402


def test_shn_uses_multiple_positives_and_negative_identities_with_live_gradient():
    torch.manual_seed(1234)
    representation = torch.nn.functional.normalize(torch.randn(16, 12, requires_grad=True), dim=1)
    pids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    selected = ndp.select_ndp_shn_losses(representation, pids)

    assert len(selected) == ndp.NDP_SHN_MAX_ANCHORS_PER_BATCH
    assert all(loss.requires_grad for loss, _margin, negative_count, positive_count in selected)
    assert all(negative_count == ndp.NDP_SHN_TOPK for _loss, _margin, negative_count, _positive_count in selected)
    assert all(positive_count == 3 for _loss, _margin, _negative_count, positive_count in selected)
    total = torch.stack([item[0] for item in selected]).mean()
    gradient = torch.autograd.grad(total, representation)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_shn_mp_balances_all_positive_terms_with_live_gradient():
    torch.manual_seed(1234)
    representation = torch.nn.functional.normalize(torch.randn(16, 12, requires_grad=True), dim=1)
    pids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    selected = ndp.select_ndp_shn_mp_losses(representation, pids)

    assert len(selected) == ndp.NDP_SHN_MAX_ANCHORS_PER_BATCH
    assert all(loss.requires_grad for loss, _margin, negative_count, _positive_count in selected)
    assert all(negative_count == ndp.NDP_SHN_TOPK for _loss, _margin, negative_count, _positive_count in selected)
    assert all(positive_count == 3 for _loss, _margin, _negative_count, positive_count in selected)
    total = torch.stack([item[0] for item in selected]).mean()
    gradient = torch.autograd.grad(total, representation)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_shn_bank_adds_cross_batch_positives_and_keeps_live_gradient():
    torch.manual_seed(1234)
    representation = torch.nn.functional.normalize(torch.randn(16, 12, requires_grad=True), dim=1)
    pids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    bank_features = torch.nn.functional.normalize(torch.randn(32, 12), dim=1)
    bank_pids = torch.tensor([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8)
    bank_indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27])
    bank_clothes = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 4)
    batch_clothes = torch.tensor([0, 0, 1, 1] * 4)

    selected = ndp.select_ndp_shn_bank_losses(
        representation,
        pids,
        bank_features,
        bank_pids,
        bank_indices,
        bank_clothes=bank_clothes,
        batch_clothes=batch_clothes,
    )

    assert len(selected) == ndp.NDP_SHN_MAX_ANCHORS_PER_BATCH
    assert all(loss.requires_grad for loss, _margin, negative_count, _positive_count in selected)
    assert all(negative_count == ndp.NDP_SHN_TOPK for _loss, _margin, negative_count, _positive_count in selected)
    assert all(positive_count > 3 for _loss, _margin, _negative_count, positive_count in selected)
    total = torch.stack([item[0] for item in selected]).mean()
    gradient = torch.autograd.grad(total, representation)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_shn_is_a_supported_loss_arm_and_b0_contract_remains_exact():
    main = torch.tensor(2.0, requires_grad=True)
    ndp_loss = torch.tensor(3.0, requires_grad=True)
    assert torch.equal(ndp.compose_total_loss(main, ndp_loss, "B0"), main)
    assert torch.isclose(ndp.compose_total_loss(main, ndp_loss, "NDP_SHN"), torch.tensor(2.003))
    assert torch.isclose(ndp.compose_total_loss(main, ndp_loss, "NDP_SHN_MP"), torch.tensor(2.003))
    assert torch.isclose(ndp.compose_total_loss(main, ndp_loss, "NDP_SHN_BANK"), torch.tensor(2.003))
    assert torch.isclose(ndp.compose_total_loss(main, ndp_loss, "NDP_SHN", 0.001), torch.tensor(2.003))
