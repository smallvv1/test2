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


def test_shn_reliable_bank_is_sparse_auxiliary_and_keeps_live_negatives():
    torch.manual_seed(1234)
    representation = torch.nn.functional.normalize(torch.randn(16, 12, requires_grad=True), dim=1)
    pids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    extra = torch.nn.functional.normalize(representation.detach() + 0.01 * torch.randn_like(representation), dim=1)
    bank_features = torch.cat([representation.detach(), extra], dim=0)
    bank_pids = torch.cat([pids, pids], dim=0)
    bank_indices = torch.arange(16)
    bank_clothes = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    batch_clothes = torch.zeros(16, dtype=torch.long)

    selected = ndp.select_ndp_shn_reliable_bank_aux_losses(
        representation,
        pids,
        bank_features,
        bank_pids,
        bank_indices,
        bank_clothes=bank_clothes,
        batch_clothes=batch_clothes,
    )

    assert 0 < len(selected) <= ndp.NDP_SHN_RELIABLE_BANK_MAX_ANCHORS
    assert all(loss.requires_grad for loss, _margin, negative_count, _positive_count, _reliability in selected)
    assert all(negative_count == ndp.NDP_SHN_TOPK for _loss, _margin, negative_count, _positive_count, _reliability in selected)
    assert all(positive_count == 1 for _loss, _margin, _negative_count, positive_count, _reliability in selected)
    total = torch.stack([item[0] for item in selected]).mean()
    gradient = torch.autograd.grad(total, representation)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_shn_reliable_risk_selector_changes_simple_positive_preference():
    torch.manual_seed(1234)
    representation = torch.nn.functional.normalize(torch.randn(16, 12, requires_grad=True), dim=1)
    pids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    extra = torch.nn.functional.normalize(representation.detach() + 0.01 * torch.randn_like(representation), dim=1)
    bank_features = torch.cat([representation.detach(), extra], dim=0)
    bank_pids = torch.cat([pids, pids], dim=0)
    bank_indices = torch.arange(16)
    bank_clothes = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    batch_clothes = torch.zeros(16, dtype=torch.long)

    legacy = ndp.select_ndp_shn_reliable_bank_aux_losses(
        representation, pids, bank_features, bank_pids, bank_indices,
        bank_clothes=bank_clothes, batch_clothes=batch_clothes, risk_aware=False,
    )
    risk_aware = ndp.select_ndp_shn_reliable_bank_aux_losses(
        representation, pids, bank_features, bank_pids, bank_indices,
        bank_clothes=bank_clothes, batch_clothes=batch_clothes, risk_aware=True,
    )

    assert len(legacy) == len(risk_aware) == ndp.NDP_SHN_RELIABLE_BANK_MAX_ANCHORS
    assert [round(item[1], 5) for item in legacy] != [round(item[1], 5) for item in risk_aware]
    assert all(item[0].requires_grad for item in risk_aware)
    total = torch.stack([item[0] for item in risk_aware]).mean()
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
    assert torch.isclose(ndp.compose_total_loss(main, ndp_loss, "NDP_SHN_RELIABLE"), torch.tensor(2.003))
    assert torch.isclose(ndp.compose_total_loss(main, ndp_loss, "NDP_SHN_RELIABLE_RISK"), torch.tensor(2.003))
    assert torch.isclose(ndp.compose_total_loss(main, ndp_loss, "NDP_SHN", 0.001), torch.tensor(2.003))
