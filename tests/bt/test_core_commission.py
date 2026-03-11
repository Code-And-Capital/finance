from __future__ import annotations

import pytest

from bt.core.commission import (
    fixed_per_trade_commission,
    notional_bps_commission,
    notional_bps_with_min_commission,
    per_share_commission,
    per_share_with_min_max_commission,
    quantity_tiered_commission,
    sec_finra_sell_fee,
    tiered_notional_bps_commission,
    zero_commission,
)


def test_zero_commission_always_returns_zero():
    assert zero_commission(100, 25.0) == 0.0
    assert zero_commission(-50, 99.0) == 0.0


def test_quantity_tiered_commission_applies_floor_and_quantity_rate():
    assert quantity_tiered_commission(10, 100.0) == 100.0
    assert quantity_tiered_commission(-10, 100.0) == 100.0
    assert quantity_tiered_commission(100_000, 1.0) == pytest.approx(210.0)


def test_notional_bps_commission_is_25bps_of_absolute_notional():
    assert notional_bps_commission(100, 10.0) == pytest.approx(2.5)
    assert notional_bps_commission(-100, 10.0) == pytest.approx(2.5)
    assert notional_bps_commission(0, 10.0) == 0.0


def test_fixed_per_trade_commission_charged_only_for_non_zero_trades():
    assert fixed_per_trade_commission(0, 10.0, fixed_fee=2.0) == 0.0
    assert fixed_per_trade_commission(10, 10.0, fixed_fee=2.0) == 2.0
    assert fixed_per_trade_commission(-10, 10.0, fixed_fee=2.0) == 2.0


def test_per_share_commission():
    assert per_share_commission(100, 10.0, per_share_fee=0.01) == pytest.approx(1.0)
    assert per_share_commission(-50, 10.0, per_share_fee=0.01) == pytest.approx(0.5)
    assert per_share_commission(0, 10.0, per_share_fee=0.01) == 0.0


def test_per_share_with_min_max_commission_applies_bounds():
    assert per_share_with_min_max_commission(
        10, 10.0, per_share_fee=0.01, min_fee=1.0, max_fee=3.0
    ) == pytest.approx(1.0)
    assert per_share_with_min_max_commission(
        500, 10.0, per_share_fee=0.01, min_fee=1.0, max_fee=3.0
    ) == pytest.approx(3.0)
    assert per_share_with_min_max_commission(
        100, 10.0, per_share_fee=0.01, min_fee=1.0, max_fee=None
    ) == pytest.approx(1.0)


def test_notional_bps_with_min_commission_applies_floor():
    assert notional_bps_with_min_commission(0, 10.0, bps=10.0, min_fee=2.0) == 0.0
    assert notional_bps_with_min_commission(10, 10.0, bps=10.0, min_fee=2.0) == 2.0
    assert notional_bps_with_min_commission(
        10_000, 10.0, bps=10.0, min_fee=2.0
    ) == pytest.approx(100.0)


def test_tiered_notional_bps_commission_selects_correct_tier():
    tiers = ((1_000.0, 20.0), (10_000.0, 10.0), (float("inf"), 5.0))
    assert tiered_notional_bps_commission(10, 10.0, tiers=tiers) == pytest.approx(0.2)
    assert tiered_notional_bps_commission(200, 10.0, tiers=tiers) == pytest.approx(2.0)
    assert tiered_notional_bps_commission(2_000, 10.0, tiers=tiers) == pytest.approx(
        10.0
    )


def test_sec_finra_sell_fee_applies_only_to_sells_and_caps_finra_component():
    assert sec_finra_sell_fee(100, 10.0, sec_bps=1.0, finra_per_share_fee=0.01) == 0.0
    assert sec_finra_sell_fee(0, 10.0, sec_bps=1.0, finra_per_share_fee=0.01) == 0.0

    fee = sec_finra_sell_fee(
        -100,
        10.0,
        sec_bps=1.0,
        finra_per_share_fee=0.02,
        finra_max_fee=1.5,
    )
    # SEC: 100*10 * 1/10000 = 0.1; FINRA: min(100*0.02,1.5)=1.5
    assert fee == pytest.approx(1.6)
