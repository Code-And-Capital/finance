"""Commission function library for execution cost modeling.

Each function follows the common signature:

``commission(quantity, price) -> fee``

where:
- ``quantity`` is signed (positive buy, negative sell)
- ``price`` is per-unit execution price
- returned ``fee`` is a non-negative cash amount
"""

from __future__ import annotations


def zero_commission(quantity: float, price: float) -> float:
    """Return zero transaction fee.

    This is the baseline commission model used when transaction costs are
    intentionally disabled.

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price.

    Returns
    -------
    float
        Always ``0.0``.
    """
    _ = quantity, price
    return 0.0


def quantity_tiered_commission(quantity: float, price: float) -> float:
    """Return commission proportional to quantity with a hard floor.

    Fee model:
    ``max(100.0, abs(quantity) * 0.0021)``

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price. Included for interface compatibility and not
        used by this function.

    Returns
    -------
    float
        Transaction fee in cash terms.
    """
    _ = price
    return max(100.0, abs(float(quantity)) * 0.0021)


def notional_bps_commission(quantity: float, price: float) -> float:
    """Return a basis-point fee on absolute traded notional.

    Fee model:
    ``abs(quantity) * price * 25 / 10000``

    This corresponds to 25 bps of traded notional.

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price.

    Returns
    -------
    float
        Transaction fee in cash terms.
    """
    notional = abs(float(quantity)) * float(price)
    return notional * 25.0 / 10000.0


def fixed_per_trade_commission(
    quantity: float,
    price: float,
    fixed_fee: float = 1.0,
) -> float:
    """Return a fixed fee per non-zero trade.

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price. Included for interface compatibility and not
        used by this function.
    fixed_fee : float, optional
        Flat fee charged for every non-zero trade.

    Returns
    -------
    float
        ``fixed_fee`` for non-zero quantity, else ``0.0``.
    """
    _ = price
    q = float(quantity)
    if q == 0.0:
        return 0.0
    return float(fixed_fee)


def per_share_commission(
    quantity: float,
    price: float,
    per_share_fee: float = 0.005,
) -> float:
    """Return linear per-share commission.

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price. Included for interface compatibility and not
        used by this function.
    per_share_fee : float, optional
        Fee charged per traded unit.

    Returns
    -------
    float
        ``abs(quantity) * per_share_fee``.
    """
    _ = price
    return abs(float(quantity)) * float(per_share_fee)


def per_share_with_min_max_commission(
    quantity: float,
    price: float,
    per_share_fee: float = 0.005,
    min_fee: float = 1.0,
    max_fee: float | None = None,
) -> float:
    """Return per-share commission with optional floor and cap.

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price. Included for interface compatibility and not
        used by this function.
    per_share_fee : float, optional
        Fee charged per traded unit.
    min_fee : float, optional
        Minimum fee for non-zero trades.
    max_fee : float | None, optional
        Maximum fee cap. If ``None``, no cap is applied.

    Returns
    -------
    float
        Bounded per-share fee.
    """
    _ = price
    q = abs(float(quantity))
    if q == 0.0:
        return 0.0
    fee = q * float(per_share_fee)
    fee = max(float(min_fee), fee)
    if max_fee is not None:
        fee = min(float(max_fee), fee)
    return fee


def notional_bps_with_min_commission(
    quantity: float,
    price: float,
    bps: float = 25.0,
    min_fee: float = 0.0,
) -> float:
    """Return basis-point commission on notional with a fee floor.

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price.
    bps : float, optional
        Basis-point charge applied to absolute notional.
    min_fee : float, optional
        Minimum fee for non-zero trades.

    Returns
    -------
    float
        ``max(min_fee, abs(quantity * price) * bps / 10000)`` for non-zero
        trades, else ``0.0``.
    """
    notional = abs(float(quantity)) * float(price)
    if notional == 0.0:
        return 0.0
    fee = notional * float(bps) / 10000.0
    return max(float(min_fee), fee)


def tiered_notional_bps_commission(
    quantity: float,
    price: float,
    tiers: tuple[tuple[float, float], ...] = (
        (100_000.0, 10.0),
        (1_000_000.0, 5.0),
        (float("inf"), 2.5),
    ),
) -> float:
    """Return notional bps commission selected from notional tiers.

    Parameters
    ----------
    quantity : float
        Signed trade size in units.
    price : float
        Per-unit execution price.
    tiers : tuple[tuple[float, float], ...], optional
        Ordered ``(notional_upper_bound, bps)`` schedule. The first tier whose
        bound is >= notional is applied.

    Returns
    -------
    float
        Tier-selected bps fee on absolute notional.
    """
    notional = abs(float(quantity)) * float(price)
    if notional == 0.0:
        return 0.0
    for upper_bound, bps in tiers:
        if notional <= float(upper_bound):
            return notional * float(bps) / 10000.0
    upper_bound, bps = tiers[-1]
    _ = upper_bound
    return notional * float(bps) / 10000.0


def sec_finra_sell_fee(
    quantity: float,
    price: float,
    sec_bps: float = 0.0,
    finra_per_share_fee: float = 0.0,
    finra_max_fee: float | None = None,
) -> float:
    """Return US-style regulatory fee add-on for sell orders only.

    This function models two common sell-side components:
    - SEC fee on notional (basis points)
    - FINRA trading activity fee on shares (with optional cap)

    Parameters
    ----------
    quantity : float
        Signed trade size in units. Only ``quantity < 0`` incurs fees.
    price : float
        Per-unit execution price.
    sec_bps : float, optional
        SEC-style notional charge in basis points for sell orders.
    finra_per_share_fee : float, optional
        FINRA-style per-share charge for sell orders.
    finra_max_fee : float | None, optional
        Optional cap for FINRA component. If ``None``, no cap is applied.

    Returns
    -------
    float
        Regulatory fee for sells, else ``0.0`` for buys/zero quantity.
    """
    q = float(quantity)
    if q >= 0.0:
        return 0.0
    shares = abs(q)
    notional = shares * float(price)

    sec_fee = notional * float(sec_bps) / 10000.0
    finra_fee = shares * float(finra_per_share_fee)
    if finra_max_fee is not None:
        finra_fee = min(finra_fee, float(finra_max_fee))
    return sec_fee + finra_fee
