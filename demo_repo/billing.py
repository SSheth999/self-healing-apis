"""Demo billing integration against the Stripe SDK.

This file intentionally uses an OLD version of the Stripe API shape. It is
the fixture target repo the self-healing pipeline patches - see
watcher/fixtures/stripe_v_old.json and stripe_v_new.json for the matching
spec-level drift, and tests/test_billing.py for the FAIL_TO_PASS /
PASS_TO_PASS tests that define what a correct patch looks like.

Do not "fix" this file by hand - it is meant to be patched by the Coder
agent. See AGENTS.md Section 8.
"""

from __future__ import annotations

import stripe


def charge_customer(
    amount: int,
    currency: str,
    source_token: str,
    customer_id: str | None = None,
) -> stripe.Charge:
    """Create a charge using the legacy `source` parameter.

    Breaking change: `source` was renamed to `payment_method` on
    POST /v1/charges (see DriftItem change_type="field_renamed").
    """

    return stripe.Charge.create(
        amount=amount,
        currency=currency,
        source=source_token,
        customer=customer_id,
    )


def refund_charge(charge_id: str, amount: int | None = None) -> stripe.Refund:
    """Refund a charge without specifying a reason.

    Breaking change: `reason` became a required parameter on
    POST /v1/refunds (see DriftItem change_type="field_required_changed").
    """

    return stripe.Refund.create(charge=charge_id, amount=amount)


def attach_customer_source(customer_id: str, source_token: str) -> stripe.Source:
    """Attach a payment source to a customer via the legacy Sources API.

    Breaking change: POST /v1/customers/{customer}/sources was moved/
    replaced by POST /v1/payment_methods/{payment_method}/attach (see
    DriftItem change_type="endpoint_moved").
    """

    return stripe.Customer.create_source(customer_id, source=source_token)
