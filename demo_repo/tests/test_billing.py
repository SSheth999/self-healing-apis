"""Tests for demo_repo/billing.py.

Per AGENTS.md Section 8, each test class below has:
  - one FAIL_TO_PASS test: fails against the current old-style call site,
    and is expected to pass once the Coder patches billing.py to the new
    Stripe API shape.
  - one PASS_TO_PASS test: already passes, and must keep passing after the
    patch (guards against the Coder incidentally breaking unrelated
    behavior in the same function).
"""

from unittest.mock import MagicMock, patch

import billing


class TestChargeCustomer:
    """Covers the field_renamed drift: source -> payment_method on /v1/charges."""

    @patch("stripe.Charge.create")
    def test_charge_uses_payment_method_not_source(self, mock_create: MagicMock) -> None:
        """FAIL_TO_PASS."""
        mock_create.return_value = MagicMock(id="ch_123")

        billing.charge_customer(amount=1000, currency="usd", source_token="tok_visa", customer_id="cus_1")

        _, kwargs = mock_create.call_args
        assert "payment_method" in kwargs, "expected payment_method kwarg, Stripe renamed `source`"
        assert "source" not in kwargs, "the old `source` kwarg must not be sent anymore"
        assert kwargs["payment_method"] == "tok_visa"

    @patch("stripe.Charge.create")
    def test_charge_still_passes_amount_currency_and_customer(self, mock_create: MagicMock) -> None:
        """PASS_TO_PASS."""
        mock_create.return_value = MagicMock(id="ch_123")

        billing.charge_customer(amount=2500, currency="eur", source_token="tok_amex", customer_id="cus_2")

        _, kwargs = mock_create.call_args
        assert kwargs["amount"] == 2500
        assert kwargs["currency"] == "eur"
        assert kwargs["customer"] == "cus_2"


class TestRefundCharge:
    """Covers the field_required_changed drift: reason now required on /v1/refunds."""

    @patch("stripe.Refund.create")
    def test_refund_includes_required_reason(self, mock_create: MagicMock) -> None:
        """FAIL_TO_PASS."""
        mock_create.return_value = MagicMock(id="re_123")

        billing.refund_charge(charge_id="ch_123", amount=500)

        _, kwargs = mock_create.call_args
        assert "reason" in kwargs, "reason is now a required parameter on /v1/refunds"
        assert kwargs["reason"], "reason must be a non-empty value, not just present"

    @patch("stripe.Refund.create")
    def test_refund_still_passes_charge_and_amount(self, mock_create: MagicMock) -> None:
        """PASS_TO_PASS."""
        mock_create.return_value = MagicMock(id="re_123")

        billing.refund_charge(charge_id="ch_999", amount=750)

        _, kwargs = mock_create.call_args
        assert kwargs["charge"] == "ch_999"
        assert kwargs["amount"] == 750


class TestAttachCustomerSource:
    """Covers the endpoint_moved drift: Sources API -> PaymentMethod.attach."""

    @patch("stripe.PaymentMethod.attach")
    @patch("stripe.Customer.create_source")
    def test_uses_payment_method_attach_not_create_source(
        self, mock_create_source: MagicMock, mock_attach: MagicMock
    ) -> None:
        """FAIL_TO_PASS."""
        mock_attach.return_value = MagicMock(id="pm_123")

        billing.attach_customer_source(customer_id="cus_1", source_token="pm_card_visa")

        mock_create_source.assert_not_called()
        mock_attach.assert_called_once()
        _, kwargs = mock_attach.call_args
        assert kwargs.get("customer") == "cus_1"

    @patch("stripe.PaymentMethod.attach")
    @patch("stripe.Customer.create_source")
    def test_attach_returns_the_stripe_object(
        self, mock_create_source: MagicMock, mock_attach: MagicMock
    ) -> None:
        """PASS_TO_PASS: whichever underlying call is used, the Stripe object
        it returns must be passed back to the caller unchanged. Both mocks
        are wired to the same sentinel so this holds true both before and
        after the patch, regardless of which one is actually invoked."""
        sentinel = MagicMock(id="pm_789")
        mock_create_source.return_value = sentinel
        mock_attach.return_value = sentinel

        result = billing.attach_customer_source(customer_id="cus_2", source_token="pm_card_mastercard")

        assert result is sentinel
