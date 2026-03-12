"""Tests for the Reorder class implemented in Rust."""

import pytest
from osc_data._core import reorder as core_reorder


class TestReorder:
    """Test cases for the Reorder class."""

    def test_reorder_money(self):
        """Test reordering money tokens - value should come before currency."""
        reorder = core_reorder.Reorder()

        # Test basic money reordering
        input_text = 'money { currency: "USD" value: "100" }'
        expected = 'money { value: "100" currency: "USD" }'
        result = reorder.reorder(input_text)
        assert result == expected

        # Test with different values
        input_text2 = 'money { currency: "CNY" value: "50.5" }'
        expected2 = 'money { value: "50.5" currency: "CNY" }'
        result2 = reorder.reorder(input_text2)
        assert result2 == expected2

    def test_reorder_date(self):
        """Test reordering date tokens - year, month, day order."""
        reorder = core_reorder.Reorder()

        input_text = 'date { day: "15" month: "3" year: "2024" }'
        expected = 'date { year: "2024" month: "3" day: "15" }'
        result = reorder.reorder(input_text)
        assert result == expected

        # Test with different format
        input_text2 = 'date { month: "12" day: "25" year: "2023" }'
        expected2 = 'date { year: "2023" month: "12" day: "25" }'
        result2 = reorder.reorder(input_text2)
        assert result2 == expected2

    def test_reorder_time(self):
        """Test reordering time tokens - noon, hour, minute, second order."""
        reorder = core_reorder.Reorder()

        input_text = 'time { minute: "30" hour: "10" }'
        expected = 'time { hour: "10" minute: "30" }'
        result = reorder.reorder(input_text)
        assert result == expected

        # Test with noon and second
        input_text2 = 'time { second: "45" hour: "3" noon: "pm" minute: "15" }'
        expected2 = 'time { noon: "pm" hour: "3" minute: "15" second: "45" }'
        result2 = reorder.reorder(input_text2)
        assert result2 == expected2

    def test_reorder_fraction(self):
        """Test reordering fraction tokens - denominator before numerator."""
        reorder = core_reorder.Reorder()

        input_text = 'fraction { numerator: "3" denominator: "4" }'
        expected = 'fraction { denominator: "4" numerator: "3" }'
        result = reorder.reorder(input_text)
        assert result == expected

    def test_reorder_measure(self):
        """Test reordering measure tokens - denominator, numerator, value order."""
        reorder = core_reorder.Reorder()

        input_text = 'measure { value: "5.5" numerator: "kg" denominator: "1" }'
        expected = 'measure { denominator: "1" numerator: "kg" value: "5.5" }'
        result = reorder.reorder(input_text)
        assert result == expected

    def test_multiple_tokens(self):
        """Test reordering multiple tokens in one input."""
        reorder = core_reorder.Reorder()

        input_text = 'money { currency: "USD" value: "100" } date { day: "15" month: "3" year: "2024" }'
        expected = 'money { value: "100" currency: "USD" } date { year: "2024" month: "3" day: "15" }'
        result = reorder.reorder(input_text)
        assert result == expected

    def test_empty_input_raises_error(self):
        """Test that empty input raises a ValueError."""
        reorder = core_reorder.Reorder()

        with pytest.raises(ValueError, match="Input cannot be empty"):
            reorder.reorder("")

    def test_unknown_token_type_not_reordered(self):
        """Test that unknown token types are not reordered."""
        reorder = core_reorder.Reorder()

        # Unknown type should keep original order
        input_text = 'custom { z: "last" a: "first" }'
        result = reorder.reorder(input_text)
        # Should keep original order since 'custom' is not in orders
        assert 'a: "first"' in result
        assert 'z: "last"' in result

    def test_get_orders(self):
        """Test getting the orders dictionary."""
        reorder = core_reorder.Reorder()

        orders = reorder.orders
        assert "money" in orders
        assert "date" in orders
        assert "time" in orders
        assert "fraction" in orders
        assert "measure" in orders

        # Check specific order
        assert orders["money"] == ["value", "currency"]
        assert orders["date"] == ["year", "month", "day"]

    def test_set_orders(self):
        """Test setting custom orders."""
        reorder = core_reorder.Reorder()

        # Set custom order
        custom_orders = {
            "custom_type": ["field_b", "field_a"]
        }
        reorder.orders = custom_orders

        # Verify custom order is used
        assert reorder.orders == custom_orders

        # Test that new order is applied
        input_text = 'custom_type { field_a: "1" field_b: "2" }'
        result = reorder.reorder(input_text)
        expected = 'custom_type { field_b: "2" field_a: "1" }'
        assert result == expected

    def test_preserve_order_flag(self):
        """Test that preserve_order flag prevents reordering."""
        reorder = core_reorder.Reorder()

        # Create input with preserve_order flag
        # Note: This test checks if the implementation respects preserve_order
        input_text = 'money { preserve_order: "true" currency: "USD" value: "100" }'
        result = reorder.reorder(input_text)

        # With preserve_order, original order should be maintained
        # currency comes before value in input
        currency_pos = result.find('currency')
        value_pos = result.find('value')
        assert currency_pos < value_pos, "preserve_order should maintain input order"

    def test_escaped_quotes_in_value(self):
        """Test handling of escaped quotes in values."""
        reorder = core_reorder.Reorder()

        # Test value with escaped quote
        input_text = r'money { currency: "US\"D" value: "100" }'
        result = reorder.reorder(input_text)
        assert 'value: "100"' in result
        assert 'currency: "US\\"D"' in result

    def test_complex_multiple_tokens(self):
        """Test complex scenarios with multiple token types."""
        reorder = core_reorder.Reorder()

        # Mix of different types
        input_text = (
            'date { day: "1" month: "1" year: "2024" } '
            'money { currency: "EUR" value: "99.99" } '
            'time { hour: "14" minute: "30" }'
        )
        result = reorder.reorder(input_text)

        # Check each type is correctly reordered
        assert 'date { year: "2024" month: "1" day: "1" }' in result
        assert 'money { value: "99.99" currency: "EUR" }' in result
        assert 'time { hour: "14" minute: "30" }' in result


class TestReorderIntegration:
    """Integration tests with TextNormalizer if available."""

    def test_reorder_from_text_module(self):
        """Test that REORDER from text module works."""
        from osc_data.text import REORDER

        # This should be the Rust implementation
        input_text = 'money { currency: "USD" value: "100" }'
        result = REORDER.reorder(input_text)
        expected = 'money { value: "100" currency: "USD" }'
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
