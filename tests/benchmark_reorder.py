"""Benchmark comparing Python vs Rust Reorder implementation performance."""

import timeit
import string
from pathlib import Path

# ============ Python Implementation (Original) ============

EOS = "<EOS>"
TN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}


class Token:
    def __init__(self, name):
        self.name = name
        self.order = []
        self.members = {}

    def append(self, key, value):
        self.order.append(key)
        self.members[key] = value

    def string(self, orders):
        output = self.name + " {"
        if self.name in orders.keys():
            if (
                "preserve_order" not in self.members.keys()
                or self.members["preserve_order"] != "true"
            ):
                self.order = orders[self.name]

        for key in self.order:
            if key not in self.members.keys():
                continue
            output += ' {}: "{}"'.format(key, self.members[key])
        return output + " }"


class TokenParser:
    def __init__(self):
        self.orders = TN_ORDERS

    def load(self, input):
        assert len(input) > 0
        self.index = 0
        self.text = input
        self.char = input[0]
        self.tokens = []

    def read(self):
        if self.index < len(self.text) - 1:
            self.index += 1
            self.char = self.text[self.index]
            return True
        self.char = EOS
        return False

    def parse_ws(self):
        not_eos = self.char != EOS
        while not_eos and self.char == " ":
            not_eos = self.read()
        return not_eos

    def parse_char(self, exp):
        if self.char == exp:
            self.read()
            return True
        return False

    def parse_chars(self, exp):
        ok = False
        for x in exp:
            ok |= self.parse_char(x)
        return ok

    def parse_key(self):
        assert self.char != EOS
        assert self.char not in string.whitespace

        key = ""
        while self.char in string.ascii_letters + "_":
            key += self.char
            self.read()
        return key

    def parse_value(self):
        assert self.char != EOS
        escape = False

        value = ""
        while self.char != '"':
            value += self.char
            escape = self.char == "\\"
            self.read()
            if escape:
                escape = False
                value += self.char
                self.read()
        return value

    def parse(self, input):
        self.load(input)
        while self.parse_ws():
            name = self.parse_key()
            self.parse_chars(" { ")

            token = Token(name)
            while self.parse_ws():
                if self.char == "}":
                    self.parse_char("}")
                    break
                key = self.parse_key()
                self.parse_chars(': "')
                value = self.parse_value()
                self.parse_char('"')
                token.append(key, value)
            self.tokens.append(token)

    def reorder(self, input):
        self.parse(input)
        output = ""
        for token in self.tokens:
            output += token.string(self.orders) + " "
        return output.strip()


# ============ Benchmark Tests ============

def benchmark():
    """Run performance benchmark comparing Python vs Rust implementation."""

    # Test data - various complexity levels
    test_cases = {
        "simple_single": 'money { currency: "USD" value: "100" }',
        "simple_double": 'money { currency: "USD" value: "100" } date { day: "15" month: "3" year: "2024" }',
        "complex_multi": (
            'date { day: "1" month: "1" year: "2024" } '
            'money { currency: "EUR" value: "99.99" } '
            'time { hour: "14" minute: "30" second: "45" noon: "pm" } '
            'fraction { numerator: "3" denominator: "4" } '
            'measure { value: "5.5" numerator: "kg" denominator: "1" }'
        ),
        "long_values": (
            'money { currency: "United States Dollar" value: "1000000.99" } '
            'date { day: "Twenty-fifth" month: "December" year: "Two Thousand Twenty-Four" }'
        ),
    }

    # Number of iterations for each test
    iterations = 10000

    print("=" * 70)
    print("Performance Benchmark: Python vs Rust Reorder Implementation")
    print("=" * 70)
    print(f"Iterations per test: {iterations:,}")
    print()

    # Initialize implementations
    python_impl = TokenParser()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from osc_data._core import reorder as core_reorder
    rust_impl = core_reorder.Reorder()

    results = []

    for test_name, test_input in test_cases.items():
        print(f"\nTest: {test_name}")
        print(f"Input length: {len(test_input)} characters")
        print("-" * 70)

        # Warm up
        for _ in range(100):
            python_impl.reorder(test_input)
            rust_impl.reorder(test_input)

        # Benchmark Python implementation
        python_time = timeit.timeit(
            lambda: python_impl.reorder(test_input),
            number=iterations
        )

        # Benchmark Rust implementation
        rust_time = timeit.timeit(
            lambda: rust_impl.reorder(test_input),
            number=iterations
        )

        # Calculate speedup
        speedup = python_time / rust_time if rust_time > 0 else float('inf')

        # Store results
        results.append({
            'test': test_name,
            'python_time': python_time,
            'rust_time': rust_time,
            'speedup': speedup
        })

        # Print results
        print(f"  Python: {python_time:.4f}s ({python_time/iterations*1e6:.2f} µs/op)")
        print(f"  Rust:   {rust_time:.4f}s ({rust_time/iterations*1e6:.2f} µs/op)")
        print(f"  Speedup: {speedup:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    total_python = sum(r['python_time'] for r in results)
    total_rust = sum(r['rust_time'] for r in results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)

    print(f"Total Python time: {total_python:.4f}s")
    print(f"Total Rust time:   {total_rust:.4f}s")
    print(f"Overall speedup:   {total_python/total_rust:.2f}x")
    print(f"Average speedup:   {avg_speedup:.2f}x")

    # Verify correctness
    print("\n" + "=" * 70)
    print("Correctness Verification")
    print("=" * 70)

    for test_name, test_input in test_cases.items():
        python_result = python_impl.reorder(test_input)
        rust_result = rust_impl.reorder(test_input)

        match = python_result == rust_result
        status = "PASS" if match else "FAIL"
        print(f"  {test_name}: {status}")

        if not match:
            print(f"    Python: {python_result}")
            print(f"    Rust:   {rust_result}")

    return results


if __name__ == "__main__":
    benchmark()
