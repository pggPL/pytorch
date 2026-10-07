# Owner(s): ["module: cpp"]

from pathlib import Path

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    parametrize,
    run_tests,
    TestCase,
)


class TestStableScalar(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            import libtorch_agn_scalar
        except ImportError:
            install_cpp_extension(
                Path(__file__).parent / "libtorch_agn_scalar_extension"
            )
            import libtorch_agn_scalar  # noqa: F401

    @dtypes(torch.int64, torch.float64, torch.complex128)
    def test_add(self, device, dtype):
        alpha = {torch.int64: 2**53 + 1, torch.float64: 0.5, torch.complex128: 1 + 2j}[
            dtype
        ]
        a = torch.ones(4, dtype=dtype, device=device)
        b = torch.ones_like(a)
        actual = torch.ops.stable_scalar_extension.add(a, b, alpha)
        self.assertEqual(actual, torch.add(a, b, alpha=alpha))
        self.assertEqual(torch.ops.stable_scalar_extension.add(a, b), a + b)

    @parametrize("bounds", [(0, 0, 1), (5, -1, -2), (2**53 + 1, 2**53 + 5, 1)])
    def test_arange(self, device, bounds):
        actual = torch.ops.stable_scalar_extension.arange(*bounds, torch.int64, device)
        self.assertEqual(
            actual, torch.arange(*bounds, dtype=torch.int64, device=device)
        )
        self.assertEqual(actual.dtype, torch.int64)

    def test_roundtrip(self, device):
        values = [True, False, 2**53 + 1, -(2**63), 1.25, 1 + 2j]
        for value in values:
            result = torch.ops.stable_scalar_extension.echo(value)
            self.assertIs(type(result), type(value))
            self.assertEqual(result, value)
        self.assertEqual(torch.ops.stable_scalar_extension.echo_list(values), values)

    def test_errors(self, device):
        with self.assertRaisesRegex(RuntimeError, "step must be nonzero"):
            torch.ops.stable_scalar_extension.arange(0, 5, 0, torch.int64, device)
        a = torch.ones(2, dtype=torch.int64, device=device)
        with self.assertRaisesRegex(RuntimeError, "alpha"):
            torch.ops.stable_scalar_extension.add(a, a, 0.5)


instantiate_device_type_tests(TestStableScalar, globals())


if __name__ == "__main__":
    run_tests()
