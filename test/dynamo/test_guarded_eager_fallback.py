# Owner(s): ["module: dynamo"]

import gc
import sys
import unittest
from unittest.mock import patch

import torch
import torch._dynamo.convert_frame
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@unittest.skipIf(sys.version_info < (3, 11), "requires exception tables")
class GuardedEagerFallbackTests(torch._dynamo.test_case.TestCase):
    @parametrize("skip_first", [True, False])
    @parametrize("dynamic", [True, False])
    def test_conditional_break_preserves_other_branches(self, skip_first, dynamic):
        def fn(x, skip, events):
            events.append("prepare")
            try:
                if skip:
                    torch._dynamo.graph_break()
                return x.sin().cos()
            finally:
                events.append("end")

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, dynamic=dynamic)
        x = torch.randn(4)
        trace_frame = torch._dynamo.convert_frame.trace_frame
        with patch("torch._dynamo.convert_frame.trace_frame", wraps=trace_frame) as trace:
            for skip in (skip_first, not skip_first):
                events = []
                self.assertEqual(compiled(x, skip, events), x.sin().cos())
                self.assertEqual(events, ["prepare", "end"])
                calls = trace.call_count
                events = []
                self.assertEqual(compiled(x, skip, events), x.sin().cos())
                self.assertEqual(events, ["prepare", "end"])
                self.assertEqual(trace.call_count, calls)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)

    def test_unsupported_instance_does_not_skip_supported_instance(self):
        class Module(torch.nn.Module):
            def __init__(self, skip):
                super().__init__()
                self.skip = skip

            def forward(self, x):
                try:
                    if self.skip:
                        torch._dynamo.graph_break()
                    return x.sin()
                finally:
                    pass

        unsupported = Module(True)
        supported = Module(False)

        def fn(x):
            return supported(unsupported(x))

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)
        x = torch.randn(4)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(counter.op_count, 1)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(counter.op_count, 1)

    @parametrize("when", ["before", "at", "after"])
    def test_exception_runs_nested_finally_with_current_locals(self, when):
        @torch.compiler.disable
        def fail():
            raise ValueError("eager failure")

        def fn(x, events):
            value = 1
            try:
                try:
                    if when == "before":
                        fail()
                    value = 2
                    torch._dynamo.graph_break()
                    if when == "at":
                        fail()
                    value = 3
                    x = x.sin()
                    fail()
                finally:
                    events.append(("inner", value))
            finally:
                events.append(("outer", value))

        compiled = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        expected_value = {"before": 1, "at": 2, "after": 3}[when]
        for _ in range(2):
            events = []
            with self.assertRaisesRegex(ValueError, "eager failure"):
                compiled(x, events)
            self.assertEqual(events, [("inner", expected_value), ("outer", expected_value)])

    @parametrize("finish", ["normal", "return", "raise"])
    def test_finally_control_flow(self, finish):
        @torch.compiler.disable
        def fail():
            raise ValueError("body failure")

        def fn(x, events):
            try:
                fail()
                return x.sin()
            finally:
                events.append("end")
                if finish == "return":
                    return x.cos()
                if finish == "raise":
                    raise RuntimeError("cleanup failure")

        compiled = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        events = []
        if finish == "return":
            self.assertEqual(compiled(x, events), x.cos())
        else:
            error = ValueError if finish == "normal" else RuntimeError
            message = "body failure" if finish == "normal" else "cleanup failure"
            with self.assertRaisesRegex(error, message):
                compiled(x, events)
        self.assertEqual(events, ["end"])

    def test_compiled_child_exception_runs_finally(self):
        def backend(gm, inputs):
            def fail(*args):
                raise ValueError("compiled child failure")

            return fail

        def child(x):
            return x.sin()

        def fn(x, events):
            try:
                torch._dynamo.graph_break()
                return child(x)
            finally:
                events.append("end")

        compiled = torch.compile(fn, backend=backend)
        events = []
        with self.assertRaisesRegex(ValueError, "compiled child failure"):
            compiled(torch.randn(4), events)
        self.assertEqual(events, ["end"])

    @parametrize("strict", ["fullgraph", "error_on_graph_break"])
    def test_strict_modes_still_raise(self, strict):
        def fn(x):
            try:
                torch._dynamo.graph_break()
                return x.sin()
            finally:
                pass

        compiled = torch.compile(fn, backend="eager", fullgraph=strict == "fullgraph")
        with torch._dynamo.error_on_graph_break(strict == "error_on_graph_break"):
            with self.assertRaises(Unsupported):
                compiled(torch.randn(4))

    def test_reset_releases_fallback_globals(self):
        def fn(x, skip):
            try:
                if skip:
                    torch._dynamo.graph_break()
                return x.sin()
            finally:
                pass

        prefix = "__builtins_dict___"
        before = {name for name in fn.__globals__ if name.startswith(prefix)}
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)
        x = torch.randn(4)
        self.assertEqual(compiled(x, True), x.sin())
        self.assertEqual(compiled(x, False), x.sin())
        self.assertEqual(counter.frame_count, 1)
        torch._dynamo.reset()
        gc.collect()
        after = {name for name in fn.__globals__ if name.startswith(prefix)}
        self.assertEqual(after, before)


instantiate_parametrized_tests(GuardedEagerFallbackTests)

if __name__ == "__main__":
    run_tests()
