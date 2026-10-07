# Owner(s): ["module: cuda"]

import ctypes
import gc
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestStablePhilox(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lib = ctypes.PyDLL(torch._C.__file__)
        handle = ctypes.c_void_p
        output = ctypes.POINTER(handle)
        signatures = {
            "torch_generator_from_pyobject": [ctypes.py_object, output],
            "torch_get_default_generator": [ctypes.c_int32, ctypes.c_int32, output],
            "torch_delete_generator": [handle],
            "torch_generator_philox_state": [
                handle,
                ctypes.c_uint64,
                output,
                output,
                output,
            ],
            "torch_tensor_to_pyobject": [handle, handle, output],
            "aoti_torch_delete_tensor_object": [handle],
        }
        for name, args in signatures.items():
            fn = getattr(cls.lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
        cls.reserve_fn = ctypes.CDLL(torch._C.__file__).torch_generator_philox_state
        cls.reserve_fn.argtypes = signatures["torch_generator_philox_state"]
        cls.reserve_fn.restype = ctypes.c_int
        cls.lib.torch_exception_get_what.restype = ctypes.c_char_p
        ctypes.pythonapi.Py_DecRef.argtypes = [ctypes.py_object]

    def check(self, result):
        if result:
            raise RuntimeError(self.lib.torch_exception_get_what().decode())

    def generator(self, obj):
        handle = ctypes.c_void_p()
        self.check(self.lib.torch_generator_from_pyobject(obj, ctypes.byref(handle)))
        self.addCleanup(self.lib.torch_delete_generator, handle)
        return handle

    def reserve(self, handle, increment):
        outputs = [ctypes.c_void_p() for _ in range(3)]
        self.check(
            self.reserve_fn(handle, increment, *(ctypes.byref(out) for out in outputs))
        )
        tensors = []
        try:
            for out in outputs:
                py = ctypes.c_void_p()
                self.check(
                    self.lib.torch_tensor_to_pyobject(out, None, ctypes.byref(py))
                )
                tensor = ctypes.cast(py, ctypes.py_object).value
                ctypes.pythonapi.Py_DecRef(tensor)
                tensors.append(tensor)
        finally:
            for out in outputs:
                self.check(self.lib.aoti_torch_delete_tensor_object(out))
        return tuple(tensors)

    def test_type_and_backend_errors(self):
        for obj in (None, 1, torch.empty(0)):
            with self.assertRaisesRegex(RuntimeError, "expected torch.Generator"):
                self.generator(obj)
        with self.assertRaisesRegex(RuntimeError, "philox_state"):
            self.reserve(self.generator(torch.Generator()), 4)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_seed_offset_and_shared_state(self):
        gen = torch.Generator(device="cuda").manual_seed(2**64 - 2)
        handle = self.generator(gen)
        actual = self.reserve(handle, 5)
        self.assertEqual([t.device.type for t in actual], ["cpu"] * 3)
        self.assertEqual([t.item() for t in actual], [-2, 0, 0])
        self.assertEqual(gen.get_offset(), 8)
        reference = gen.philox_state(4)
        self.assertEqual([t.item() for t in reference], [-2, 8, 0])
        self.assertEqual(self.reserve(handle, 4)[1].item(), 12)
        del gen
        gc.collect()
        self.assertEqual(self.reserve(handle, 4)[1].item(), 16)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_concurrent_reservations(self):
        gen = torch.Generator(device="cuda").manual_seed(123)
        handle = self.generator(gen)
        with ThreadPoolExecutor(max_workers=4) as executor:
            offsets = list(
                executor.map(lambda _: self.reserve(handle, 4)[1].item(), range(32))
            )
        self.assertEqual(sorted(offsets), list(range(0, 128, 4)))
        self.assertEqual(gen.get_offset(), 128)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_default_generator_identity(self):
        torch.cuda.init()
        device = torch.cuda.current_device()
        self.lib.aoti_torch_device_type_cuda.restype = ctypes.c_int32
        handle = ctypes.c_void_p()
        self.check(
            self.lib.torch_get_default_generator(
                self.lib.aoti_torch_device_type_cuda(), device, ctypes.byref(handle)
            )
        )
        self.addCleanup(self.lib.torch_delete_generator, handle)
        gen = torch.cuda.default_generators[device]
        state = gen.get_state()
        self.addCleanup(gen.set_state, state)
        gen.manual_seed(321)
        self.assertEqual(self.reserve(handle, 4)[0].item(), 321)
        self.assertEqual(gen.get_offset(), 4)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_capture_replay_and_multiple_graphs(self):
        gen = torch.Generator(device="cuda").manual_seed(123)
        handle = self.generator(gen)
        self.reserve(handle, 0)
        stream = torch.cuda.Stream()
        graphs = []
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                seed, offset, intra = self.reserve(handle, 4)
                result = seed + offset + intra.item()
            self.assertEqual(seed.device.type, "cuda")
            self.assertEqual(offset.device.type, "cuda")
            self.assertEqual(intra.device.type, "cpu")
            graphs.append((graph, result))
        del gen, seed, offset, intra
        gc.collect()
        expected = 123
        for _ in range(3):
            for graph, result in graphs:
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(result.item(), expected)
                expected += 4


if __name__ == "__main__":
    run_tests()
