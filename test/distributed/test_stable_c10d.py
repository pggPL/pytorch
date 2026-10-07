# Owner(s): ["oncall: distributed"]

import ctypes
import gc
import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import run_tests, TestCase


class StableC10d:
    def __init__(self, group):
        self.lib = ctypes.CDLL(torch._C.__file__)
        self.py = ctypes.PyDLL(torch._C.__file__)
        handle = ctypes.c_void_p
        output = ctypes.POINTER(handle)
        tensor_array = ctypes.POINTER(handle)
        signatures = {
            "torch_process_group_from_pyobject": [ctypes.py_object, output],
            "torch_tensor_from_pyobject": [ctypes.py_object, output],
            "torch_process_group_rank": [handle, ctypes.POINTER(ctypes.c_int64)],
            "torch_process_group_size": [handle, ctypes.POINTER(ctypes.c_int64)],
            "torch_process_group_backend": [handle, output],
            "torch_string_c_str": [handle, ctypes.POINTER(ctypes.c_char_p)],
            "torch_delete_string": [handle],
            "torch_process_group_allreduce": [
                handle,
                tensor_array,
                ctypes.c_size_t,
                ctypes.c_int32,
                output,
            ],
            "torch_process_group_allreduce_coalesced": [
                handle,
                tensor_array,
                ctypes.c_size_t,
                ctypes.c_int32,
                output,
            ],
            "torch_process_group_broadcast": [
                handle,
                tensor_array,
                ctypes.c_size_t,
                ctypes.c_int64,
                ctypes.c_int64,
                output,
            ],
            "torch_process_group_allgather": [
                handle,
                handle,
                tensor_array,
                ctypes.c_size_t,
                output,
            ],
            "torch_process_group_barrier": [
                handle,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_size_t,
                output,
            ],
            "torch_work_wait": [handle, ctypes.c_int64, ctypes.POINTER(ctypes.c_bool)],
            "torch_work_is_completed": [handle, ctypes.POINTER(ctypes.c_bool)],
            "torch_delete_work": [handle],
            "torch_delete_process_group": [handle],
            "aoti_torch_delete_tensor_object": [handle],
        }
        for library in (self.lib, self.py):
            for name, args in signatures.items():
                fn = getattr(library, name)
                fn.argtypes = args
                fn.restype = ctypes.c_int
            library.torch_exception_get_what.restype = ctypes.c_char_p
        self.group = handle()
        self.check(
            self.py.torch_process_group_from_pyobject(group, ctypes.byref(self.group))
        )

    def check(self, result):
        if result:
            raise RuntimeError(self.lib.torch_exception_get_what().decode())

    def close(self):
        self.check(self.lib.torch_delete_process_group(self.group))
        self.group = None

    def tensors(self, tensors):
        handles = (ctypes.c_void_p * len(tensors))()
        for index, tensor in enumerate(tensors):
            out = ctypes.c_void_p()
            self.check(self.py.torch_tensor_from_pyobject(tensor, ctypes.byref(out)))
            handles[index] = out
        return handles

    def launch(self, name, tensors, *args):
        handles = self.tensors(tensors)
        work = ctypes.c_void_p()
        try:
            fn = getattr(self.lib, "torch_process_group_" + name)
            self.check(fn(self.group, handles, len(handles), *args, ctypes.byref(work)))
        finally:
            for handle in handles:
                self.check(self.lib.aoti_torch_delete_tensor_object(handle))
        return work

    def wait(self, work):
        done = ctypes.c_bool()
        try:
            self.check(self.lib.torch_work_wait(work, 30000, ctypes.byref(done)))
            if not done.value:
                raise RuntimeError("collective was aborted")
            self.check(self.lib.torch_work_is_completed(work, ctypes.byref(done)))
            if not done.value:
                raise RuntimeError("Gloo work is incomplete after wait")
        finally:
            self.check(self.lib.torch_delete_work(work))


def run_rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    case = TestCase()
    pg = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=45))
    api = StableC10d(pg)
    try:
        for name, expected in (("rank", rank), ("size", 2)):
            value = ctypes.c_int64()
            api.check(
                getattr(api.lib, "torch_process_group_" + name)(
                    api.group, ctypes.byref(value)
                )
            )
            case.assertEqual(value.value, expected)
        backend = ctypes.c_void_p()
        api.check(api.lib.torch_process_group_backend(api.group, ctypes.byref(backend)))
        name = ctypes.c_char_p()
        api.check(api.lib.torch_string_c_str(backend, ctypes.byref(name)))
        case.assertEqual(name.value, b"gloo")
        api.check(api.lib.torch_delete_string(backend))
        with case.assertRaisesRegex(RuntimeError, "expected ProcessGroup"):
            StableC10d(None)
        single_group = dist.new_group(
            [0], backend="gloo", timeout=timedelta(seconds=45)
        )
        if rank == 0:
            single = StableC10d(single_group)
            try:
                size = ctypes.c_int64()
                single.check(
                    single.lib.torch_process_group_size(
                        single.group, ctypes.byref(size)
                    )
                )
                case.assertEqual(size.value, 1)
                only = torch.full((2,), 11.0)
                single.wait(single.launch("allreduce", [only], 0))
                case.assertEqual(only, torch.full((2,), 11.0))
            finally:
                single.close()
                dist.destroy_process_group(single_group)
        value = torch.full((4,), float(rank + 1))
        api.wait(api.launch("allreduce", [value], 0))
        case.assertEqual(value, torch.full((4,), 3.0))
        tensors = [torch.full((2,), float(rank)), torch.full((3,), float(3 - rank))]
        api.wait(api.launch("allreduce_coalesced", tensors, 4))
        case.assertEqual(tensors, [torch.ones(2), torch.full((3,), 3.0)])
        value.fill_(rank + 7)
        api.wait(api.launch("broadcast", [value], 1, 0))
        case.assertEqual(value, torch.full((4,), 8.0))
        with case.assertRaisesRegex(RuntimeError, "invalid root rank"):
            api.launch("broadcast", [value], 2, 0)
        with case.assertRaisesRegex(RuntimeError, "invalid stable reduction"):
            api.launch("allreduce", [value], 99)
        with case.assertRaisesRegex(RuntimeError, "nonempty"):
            api.launch("allreduce", [], 0)
        outputs = [torch.empty(2), torch.empty(2)]
        inp = torch.full((2,), float(rank))
        ins = api.tensors([inp])
        outs = api.tensors(outputs)
        work = ctypes.c_void_p()
        try:
            api.check(
                api.lib.torch_process_group_allgather(
                    api.group, ins[0], outs, 2, ctypes.byref(work)
                )
            )
        finally:
            for handle in [*ins, *outs]:
                api.check(api.lib.aoti_torch_delete_tensor_object(handle))
        api.wait(work)
        case.assertEqual(outputs, [torch.zeros(2), torch.ones(2)])
        work = ctypes.c_void_p()
        api.check(
            api.lib.torch_process_group_barrier(api.group, None, 0, ctypes.byref(work))
        )
        api.wait(work)
        value.fill_(rank + 1)
        work = api.launch("allreduce", [value], 0)
        # Work keeps native group/tensor references after releasing stable handles.
        api.close()
        del pg
        gc.collect()
        api.wait(work)
        case.assertEqual(value, torch.full((4,), 3.0))
    finally:
        if api.group:
            api.close()
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
class TestStableC10d(TestCase):
    def test_two_rank_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                run_rank, args=(os.path.join(directory, "store"),), nprocs=2, join=True
            )


if __name__ == "__main__":
    run_tests()
