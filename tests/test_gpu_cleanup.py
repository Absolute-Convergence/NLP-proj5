"""Cleanup must release interrupt-held objects without resetting the dataset."""
import ast
import contextlib
import gc
import io
import json
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

from notebook_helpers import NOTEBOOK, load_tag


def cleanup_namespace(cuda):
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    source = ''.join(next(c['source'] for c in notebook['cells']
                          if c['cell_type'] == 'code' and 'def free_gpu_resources():' in ''.join(c['source'])))
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    namespace = {'gc': gc}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return namespace, SimpleNamespace(cuda=cuda, _C=SimpleNamespace(_cuda_clearCublasWorkspaces=Mock()))


class GPUCleanupTests(unittest.TestCase):
    def test_running_the_cleanup_cell_actually_releases_resources(self):
        cuda = SimpleNamespace(is_initialized=lambda: True, empty_cache=Mock(), synchronize=Mock(),
                               memory_allocated=lambda: 0, memory_reserved=lambda: 0)
        namespace, fake_torch = cleanup_namespace(cuda)
        class Model:
            pass
        model = Model()
        reference = weakref.ref(model)
        overrides = [(model, 'weight', {'optim_bits': 32})]
        manager = SimpleNamespace(initialize=Mock(side_effect=overrides.clear))
        fake_bnb = SimpleNamespace(optim=SimpleNamespace(GlobalOptimManager=SimpleNamespace(
            get_instance=lambda: manager)))
        namespace.update({'model': model, 'data': ['keep']})
        del model
        with patch.dict(sys.modules, {'torch': fake_torch, 'bitsandbytes': fake_bnb}), \
             patch.object(sys, 'last_traceback', None, create=True), \
             patch.object(sys, 'last_value', None, create=True), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            load_tag('gpu_cleanup', namespace)
        cuda.empty_cache.assert_called_once()
        cuda.synchronize.assert_called_once()
        fake_torch._C._cuda_clearCublasWorkspaces.assert_called_once()
        manager.initialize.assert_called_once()
        self.assertIsNone(reference())
        self.assertNotIn('model', namespace)
        self.assertEqual(namespace['data'], ['keep'])
        self.assertIn('GPU cleanup:', output.getvalue())

    def test_interrupt_traceback_and_output_cache_release_objects(self):
        cuda = SimpleNamespace(is_initialized=lambda: True, empty_cache=Mock(), synchronize=Mock(),
                               memory_allocated=lambda: 0, memory_reserved=lambda: 0)
        namespace, fake_torch = cleanup_namespace(cuda)
        output_cache = {}
        shell = SimpleNamespace(InteractiveTB=SimpleNamespace(tb=None), last_execution_result=None,
                                run_line_magic=lambda name, args: output_cache.clear())

        class GPUObject:
            pass

        model = GPUObject()
        model_ref = weakref.ref(model)
        namespace['model'] = model
        output_cache[1] = model
        del model

        def interrupted():
            trainer = GPUObject()
            namespace['trainer_ref'] = weakref.ref(trainer)
            raise KeyboardInterrupt()

        try:
            interrupted()
        except KeyboardInterrupt as error:
            saved_error = error
        shell.InteractiveTB.tb = saved_error.__traceback__
        shell.last_execution_result = SimpleNamespace(error_in_exec=saved_error)
        namespace.update({'get_ipython': lambda: shell, 'data': ['keep'], 'tokenized_train': ['keep tokens']})
        with patch.dict(sys.modules, {'torch': fake_torch}), \
             patch.object(sys, 'last_traceback', saved_error.__traceback__, create=True), \
             patch.object(sys, 'last_value', saved_error, create=True), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            namespace['free_gpu_resources']()
            self.assertIsNone(sys.last_traceback)
            self.assertIsNone(sys.last_value)
            namespace['free_gpu_resources']()  # Safe to call again.
        self.assertIsNone(model_ref())
        self.assertIsNone(namespace['trainer_ref']())
        self.assertEqual(namespace['data'], ['keep'])
        self.assertEqual(namespace['tokenized_train'], ['keep tokens'])
        self.assertEqual(output_cache, {})
        self.assertEqual(cuda.empty_cache.call_count, 2)
        self.assertIn('allocated 0.00 GiB', output.getvalue())

    def test_does_not_initialize_cuda_and_explains_failed_cleanup(self):
        cuda = SimpleNamespace(is_initialized=lambda: False, empty_cache=Mock(), synchronize=Mock())
        namespace, fake_torch = cleanup_namespace(cuda)
        with patch.dict(sys.modules, {'torch': fake_torch}), \
             patch.object(sys, 'last_traceback', None, create=True), \
             patch.object(sys, 'last_value', None, create=True), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            namespace['free_gpu_resources']()
            cuda.empty_cache.assert_not_called()
            cuda.is_initialized = lambda: True
            cuda.empty_cache.side_effect = RuntimeError('device error')
            namespace['free_gpu_resources']()
        self.assertIn('CUDA is not initialized', output.getvalue())
        self.assertIn('Restart the kernel', output.getvalue())


if __name__ == '__main__':
    unittest.main()
