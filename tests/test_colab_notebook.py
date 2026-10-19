"""Check Colab orchestration locally without Drive access or GPU training."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


NOTEBOOK = Path(__file__).resolve().parents[1] / 'Colab_RiskMotion.ipynb'


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.cells = {cell['id']: ''.join(cell['source']) for cell in self.notebook['cells']}

    def test_notebook_cells_compile(self):
        self.assertEqual(len(self.cells), len(self.notebook['cells']))
        self.assertNotIn('/kaggle/', NOTEBOOK.read_text(encoding='utf-8'))
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')

    def run_training_cell(self, drive_root, stop_at, resume_at=0, prune=True):
        namespace = {'Path': Path, 'DRIVE_ROOT': drive_root, 'sys': sys}
        exec(compile(self.cells['configure'], 'configure', 'exec'), namespace)
        namespace.update(STOP_AT_EPOCH=stop_at, PRUNE_INTERMEDIATE_CHECKPOINTS=prune)
        run_name = 'RiskMotion_v1_r1_c1_a0_attn1_s1'
        run_dir = namespace['OUTPUT_DIR'] / '1' / run_name
        run_dir.mkdir(parents=True)
        if resume_at:
            for epoch in (100, resume_at):
                (run_dir / f'{run_name}_{epoch}.tar').touch()
        unrelated = run_dir / 'keep_notes.txt'
        unrelated.touch()
        (run_dir / f'{run_name}_999.tar.tmp').touch()
        (run_dir / f'{run_name}_draft.tar').touch()
        calls = []

        def fake_training(command, check):
            self.assertTrue(check)
            self.assertEqual(command[1], 'train.py')
            flags = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(flags['--num_epochs'], '1000')
            self.assertEqual(Path(flags['--save_base_dir']), namespace['OUTPUT_DIR'])
            previous, count = int(flags['--load_model']), int(flags['--epochs_per_run'])
            calls.append((previous, count))
            for epoch in range(previous + 1, previous + count + 1):
                (run_dir / f'{run_name}_{epoch}.tar').touch()

        namespace['subprocess'] = SimpleNamespace(run=fake_training)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(self.cells['train'], 'train', 'exec'), namespace)
            # Rerunning the cell at the requested endpoint does not retrain.
            exec(compile(self.cells['train'], 'train', 'exec'), namespace)
        self.assertTrue(unrelated.exists())
        self.assertTrue((run_dir / f'{run_name}_draft.tar').exists())
        return namespace, calls

    def test_resume_from_drive_and_checkpoint_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace, calls = self.run_training_cell(Path(temporary), stop_at=305, resume_at=120)
            self.assertEqual(calls, [(120, 100), (220, 85)])
            self.assertEqual(set(namespace['checkpoint_files']()), {100, 200, 300, 305})
            self.assertEqual(namespace['latest_epoch'](), 305)

    def test_optional_retention_of_all_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace, calls = self.run_training_cell(Path(temporary), stop_at=3, prune=False)
            self.assertEqual(calls, [(0, 3)])
            self.assertEqual(set(namespace['checkpoint_files']()), {1, 2, 3})


if __name__ == '__main__':
    unittest.main()
