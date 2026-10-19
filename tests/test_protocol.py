"""CPU checks for the experimental mechanism and training continuation."""
import contextlib
import io
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from basemodel import Global_interaction
from models import GATraj
from motion_interaction import MotionEdgeEncoder, motion_geometry
from Processor import Processor
from train import get_parser, prepare_seed


def arguments(**overrides):
    args = get_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_model(args):
    prepare_seed(3)
    with contextlib.redirect_stdout(io.StringIO()):
        return GATraj(args)


def example_inputs():
    torch.manual_seed(4)
    absolute = torch.randn(20, 3, 2).cumsum(0) * 0.05
    normalized = absolute - absolute[7:8]
    mask = np.ones((3, 3)) - np.eye(3)
    neighbors = [np.repeat(mask[None], 20, axis=0)]
    count = torch.tensor(mask.sum(1)).float()[:, None].expand(3, 20)
    return absolute, normalized, neighbors, count, np.array([[0, 3]])


class MotionTests(unittest.TestCase):
    def test_approach_and_stationary_are_finite(self):
        observed = torch.tensor([[[0., 0.], [6., 0.]], [[1., 0.], [5., 0.]]])
        _, risk, confidence, geometry = motion_geometry(observed, horizon=12)
        self.assertAlmostEqual(geometry['time'][0, 1].item(), 2.)
        self.assertAlmostEqual(geometry['closest'][0, 1].item(), 0.)
        self.assertTrue(geometry['approaching'][0, 1])
        self.assertTrue(torch.isfinite(risk).all())
        self.assertTrue(torch.equal(confidence, torch.ones_like(confidence)))
        stationary = observed[-1:].expand(8, -1, -1)
        _, risk, _, geometry = motion_geometry(stationary)
        self.assertTrue(torch.isfinite(risk).all())
        self.assertFalse(geometry['approaching'].any())

    def test_confidence_reduces_noisy_motion(self):
        observed = torch.zeros(4, 2, 2)
        observed[:, 0, 0] = torch.tensor([0., 1., 0., 1.])
        observed[:, 1, 0] = torch.tensor([4., 5., 6., 7.])
        _, _, confidence, _ = motion_geometry(observed)
        self.assertLess(confidence[0, 1].item(), confidence[1, 1].item())

    def test_adaptive_neighbors_are_bounded_and_no_self_edges(self):
        args = arguments(adaptive_neighbors=True)
        module = MotionEdgeEncoder(args)
        observed = torch.tensor([[[0., 0.], [14., 0.], [100., 0.]],
                                 [[1., 0.], [13., 0.], [99., 0.]]])
        _, mask = module(observed, torch.zeros(3, 3))
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask.diagonal().any())
        self.assertFalse(mask[0, 2])

    def test_attention_keeps_zero_scores_and_preserves_isolated_agents(self):
        args = arguments()
        layer = Global_interaction(args)
        torch.nn.init.zeros_(layer.WAr.linear.weight)
        torch.nn.init.zeros_(layer.WAr.linear.bias)
        hidden, cell = torch.randn(3, 64), torch.randn(3, 64)
        mask = torch.tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        updated, updated_cell = layer(torch.randn(3, 3, 2), mask, mask.sum(1), hidden, cell)
        torch.testing.assert_close(updated[1:], hidden[1:], rtol=0, atol=0)
        torch.testing.assert_close(updated_cell[1:], cell[1:], rtol=0, atol=0)
        self.assertFalse(torch.equal(updated[0], hidden[0]))
        self.assertTrue(torch.isfinite(updated).all())

    def test_zero_initialized_residual_starts_at_shared_baseline(self):
        baseline = make_model(arguments()).eval()
        modified = make_model(arguments(motion_module=True)).eval()
        inputs = example_inputs()
        with torch.no_grad():
            loss_a, predictions_a = baseline(inputs, 1)
            loss_b, predictions_b = modified(inputs, 1)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        for a, b in zip(predictions_a, predictions_b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_gradients_and_future_label_independence(self):
        model = make_model(arguments(motion_module=True))
        inputs = example_inputs()
        loss, _ = model(inputs, 1)
        loss.backward()
        self.assertGreater(model.motion_edges.projection[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.Global_interaction[0].WAr.linear.weight.grad.abs().sum().item(), 0)
        # Activate the residual so zero initialization cannot hide a future leak.
        torch.nn.init.normal_(model.motion_edges.projection[-1].weight, std=0.01)
        model.eval()
        altered = list(inputs)
        altered[0] = inputs[0].clone()
        altered[1] = inputs[1].clone()
        altered[0][8:] += 1000
        altered[1][8:] -= 1000
        with torch.no_grad():
            original_mixture = model(inputs, 1, return_mixture=True)
            altered_mixture = model(tuple(altered), 1, return_mixture=True)
        for original, changed in zip(original_mixture, altered_mixture):
            torch.testing.assert_close(original, changed, rtol=0, atol=0)

    def test_component_ablations_and_singletons(self):
        for velocity, risk, confidence, adaptive in [
                (True, False, False, False), (False, True, False, False),
                (True, True, False, False), (True, True, True, True)]:
            model = make_model(arguments(motion_module=True, relative_velocity=velocity,
                                        closest_approach=risk, confidence_gate=confidence,
                                        adaptive_neighbors=adaptive)).eval()
            with torch.no_grad():
                loss, predictions = model(example_inputs(), 1)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(tuple(predictions[0].shape), (19, 3, 2))
        inputs = list(example_inputs())
        inputs[0], inputs[1] = inputs[0][:, :1], inputs[1][:, :1]
        inputs[2], inputs[3], inputs[4] = [np.zeros((20, 1, 1))], torch.zeros(1, 20), [[0, 1]]
        with torch.no_grad():
            loss, _ = model(tuple(inputs), 1)
        self.assertTrue(torch.isfinite(loss))


class ResumeTests(unittest.TestCase):
    def processor(self, directory, load_model=0, **kwargs):
        prepare_seed(8)
        processor = Processor.__new__(Processor)
        processor.args = arguments(save_dir=directory, train_model='check',
                                   model_dir=str(Path(directory) / 'check'),
                                   load_model=load_model, **kwargs)
        processor.net = torch.nn.Linear(2, 2)
        processor.lr = processor.args.learning_rate
        processor.set_optimizer()
        processor.epoch = 1
        processor.load_model()
        return processor

    def step(self, processor):
        processor.optimizer.zero_grad()
        values = torch.randn(5, 2) + np.random.random() + random.random()
        processor.net(values).square().mean().backward()
        processor.optimizer.step()
        processor.scheduler.step()

    def test_exact_cpu_resume_and_schedule_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'check').mkdir()
            uninterrupted = self.processor(directory)
            for _ in range(4):
                self.step(uninterrupted)
            expected = {key: value.clone() for key, value in uninterrupted.net.state_dict().items()}
            segmented = self.processor(directory)
            for _ in range(2):
                self.step(segmented)
            segmented.save_model(2)
            resumed = self.processor(directory, load_model=2)
            self.assertEqual(resumed.epoch, 3)
            self.assertEqual(resumed.scheduler.T_max, 1000)
            for _ in range(2):
                self.step(resumed)
            for key, value in resumed.net.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertEqual(uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict())
            with self.assertRaisesRegex(ValueError, 'num_epochs'):
                self.processor(directory, load_model=2, num_epochs=100)

    def test_gpu_state_remapping_with_fewer_devices(self):
        # Mock only device-state APIs; this does not claim CUDA kernel coverage.
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'check').mkdir()
            processor = self.processor(directory)
            processor.args.gpu = 1
            states = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
            with patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.get_rng_state_all', return_value=states):
                processor.save_model(2)
            with patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.device_count', return_value=1), \
                    patch('torch.cuda.set_rng_state_all') as restore_all, \
                    patch('torch.cuda.set_rng_state') as restore_active:
                self.processor(directory, load_model=2)
            self.assertEqual(len(restore_all.call_args.args[0]), 1)
            torch.testing.assert_close(restore_active.call_args.args[0], states[1])
            self.assertEqual(restore_active.call_args.kwargs['device'], 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
