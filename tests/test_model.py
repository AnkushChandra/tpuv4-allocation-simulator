import unittest

import numpy as np

from tpusim.model import (Pod, largest_job, reconfigurable_analytic, reconfigurable_find,
                          reconfigurable_fits, sample_down, sample_occupied, static_find,
                          static_fits, static_shapes, success_rates, valid_static_sizes)
from tpusim.runtime import POLICIES, RunConfig, simulate_job


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.pod = Pod()
        self.rng = np.random.default_rng(0)

    def test_pod_dimensions(self):
        self.assertEqual(self.pod.machines, 1024)
        self.assertEqual(self.pod.cubes, 64)
        self.assertEqual(self.pod.chips_per_cube, 64)

    def test_shapes(self):
        self.assertEqual(static_shapes(8, 16), ((4, 4),))
        self.assertEqual(static_shapes(8, 32), ((4, 8), (8, 4)))
        self.assertEqual(static_shapes(8, 11), ())
        self.assertIn(64, valid_static_sizes(self.pod))

    def test_healthy_pod_fits_everything(self):
        blocked = np.zeros((1, 32, 32), dtype=bool)
        for c in valid_static_sizes(self.pod):
            self.assertTrue(static_fits(self.pod, blocked, c)[0])
            self.assertTrue(reconfigurable_fits(self.pod, blocked, c)[0])

    def test_single_failure_blocks_full_pod_for_both(self):
        blocked = np.zeros((1, 32, 32), dtype=bool)
        blocked[0, 17, 5] = True
        self.assertFalse(static_fits(self.pod, blocked, 64)[0])
        self.assertFalse(reconfigurable_fits(self.pod, blocked, 64)[0])
        self.assertTrue(reconfigurable_fits(self.pod, blocked, 63)[0])

    def test_scattered_occupancy_hurts_only_static(self):
        blocked = np.zeros((1, 32, 32), dtype=bool)
        blocked[0, 16:20, 16:20] = True  # one occupied cube in the middle
        self.assertTrue(reconfigurable_fits(self.pod, blocked, 63)[0])
        self.assertFalse(static_fits(self.pod, blocked, 36)[0])

    def test_reconfigurable_matches_binomial(self):
        p = 0.01
        for c in (16, 48, 56):
            _, mc = success_rates(self.rng, self.pod, p, c, trials=20000)
            exact = reconfigurable_analytic(self.pod, p, c)
            self.assertAlmostEqual(mc, exact, delta=4 * np.sqrt(exact * (1 - exact) / 20000) + 1e-3)

    def test_find_returns_right_size(self):
        blocked = sample_down(self.rng, self.pod, 0.002, 1)[0]
        for find in (static_find, reconfigurable_find):
            alloc = find(self.pod, blocked, 16)
            self.assertIsNotNone(alloc)
            self.assertEqual(alloc.sum(), 16 * 16)
            self.assertFalse((alloc & blocked).any())

    def test_packed_occupancy_is_contiguous(self):
        occ = sample_occupied(self.rng, self.pod, 0.5, 1, layout="packed")[0]
        self.assertTrue(occ[:16].all())
        self.assertFalse(occ[16:].any())

    def test_largest_job_healthy(self):
        s, r = largest_job(self.pod, np.zeros((2, 32, 32), dtype=bool))
        self.assertEqual(list(s), [64, 64])
        self.assertEqual(list(r), [64, 64])


class RuntimeTests(unittest.TestCase):
    def test_no_failures_means_full_goodput(self):
        pod = Pod()
        for policy in POLICIES:
            rc = RunConfig(job_cubes=16, policy=policy, fail_per_machine_day=1e-12)
            res = simulate_job(pod, rc, np.random.default_rng(1))
            self.assertAlmostEqual(res.goodput, 1.0, places=6)
            self.assertEqual(res.interruptions, 0)

    def test_failures_cost_time(self):
        pod = Pod()
        rc = RunConfig(job_cubes=32, policy="reconfig", fail_per_machine_day=0.005)
        res = simulate_job(pod, rc, np.random.default_rng(2))
        self.assertGreater(res.interruptions, 0)
        self.assertLess(res.goodput, 1.0)


if __name__ == "__main__":
    unittest.main()
