import math
import unittest

from least_squares import evaluate_fit, generate_simulated_data, least_squares, parse_xy_text


class LeastSquaresTests(unittest.TestCase):
    def test_exact_line(self):
        x = (0.0, 1.0, 2.0, 3.0)
        y = (1.0, 3.0, 5.0, 7.0)
        beta0, beta1 = least_squares(x, y)
        self.assertAlmostEqual(beta0, 1.0)
        self.assertAlmostEqual(beta1, 2.0)
        fit = evaluate_fit(x, y, beta0, beta1)
        self.assertAlmostEqual(fit.sse, 0.0)
        self.assertAlmostEqual(fit.r2 or 0.0, 1.0)

    def test_manual_fit_can_have_negative_r2(self):
        fit = evaluate_fit((0, 1, 2), (0, 1, 2), 100, -20)
        self.assertIsNotNone(fit.r2)
        self.assertLess(fit.r2, 0)

    def test_parse_multiple_separators(self):
        x, y = parse_xy_text("X,Y\n0,1\n1;3\n2 5")
        self.assertEqual(x, (0.0, 1.0, 2.0))
        self.assertEqual(y, (1.0, 3.0, 5.0))

    def test_reproducible_simulation(self):
        first = generate_simulated_data(2, 1, 0.75, 2026)
        second = generate_simulated_data(2, 1, 0.75, 2026)
        self.assertEqual(first, second)
        self.assertTrue(all(math.isfinite(value) for value in first[1]))

    def test_constant_x_rejected(self):
        with self.assertRaises(ValueError):
            least_squares((1, 1, 1), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
