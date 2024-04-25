"""
Author: Ibrahim Almakky
Date: 19/05/2021

"""
import random
import unittest
import numpy as np


class Metric:
    def __init__(self, iters_tol, increasing=True) -> None:
        self.iters_tol = iters_tol
        self.increasing = increasing
        self.values = []

    def add_value(self, new_value):
        self.values.append(new_value)

    def terminate(self):
        if len(self.values) > self.iters_tol:
            if self.increasing:
                best_indx = np.argmax(self.values)
            else:
                best_indx = np.argmin(self.values)
            if self.iters_tol < (len(self.values) - best_indx):
                return True
        return False


class TerminationChecker:
    def __init__(self):
        self.metrics = {}

    def add_termination_cnd(self, name, iters_tol, increasing=True):
        self.metrics[name] = Metric(iters_tol, increasing=increasing)

    def update_metric(self, name, value):
        try:
            self.metrics[name].add_value(value)
        except KeyError as k_error:
            raise KeyError(
                "Metric must be fist added to the list of termination conditions."
            ) from k_error

    def terminate(self):
        votes = []
        for metric in self.metrics.values():
            votes.append(int(metric.terminate()))
        # All metrics should vote to terminate for
        # the experiment to be terminated.
        if np.sum(votes) < len(votes):
            return False
        return True


class TestTerminate(unittest.TestCase):
    def test_exp(self):
        loss = 1.5
        accuracy = 0
        epochs = 1000
        tolerance = random.randint(0, (epochs / 4))
        turning_point = random.randint(0, epochs)
        terminator = TerminationChecker()
        terminator.add_termination_cnd("accuracy", tolerance, increasing=True)
        terminator.add_termination_cnd("loss", tolerance, increasing=False)
        for epoch in range(0, epochs):
            if epoch > turning_point:
                loss += 0.01
                accuracy -= 0.1
            else:
                loss -= 0.001
                accuracy += 0.01
            # print("epoch %d, acc=%f, loss=%f" % (epoch, accuracy, loss))
            terminator.update_metric("loss", loss)
            terminator.update_metric("accuracy", accuracy)
            if terminator.terminate():
                exp_stop = tolerance + turning_point
                self.assertEqual(epoch, exp_stop)
                break


if __name__ == "__main__":
    unittest.main()
