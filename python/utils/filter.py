import math
import numpy as np

class OneEuroFilter:
  def __init__(self, x0, t0, dx0=None, min_cutoff=1.0, beta=0.0,
                d_cutoff=1.0):
    self.min_cutoff = float(min_cutoff)
    self.beta = float(beta)
    self.d_cutoff = float(d_cutoff)
    self.x_prev = x0
    self.dx_prev = np.zeros_like(x0) if dx0 is None else dx0
    self.t_prev = float(t0)

  def __call__(self, x, t=None, dt=None):
    t_e = dt if dt is not None else t - self.t_prev

    a_d = self.smoothing_factor(t_e, self.d_cutoff)
    dx = (x - self.x_prev) / max(t_e, 0.001)
    dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

    cutoff = self.min_cutoff + self.beta * abs(dx_hat)
    a = self.smoothing_factor(t_e, cutoff)
    x_hat = self.exponential_smoothing(a, x, self.x_prev)

    self.x_prev = x_hat
    self.dx_prev = dx_hat
    self.t_prev = t if t is not None else self.t_prev + dt
    return x_hat

  def smoothing_factor(self, t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

  def multi_smoothing_factor(self, t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return np.divide(r, r+1)

  def exponential_smoothing(self, a, x, x_prev):
    return a * x + (1 - a) * x_prev