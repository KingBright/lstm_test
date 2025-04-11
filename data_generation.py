# data_generation.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config # Import config for parameters like SEED, DT, T_SPAN_LONG etc.

# --- Global Torque Handling ---
# (保持不变)
_global_tau_values = None
_global_t_span_start = 0
_global_dt = config.DT
def set_global_torque_params(tau_values, t_span_start, dt):
    global _global_tau_values, _global_t_span_start, _global_dt
    if tau_values is not None:
        _global_tau_values = tau_values; _global_t_span_start = t_span_start
        _global_dt = dt if dt > 0 else config.DT
    else: _global_tau_values = None
def get_tau_at_time_global(time):
    if _global_tau_values is None or _global_dt <= 0 or len(_global_tau_values) == 0: return 0.0
    idx = int((time - _global_t_span_start) / _global_dt)
    idx = max(0, min(idx, len(_global_tau_values) - 1)); return _global_tau_values[idx]

# --- Pendulum System Definition ---
# (保持不变)
class PendulumSystem:
    def __init__(self, m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF):
        self.m = m; self.L = L; self.g = g; self.c = c
        if self.m <= 0 or self.L <= 0: raise ValueError("Mass (m) and Length (L) must be positive.")
        self.beta = self.c / (self.m * self.L**2)
        self.omega0_sq = self.g / self.L
        # print(f"PendulumSystem initialized: m={m}, L={L}, g={g}, c={c} -> beta={self.beta:.3f}, omega0_sq={self.omega0_sq:.3f}")

    def ode(self, t, x, current_tau=None): # Accepts tau directly
        theta, theta_dot = x
        tau = current_tau if current_tau is not None else get_tau_at_time_global(t) # Fallback for solve_ivp if needed
        dtheta_dt = theta_dot
        dtheta_dot_dt = (-self.beta * theta_dot - self.omega0_sq * np.sin(theta) + tau / (self.m * self.L**2))
        return np.array([dtheta_dt, dtheta_dot_dt])

# --- Torque Sequence Generation - CORRECTED ---
def generate_torque_sequence(t, type="highly_random"): # Default to highly_random
    """Generates different types of torque input sequences."""
    t = np.asarray(t)
    if t.ndim == 0: t = np.array([t])
    num_steps = len(t)
    if num_steps == 0: return np.array([])
    max_time = t.max() if len(t) > 0 else 0

    np.random.seed(config.SEED) # Ensure reproducibility for random generation

    if type == "zero":
        return np.zeros_like(t)
    elif type == "step":
        tau = np.zeros_like(t); step_time = max_time * 0.3; tau[t >= step_time] = 1.0; return tau
    elif type == "sine":
        amplitude = 0.5; frequency = 0.5; return amplitude * np.sin(2 * np.pi * frequency * t)
    elif type == "random": # Original piecewise random
        segment_length = 20
        # --- VVVVVV Syntax Corrected VVVVVV ---
        if segment_length <= 0:
            segment_length = 1
        # --- ^^^^^^ Syntax Corrected ^^^^^^ ---
        num_segments = (num_steps + segment_length - 1) // segment_length; segments = np.random.uniform(-0.8, 0.8, num_segments)
        repeated_segments = np.repeat(segments, segment_length); return repeated_segments[:num_steps]
    elif type == "mixed": # Original mixed sequence
        tau = np.zeros_like(t); quarter = num_steps // 4; idx_q1, idx_q2, idx_q3 = quarter, 2 * quarter, 3 * quarter
        if idx_q1 < idx_q2: tau[idx_q1:idx_q2] = 0.8 # Step
        t_sine_segment = t[idx_q2:idx_q3];
        if len(t_sine_segment) > 0: sine_part = 0.5 * np.sin(2 * np.pi * 1.0 * t_sine_segment); tau[idx_q2 : idx_q2 + len(sine_part)] = sine_part # Sine
        segment_length = 10; random_start_idx = idx_q3; num_random_steps_needed = num_steps - random_start_idx
        if num_random_steps_needed > 0 and segment_length > 0:
             num_segments = (num_random_steps_needed + segment_length - 1) // segment_length; segments = np.random.uniform(-0.8, 0.8, num_segments)
             repeated_segments = np.repeat(segments, segment_length); tau[random_start_idx:] = repeated_segments[:num_random_steps_needed] # Random
        return tau
    elif type == "highly_random": # New random type
        tau = np.zeros_like(t)
        change_steps = config.TORQUE_CHANGE_STEPS
        # --- VVVVVV Syntax Corrected VVVVVV ---
        if change_steps <= 0:
             change_steps = 1
        # --- ^^^^^^ Syntax Corrected ^^^^^^ ---
        num_segments = (num_steps + change_steps - 1) // change_steps
        # Generate random values for each segment from the specified range
        torque_values_for_segments = np.random.uniform(config.TORQUE_RANGE[0], config.TORQUE_RANGE[1], num_segments)
        # Repeat values for the duration of each segment
        tau = np.repeat(torque_values_for_segments, change_steps)[:num_steps] # Ensure correct length
        # print(f"  Generated 'highly_random' torque: {num_segments} segments, changing every {change_steps} steps.") # Less verbose
        return tau
    else:
        raise ValueError(f"Unknown torque type: {type}")


# --- Simulation Execution (using solve_ivp) ---
# (保持不变, 使用修正后的版本)
def run_simulation(pendulum, t_span, dt, x0, tau_values, t_eval=None):
    if t_eval is None: t_eval = np.arange(t_span[0], t_span[1] + dt/2, dt) # Include endpoint approx
    if not isinstance(t_eval, np.ndarray) or t_eval.ndim != 1 or len(t_eval) == 0: return np.array([]), np.array([]), np.array([])
    effective_t_span = (t_eval[0], t_eval[-1])
    set_global_torque_params(tau_values, t_span[0], dt)
    try:
        sol = solve_ivp(lambda t, x: pendulum.ode(t, x, current_tau=None), # Use global lookup for tau
                        effective_t_span, x0, method='RK45', t_eval=t_eval, dense_output=False)
        set_global_torque_params(None, 0, config.DT) # Reset globals
        if sol.status != 0: print(f"Warning: ODE solver status {sol.status}: {sol.message}"); return np.array([]), np.array([]), np.array([])
        if len(sol.t) != len(t_eval): print(f"Warning: Solver output length mismatch.")
        return sol.t, sol.y[0], sol.y[1]
    except Exception as e: print(f"Error during ODE solving: {e}"); set_global_torque_params(None, 0, config.DT); return np.array([]), np.array([]), np.array([])

# --- Generate Simulation DataFrame - Using Correct Default ---
# VVVVVV 确认默认 t_span 是 T_SPAN_LONG (如果使用长轨迹策略) 或 T_SPAN_SHORT (如果使用短仿真策略) VVVVVV
# 根据你当前的策略选择一个，这里假设我们回到了长轨迹策略
def generate_simulation_data(pendulum, t_span=config.T_SPAN_LONG, dt=config.DT, x0=None, torque_type="highly_random"):
    """Generates simulation data for a given configuration."""
    if x0 is None: x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
    elif len(x0) != 2: print("Warning: Invalid x0 provided, using random."); x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]

    # Ensure t_full covers the span correctly for arange
    t_full = np.arange(t_span[0], t_span[1] + dt/2, dt)
    if len(t_full) == 0: return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    tau_values = generate_torque_sequence(t_full, type=torque_type)

    # Pass t_full as t_eval to ensure output at desired points
    time_points, theta_values, theta_dot_values = run_simulation(pendulum, t_span, dt, x0, tau_values, t_eval=t_full)

    if len(time_points) == 0: return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    # Align torque values with actual output times
    set_global_torque_params(tau_values, t_span[0], dt); tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]; set_global_torque_params(None, 0, config.DT)

    data = {'time': time_points, 'theta': theta_values, 'theta_dot': theta_dot_values, 'tau': tau_at_output_times}; df = pd.DataFrame(data)
    if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any(): print("Warning: Simulation resulted in NaN or Inf values.")
    return df


# --- RK4 Step Function (保持不变) ---
def rk4_step(pendulum, state_t, tau_t, dt):
    """ Performs a single RK4 integration step. """
    # ... (内容不变) ...
    t_dummy = 0
    try:
        k1 = pendulum.ode(t_dummy, state_t,       current_tau=tau_t); k2 = pendulum.ode(t_dummy, state_t + 0.5*dt*k1, current_tau=tau_t); k3 = pendulum.ode(t_dummy, state_t + 0.5*dt*k2, current_tau=tau_t); k4 = pendulum.ode(t_dummy, state_t + dt*k3,     current_tau=tau_t)
        state_t_plus_dt = state_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.all(np.isfinite(state_t_plus_dt)): print(f"Warning: NaN/Inf detected during RK4 step."); return np.array([np.nan, np.nan])
        return state_t_plus_dt
    except Exception as e: print(f"Error during RK4 step calculation: {e}"); return np.array([np.nan, np.nan])

