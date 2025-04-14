# data_generation.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config # Import config for parameters like SEED, DT, T_SPAN_LONG etc.

# --- Global Torque Handling ---
# (No changes needed here)
_global_tau_values = None
_global_t_span_start = 0
_global_dt = config.DT
def set_global_torque_params(tau_values, t_span_start, dt):
    """Sets global torque parameters for use by the ODE solver."""
    global _global_tau_values, _global_t_span_start, _global_dt
    if tau_values is not None:
        _global_tau_values = tau_values
        _global_t_span_start = t_span_start
        _global_dt = dt if dt > 0 else config.DT
    else:
        _global_tau_values = None

def get_tau_at_time_global(time):
    """Gets the torque value at a specific time from the global sequence."""
    if _global_tau_values is None or _global_dt <= 0 or len(_global_tau_values) == 0:
        return 0.0
    idx = int((time - _global_t_span_start) / _global_dt)
    idx = max(0, min(idx, len(_global_tau_values) - 1))
    return _global_tau_values[idx]

# --- Pendulum System Definition ---
# (No changes needed here)
class PendulumSystem:
    """Represents the damped pendulum system with external torque."""
    def __init__(self, m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF):
        self.m = m; self.L = L; self.g = g; self.c = c
        if self.m <= 0 or self.L <= 0: raise ValueError("Mass (m) and Length (L) must be positive.")
        self.beta = self.c / (self.m * self.L**2)
        self.omega0_sq = self.g / self.L
        # print(f"PendulumSystem initialized: m={m}, L={L}, g={g}, c={c} -> beta={self.beta:.3f}, omega0_sq={self.omega0_sq:.3f}")

    def ode(self, t, x, current_tau=None):
        """The ordinary differential equation for the pendulum system."""
        theta, theta_dot = x
        tau = current_tau if current_tau is not None else get_tau_at_time_global(t)
        dtheta_dt = theta_dot
        dtheta_dot_dt = (-self.beta * theta_dot - self.omega0_sq * np.sin(theta) + tau / (self.m * self.L**2))
        return np.array([dtheta_dt, dtheta_dot_dt])

# --- Enhanced Torque Sequence Generation ---
# (No changes from previous version)
def generate_torque_sequence(t, type="highly_random", torque_change_steps=None):
    """
    Generates different types of torque input sequences with refined parameter sampling.
    """
    t = np.asarray(t); num_steps = len(t)
    if num_steps == 0: return np.array([])
    dt = t[1] - t[0] if num_steps > 1 else config.DT
    max_time = t[-1] if num_steps > 0 else 0
    torque_min, torque_max = config.TORQUE_RANGE
    torque_range_size = torque_max - torque_min
    tau = np.zeros_like(t)

    if type == "zero": pass
    elif type == "step":
        num_steps_change = np.random.randint(1, 6)
        min_step_time = t[0] + 5 * dt; max_step_time = max_time - 5 * dt
        if max_step_time <= min_step_time: step_times = np.array([np.random.uniform(t[0], max_time)]) if max_time > t[0] else []
        else: step_times = np.sort(np.random.uniform(min_step_time, max_step_time, num_steps_change))
        last_time = t[0]; last_val = np.random.uniform(torque_min, torque_max)
        tau[t < step_times[0]] = last_val if len(step_times) > 0 else last_val
        for i, step_time in enumerate(step_times):
            current_val = np.random.uniform(torque_min, torque_max)
            if np.random.rand() < 0.3 and abs(current_val - last_val) < torque_range_size * 0.3:
                 if last_val < (torque_min + torque_max) / 2: current_val = np.random.uniform(last_val + torque_range_size * 0.3, torque_max)
                 else: current_val = np.random.uniform(torque_min, last_val - torque_range_size * 0.3)
                 current_val = np.clip(current_val, torque_min, torque_max)
            tau[(t >= last_time) & (t < step_time)] = last_val
            last_val = current_val; last_time = step_time
        tau[t >= last_time] = last_val
    elif type == "sine":
        max_amp = torque_range_size / 2.1; amplitude = np.random.uniform(max_amp * 0.1, max_amp)
        min_freq = 0.1 / (max_time if max_time > 0 else 1); max_freq = 8.0 / (max_time if max_time > 0 else 1)
        frequency = np.random.uniform(min_freq, max_freq); phase = np.random.uniform(0, 2 * np.pi)
        center_min = torque_min + amplitude; center_max = torque_max - amplitude
        if center_min >= center_max: center = (torque_min + torque_max) / 2
        else: center = np.random.uniform(center_min, center_max)
        tau = center + amplitude * np.sin(2 * np.pi * frequency * t + phase)
    elif type == "ramp":
        start_val = np.random.uniform(torque_min, torque_max); end_val = np.random.uniform(torque_min, torque_max)
        if np.random.rand() < 0.2:
            if np.random.rand() < 0.5: start_val, end_val = torque_min, torque_max
            else: start_val, end_val = torque_max, torque_min
        tau = np.linspace(start_val, end_val, num_steps)
    elif type == "highly_random":
        if torque_change_steps is None:
            if hasattr(config, 'TORQUE_CHANGE_STEPS_RANGE'): change_steps = np.random.randint(config.TORQUE_CHANGE_STEPS_RANGE[0], config.TORQUE_CHANGE_STEPS_RANGE[1] + 1)
            else: change_steps = getattr(config, 'TORQUE_CHANGE_STEPS', 20)
        else: change_steps = torque_change_steps
        change_steps = max(1, int(change_steps))
        num_segments = (num_steps + change_steps - 1) // change_steps
        segment_values = np.random.uniform(torque_min, torque_max, num_segments)
        tau = np.repeat(segment_values, change_steps)[:num_steps]
    else: print(f"Warning: Unknown torque type '{type}'. Using zero torque.")
    tau = np.clip(tau, torque_min, torque_max)
    return tau


# --- Simulation Execution (MODIFIED TO WRAP THETA) ---
def run_simulation(pendulum, t_span, dt, x0, tau_values, t_eval=None):
    """Runs the ODE simulation using solve_ivp and wraps the output theta."""
    if t_eval is None: t_eval = np.arange(t_span[0], t_span[1] + dt/2, dt)
    if not isinstance(t_eval, np.ndarray) or t_eval.ndim != 1 or len(t_eval) == 0:
        print("Warning: Invalid t_eval in run_simulation."); return np.array([]), np.array([]), np.array([])

    # Ensure tau_values match t_eval length
    if len(tau_values) < len(t_eval):
        # print(f"Warning: tau_values length ({len(tau_values)}) < t_eval length ({len(t_eval)}). Padding.") # Less verbose
        padding = np.full(len(t_eval) - len(tau_values), tau_values[-1] if len(tau_values)>0 else 0)
        tau_values = np.concatenate((tau_values, padding))
    elif len(tau_values) > len(t_eval):
         # print(f"Warning: tau_values length ({len(tau_values)}) > t_eval length ({len(t_eval)}). Truncating.") # Less verbose
         tau_values = tau_values[:len(t_eval)]

    effective_t_span = (t_eval[0], t_eval[-1])
    set_global_torque_params(tau_values, t_span[0], dt)

    try:
        sol = solve_ivp(fun=lambda t, x: pendulum.ode(t, x, None), t_span=effective_t_span, y0=x0, method='RK45', t_eval=t_eval, dense_output=False)
        set_global_torque_params(None, 0, config.DT) # Reset global
        if sol.status != 0: print(f"Warning: ODE solver status {sol.status}: {sol.message}"); return np.array([]), np.array([]), np.array([])
        if len(sol.t) != len(t_eval):
            print(f"Warning: Solver output length mismatch.");
            if abs(len(sol.t) - len(t_eval)) > 1: return np.array([]), np.array([]), np.array([])

        time_points_out = sol.t
        theta_values_out = sol.y[0]
        theta_dot_values_out = sol.y[1]

        # *** ADDED: Wrap theta output to [-pi, pi] ***
        # This ensures the angle stays within a consistent range for scaling and learning
        theta_values_wrapped = (theta_values_out + np.pi) % (2 * np.pi) - np.pi

        # Return time, WRAPPED theta, and theta_dot
        return time_points_out, theta_values_wrapped, theta_dot_values_out

    except Exception as e:
        print(f"Error during ODE solving: {e}")
        set_global_torque_params(None, 0, config.DT)
        return np.array([]), np.array([]), np.array([])


# --- Original Simulation Data Generation (Uses modified run_simulation) ---
def generate_simulation_data(pendulum, t_span=(0, config.SIMULATION_DURATION_MEDIUM), dt=config.DT, x0=None,
                            torque_type="highly_random", torque_change_steps=None):
    """
    Generates simulation data for a single trajectory. Uses run_simulation which now wraps theta.
    """
    if x0 is None: x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
    elif len(x0) != 2: print("Warning: Invalid x0 provided, using random."); x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]

    t_full = np.arange(t_span[0], t_span[1] + dt/2, dt)
    if len(t_full) == 0: return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    tau_values = generate_torque_sequence(t_full, type=torque_type, torque_change_steps=torque_change_steps)

    # run_simulation now returns wrapped theta
    time_points, theta_values_wrapped, theta_dot_values = run_simulation(
        pendulum, t_span, dt, x0, tau_values, t_eval=t_full
    )

    if len(time_points) == 0: return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    # Align original tau_values (used in simulation) with output time points
    set_global_torque_params(tau_values, t_span[0], dt)
    tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]
    set_global_torque_params(None, 0, config.DT)

    # Create DataFrame using the wrapped theta
    data = {'time': time_points, 'theta': theta_values_wrapped, 'theta_dot': theta_dot_values, 'tau': tau_at_output_times}
    df = pd.DataFrame(data)
    if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any():
        print("Warning: Simulation resulted in NaN or Inf values.")
    return df

# --- RK4 Step Function ---
# (No changes)
def rk4_step(pendulum, state_t, tau_t, dt):
    """ Performs a single RK4 integration step. """
    t_dummy = 0
    try:
        k1 = pendulum.ode(t_dummy, state_t, current_tau=tau_t)
        k2 = pendulum.ode(t_dummy, state_t + 0.5*dt*k1, current_tau=tau_t)
        k3 = pendulum.ode(t_dummy, state_t + 0.5*dt*k2, current_tau=tau_t)
        k4 = pendulum.ode(t_dummy, state_t + dt*k3, current_tau=tau_t)
        state_t_plus_dt = state_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.all(np.isfinite(state_t_plus_dt)): print(f"Warning: NaN/Inf detected during RK4 step."); return np.array([np.nan, np.nan])
        return state_t_plus_dt
    except Exception as e: print(f"Error during RK4 step calculation: {e}"); return np.array([np.nan, np.nan])
