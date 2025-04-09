# data_generation.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config # Import config for parameters

# --- Global Torque Handling ---
# Using globals is simpler for solve_ivp callback, but not ideal for complex scenarios.
_global_tau_values = None
_global_t_span_start = 0
_global_dt = 0.02

def set_global_torque_params(tau_values, t_span_start, dt):
    """Sets global parameters for the torque function used by the ODE solver."""
    global _global_tau_values, _global_t_span_start, _global_dt
    if tau_values is not None:
        _global_tau_values = tau_values
        _global_t_span_start = t_span_start
        _global_dt = dt
    else:
        # Reset or handle None case
        _global_tau_values = None
        print("Warning: Attempted to set None torque values globally.")


def get_tau_at_time_global(time):
    """
    Retrieves the torque value at a specific time using global parameters.
    NOTE: Relies on set_global_torque_params being called beforehand.
    """
    if _global_tau_values is None or _global_dt <= 0:
        # Handle cases where parameters aren't set or invalid
        # print(f"Warning: Global torque parameters not set or invalid (dt={_global_dt}). Returning 0 torque.")
        # Returning 0 might be okay for some steps but could mask errors.
        # Consider raising an error if this state is unexpected.
        return 0.0

    # Calculate index, ensuring it's within bounds
    idx = int((time - _global_t_span_start) / _global_dt)
    # Clamp index to valid range [0, len-1]
    idx = max(0, min(idx, len(_global_tau_values) - 1))
    return _global_tau_values[idx]


# --- Pendulum System Definition ---
class PendulumSystem:
    """Represents the damped, driven pendulum system."""
    def __init__(self, m=1.0, L=1.0, g=9.81, c=0.5):
        self.m = m
        self.L = L
        self.g = g
        self.c = c
        # Derived parameters
        # Add checks for L > 0, m > 0 to avoid division by zero
        if self.m <= 0 or self.L <= 0:
            raise ValueError("Mass (m) and Length (L) must be positive.")
        self.beta = self.c / (self.m * self.L**2)
        self.omega0_sq = self.g / self.L

    def ode(self, t, x):
        """
        The ODE function dx/dt = f(t, x) for the pendulum.
        Uses get_tau_at_time_global() to find the torque at time t.
        """
        theta, theta_dot = x
        current_tau = get_tau_at_time_global(t)

        dtheta_dt = theta_dot
        dtheta_dot_dt = (-self.beta * theta_dot -
                          self.omega0_sq * np.sin(theta) +
                          current_tau / (self.m * self.L**2))
        return np.array([dtheta_dt, dtheta_dot_dt])

# --- Torque Sequence Generation ---
def generate_torque_sequence(t, type="mixed"):
    """Generates different types of torque input sequences."""
    # Ensure t is a numpy array for vectorized operations
    t = np.asarray(t)
    if t.ndim == 0: # Handle scalar time input if necessary, though unlikely here
         t = np.array([t])
    if len(t) == 0:
         return np.array([]) # Return empty if time vector is empty

    max_time = t.max() if len(t) > 0 else 0

    if type == "zero":
        return np.zeros_like(t)
    elif type == "step":
        tau = np.zeros_like(t)
        step_time = max_time * 0.3
        tau[t >= step_time] = 1.0
        return tau
    elif type == "sine":
        amplitude = 0.5
        frequency = 0.5 # Hz
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif type == "random":
        np.random.seed(config.SEED) # Use seed from config
        segment_length = 20 # Number of time steps per segment
        num_steps = len(t)
        # Calculate num_segments needed, avoid division by zero
        num_segments = (num_steps + segment_length - 1) // segment_length if segment_length > 0 else 1
        segments = np.random.uniform(-0.8, 0.8, num_segments)
        repeated_segments = np.repeat(segments, segment_length)
        return repeated_segments[:num_steps] # Trim to exact length
    elif type == "mixed":
        tau = np.zeros_like(t)
        num_steps = len(t)
        quarter = num_steps // 4

        # Ensure indices are valid
        idx_q1 = quarter
        idx_q2 = 2 * quarter
        idx_q3 = 3 * quarter

        # Step part
        if idx_q1 < idx_q2: tau[idx_q1:idx_q2] = 0.8

        # Sine part
        t_sine_segment = t[idx_q2:idx_q3]
        if len(t_sine_segment) > 0:
            sine_part = 0.5 * np.sin(2 * np.pi * 1.0 * t_sine_segment)
            tau[idx_q2 : idx_q2 + len(sine_part)] = sine_part

        # Random part
        np.random.seed(config.SEED)
        segment_length = 10
        random_start_idx = idx_q3
        num_random_steps_needed = num_steps - random_start_idx

        if num_random_steps_needed > 0 and segment_length > 0:
             num_segments = (num_random_steps_needed + segment_length - 1) // segment_length
             segments = np.random.uniform(-0.8, 0.8, num_segments)
             repeated_segments = np.repeat(segments, segment_length)
             tau[random_start_idx:] = repeated_segments[:num_random_steps_needed]

        return tau
    else:
        raise ValueError(f"Unknown torque type: {type}")


# --- Simulation Execution ---
def run_simulation(pendulum, t_span, dt, x0, tau_values, t_eval=None): # Add t_eval parameter
    """
    Executes the simulation using solve_ivp for a given pendulum instance.
    Uses provided t_eval if available, otherwise generates from t_span, dt.
    """
    if t_eval is None:
         t_eval = np.arange(t_span[0], t_span[1], dt)

    if len(t_eval) == 0:
        return np.array([]), np.array([]), np.array([])

    # Set global torque parameters for the ODE function
    # Ensure tau_values length matches expected duration based on t_eval/t_span/dt
    # This assumes tau_values were generated correctly beforehand.
    if tau_values is not None:
         set_global_torque_params(tau_values, t_span[0], dt) # Use original t_span[0], dt for indexing tau
    else:
         print("Warning: tau_values is None in run_simulation.")
         # Decide behaviour - maybe default to zero torque?
         set_global_torque_params(np.zeros_like(t_eval), t_span[0], dt)


    try:
        sol = solve_ivp(
            pendulum.ode,
            (t_eval[0], t_eval[-1]), # Use actual min/max time from t_eval for span
            x0,
            method='RK45',
            t_eval=t_eval, # Use the provided evaluation times
            dense_output=False,
        )
        # Reset global torque parameters after use (good practice)
        # set_global_torque_params(None, 0, 0.02)

        if sol.status != 0:
            print(f"Warning: ODE solver finished with status {sol.status}: {sol.message}")
            return np.array([]), np.array([]), np.array([])

        # Return t (should match t_eval), theta, theta_dot
        return sol.t, sol.y[0], sol.y[1]

    except Exception as e:
        print(f"Error during ODE solving: {e}")
        # Reset global torque parameters in case of error
        # set_global_torque_params(None, 0, 0.02)
        return np.array([]), np.array([]), np.array([])


def generate_simulation_data(pendulum, t_span=(0, 10), dt=0.02, x0=None, torque_type="mixed"):
    """
    Generates the simulation data DataFrame by creating torque and running the simulation.
    """
    if x0 is None:
        x0 = [0.1, 0.0]

    # Generate time vector for torque generation
    t_full = np.arange(t_span[0], t_span[1], dt)
    if len(t_full) == 0:
        print(f"Warning: Simulation time span resulted in no time steps: {t_span}, dt={dt}")
        return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    tau_values = generate_torque_sequence(t_full, type=torque_type)

    # Run the simulation
    time_points, theta_values, theta_dot_values = run_simulation(pendulum, t_span, dt, x0, tau_values)

    if len(time_points) == 0:
        print(f"Warning: Simulation returned no results for t_span={t_span}, x0={x0}")
        return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    # Re-generate torque values aligned with the actual output times from solve_ivp
    # Re-setting global params just to be safe if run_simulation didn't reset
    set_global_torque_params(tau_values, t_span[0], dt)
    tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]
    # Reset global torque params after use
    set_global_torque_params(None, 0, 0.02)

    # Create DataFrame
    data = {
        'time': time_points,
        'theta': theta_values,
        'theta_dot': theta_dot_values,
        'tau': tau_at_output_times
    }
    df = pd.DataFrame(data)

    # Optional: Add checks for NaNs or Infs in the resulting DataFrame
    if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any():
        print("Warning: Simulation resulted in NaN or Inf values.")
        # Handle this - e.g., return empty, remove bad rows, etc.

    return df