# data_generation.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config # Import config for parameters like SEED, DT

# --- Global Torque Handling ---
# Using globals is simpler for solve_ivp callback in this context,
# but ensure set_global_torque_params is called before run_simulation.
_global_tau_values = None
_global_t_span_start = 0
_global_dt = config.DT # Use DT from config as default

def set_global_torque_params(tau_values, t_span_start, dt):
    """Sets global parameters for the torque function used by the ODE solver."""
    global _global_tau_values, _global_t_span_start, _global_dt
    if tau_values is not None:
        _global_tau_values = tau_values
        _global_t_span_start = t_span_start
        _global_dt = dt if dt > 0 else config.DT # Ensure dt is positive
    else:
        _global_tau_values = None
        # print("Warning: Attempted to set None torque values globally.")

def get_tau_at_time_global(time):
    """
    Retrieves the torque value at a specific time using global parameters.
    NOTE: Relies on set_global_torque_params being called beforehand.
    """
    if _global_tau_values is None or _global_dt <= 0 or len(_global_tau_values) == 0:
        # print(f"Warning: Global torque parameters not set or invalid. Returning 0 torque.")
        return 0.0

    # Calculate index, ensuring it's within bounds [0, len-1]
    idx = int((time - _global_t_span_start) / _global_dt)
    idx = max(0, min(idx, len(_global_tau_values) - 1))
    return _global_tau_values[idx]


# --- Pendulum System Definition ---
class PendulumSystem:
    """Represents the damped, driven pendulum system."""
    def __init__(self, m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                 g=config.GRAVITY, c=config.DAMPING_COEFF):
        self.m = m
        self.L = L
        self.g = g
        self.c = c
        # Derived parameters
        if self.m <= 0 or self.L <= 0:
            raise ValueError("Mass (m) and Length (L) must be positive.")
        self.beta = self.c / (self.m * self.L**2)
        self.omega0_sq = self.g / self.L
        print(f"PendulumSystem initialized: m={m}, L={L}, g={g}, c={c} -> beta={self.beta:.3f}, omega0_sq={self.omega0_sq:.3f}")

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
    t = np.asarray(t)
    if t.ndim == 0: t = np.array([t])
    if len(t) == 0: return np.array([])

    max_time = t.max() if len(t) > 0 else 0
    num_steps = len(t)

    if type == "zero":
        return np.zeros_like(t)
    elif type == "step":
        tau = np.zeros_like(t)
        step_time = max_time * 0.3
        tau[t >= step_time] = 1.0 # Constant torque after step time
        return tau
    elif type == "sine":
        amplitude = 0.5
        frequency = 0.5 # Hz
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif type == "random":
        np.random.seed(config.SEED) # Use seed from config
        segment_length = 20 # Time steps per segment
        if segment_length <= 0: segment_length = 1
        num_segments = (num_steps + segment_length - 1) // segment_length
        segments = np.random.uniform(-0.8, 0.8, num_segments)
        repeated_segments = np.repeat(segments, segment_length)
        return repeated_segments[:num_steps] # Trim to exact length
    elif type == "mixed":
        tau = np.zeros_like(t)
        quarter = num_steps // 4
        idx_q1, idx_q2, idx_q3 = quarter, 2 * quarter, 3 * quarter

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
def run_simulation(pendulum, t_span, dt, x0, tau_values, t_eval=None):
    """
    Executes the simulation using solve_ivp for a given pendulum instance.
    Uses provided t_eval if available, otherwise generates from t_span, dt.
    """
    if t_eval is None:
         t_eval = np.arange(t_span[0], t_span[1], dt)

    if not isinstance(t_eval, np.ndarray) or t_eval.ndim != 1 or len(t_eval) == 0:
         print("Warning: t_eval is invalid or empty in run_simulation.")
         return np.array([]), np.array([]), np.array([])

    # Ensure t_span correctly covers the t_eval range for solve_ivp
    effective_t_span = (t_eval[0], t_eval[-1])

    # Set global torque parameters for the ODE function
    # Pass the original dt used for generating tau_values
    set_global_torque_params(tau_values, t_span[0], dt)

    try:
        sol = solve_ivp(
            pendulum.ode,
            effective_t_span, # Use span covering t_eval
            x0,
            method='RK45',
            t_eval=t_eval, # Use the specified evaluation times
            dense_output=False,
            # rtol=1e-6, atol=1e-9 # Optional stricter tolerances
        )
        # Reset global torque parameters after use
        set_global_torque_params(None, 0, config.DT)

        if sol.status != 0:
             print(f"Warning: ODE solver finished with status {sol.status}: {sol.message}")
             # Return empty arrays on solver failure
             return np.array([]), np.array([]), np.array([])

        # Check if solver output length matches t_eval length
        if len(sol.t) != len(t_eval):
             print(f"Warning: Solver output length ({len(sol.t)}) does not match t_eval length ({len(t_eval)}).")
             # This might indicate partial integration. Handle carefully.
             # For now, return what was solved, but caller should be aware.
             # Or return empty if mismatch is critical? Returning partial for now.
             pass

        return sol.t, sol.y[0], sol.y[1] # time, theta, theta_dot

    except Exception as e:
        print(f"Error during ODE solving: {e}")
        # Reset global torque parameters in case of error
        set_global_torque_params(None, 0, config.DT)
        return np.array([]), np.array([]), np.array([])


def generate_simulation_data(pendulum, t_span=config.T_SPAN_TRAIN, dt=config.DT, x0=None, torque_type="mixed"):
    """
    Generates the simulation data DataFrame by creating torque and running the simulation.
    """
    if x0 is None: x0 = [0.1, 0.0] # Default initial condition

    # Generate time vector for torque generation based on full span
    t_full = np.arange(t_span[0], t_span[1], dt)
    if len(t_full) == 0:
        print(f"Warning: Simulation time span resulted in no time steps: {t_span}, dt={dt}")
        return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    tau_values = generate_torque_sequence(t_full, type=torque_type)

    # Run the simulation - t_eval will be generated inside run_simulation if not provided
    # Or pass t_full as t_eval if we want results at exactly those points
    time_points, theta_values, theta_dot_values = run_simulation(pendulum, t_span, dt, x0, tau_values, t_eval=t_full)

    if len(time_points) == 0:
        # print(f"Warning: Simulation returned no results for t_span={t_span}, x0={x0}") # run_simulation prints warnings
        return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    # Re-generate/align torque values with the actual output times from solve_ivp
    # This might be redundant if t_eval=t_full was used and solve_ivp succeeded fully
    # For safety, let's realign using the returned time_points
    set_global_torque_params(tau_values, t_span[0], dt) # Ensure globals are set based on original tau generation
    tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]
    set_global_torque_params(None, 0, config.DT) # Reset globals

    # Create DataFrame
    data = {
        'time': time_points,
        'theta': theta_values,
        'theta_dot': theta_dot_values,
        'tau': tau_at_output_times
    }
    df = pd.DataFrame(data)

    # Check for NaNs or Infs
    if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any():
        print("Warning: Simulation resulted in NaN or Inf values. Check parameters or solver stability.")
        # Optional: clean rows with NaN/Inf, or return empty/raise error
        # df = df.replace([np.inf, -np.inf], np.nan).dropna() # Example cleaning

    return df

