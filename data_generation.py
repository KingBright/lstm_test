# data_generation.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import config # Import config for parameters like SEED, DT, T_SPAN_LONG etc.

# --- Global Torque Handling ---
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
    # Calculate index, ensuring it's within bounds
    idx = int((time - _global_t_span_start) / _global_dt)
    idx = max(0, min(idx, len(_global_tau_values) - 1))
    return _global_tau_values[idx]

# --- Pendulum System Definition ---
class PendulumSystem:
    """Represents the damped pendulum system with external torque."""
    def __init__(self, m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF):
        self.m = m
        self.L = L
        self.g = g
        self.c = c
        if self.m <= 0 or self.L <= 0:
            raise ValueError("Mass (m) and Length (L) must be positive.")
        # Pre-calculate constants for the ODE
        self.beta = self.c / (self.m * self.L**2) # Damping term coefficient
        self.omega0_sq = self.g / self.L          # Natural frequency squared
        # print(f"PendulumSystem initialized: m={m}, L={L}, g={g}, c={c} -> beta={self.beta:.3f}, omega0_sq={self.omega0_sq:.3f}")

    def ode(self, t, x, current_tau=None):
        """The ordinary differential equation for the pendulum system."""
        theta, theta_dot = x
        # Use provided torque if available, otherwise lookup from global sequence
        tau = current_tau if current_tau is not None else get_tau_at_time_global(t)
        dtheta_dt = theta_dot
        # Pendulum equation: theta'' = -beta*theta' - omega0^2*sin(theta) + tau / (m*L^2)
        dtheta_dot_dt = (-self.beta * theta_dot
                         - self.omega0_sq * np.sin(theta)
                         + tau / (self.m * self.L**2))
        return np.array([dtheta_dt, dtheta_dot_dt])

# --- Enhanced Torque Sequence Generation ---
def generate_torque_sequence(t, type="highly_random", torque_change_steps=None):
    """
    Generates different types of torque input sequences.

    Args:
        t (np.array): Array of time points.
        type (str): Torque type ("zero", "step", "sine", "ramp", "highly_random").
        torque_change_steps (int, optional): Steps for torque change in 'highly_random'.
                                             Defaults to None, using config range.

    Returns:
        np.array: Generated torque sequence, clipped to config.TORQUE_RANGE.
    """
    t = np.asarray(t)
    if t.ndim == 0: t = np.array([t]) # Ensure t is an array
    num_steps = len(t)
    if num_steps == 0: return np.array([])
    dt = t[1] - t[0] if num_steps > 1 else config.DT # Estimate dt
    max_time = t[-1] if num_steps > 0 else 0
    torque_min, torque_max = config.TORQUE_RANGE

    tau = np.zeros_like(t) # Initialize torque array

    if type == "zero":
        pass # Already zeros
    elif type == "step":
        # Generate 1 to 3 random steps
        num_steps_change = np.random.randint(1, 4)
        step_times = np.sort(np.random.uniform(t[0] + dt, max_time - dt, num_steps_change))
        last_time = t[0]
        last_val = 0.0 # Start at zero torque
        for step_time in step_times:
            # Assign value for the segment before the step
            tau[(t >= last_time) & (t < step_time)] = last_val
            # Choose a new random value for the next segment
            last_val = np.random.uniform(torque_min, torque_max)
            last_time = step_time
        # Assign value for the last segment
        tau[t >= last_time] = last_val
    elif type == "sine":
        # Random amplitude and frequency
        # Ensure amplitude doesn't exceed half the range for centering
        max_amp = (torque_max - torque_min) / 2.5 # Slightly less than half range
        amplitude = np.random.uniform(max_amp * 0.1, max_amp)
        # Frequency in Hz (cycles per second)
        min_freq = 0.1 / (max_time if max_time > 0 else 1) # At least 0.1 cycle
        max_freq = 5.0 / (max_time if max_time > 0 else 1) # Up to 5 cycles
        frequency = np.random.uniform(min_freq, max_freq)
        phase = np.random.uniform(0, 2 * np.pi)
        # Center the sine wave within the torque range
        center = np.random.uniform(torque_min + amplitude, torque_max - amplitude)
        tau = center + amplitude * np.sin(2 * np.pi * frequency * t + phase)
    elif type == "ramp":
        # Linear ramp between two random points within the range
        start_val = np.random.uniform(torque_min, torque_max)
        end_val = np.random.uniform(torque_min, torque_max)
        tau = np.linspace(start_val, end_val, num_steps)
    elif type == "highly_random": # Original piecewise constant random
        # Determine steps for change
        if torque_change_steps is None:
            if hasattr(config, 'TORQUE_CHANGE_STEPS_RANGE'):
                change_steps = np.random.randint(
                    config.TORQUE_CHANGE_STEPS_RANGE[0],
                    config.TORQUE_CHANGE_STEPS_RANGE[1] + 1
                )
            else:
                change_steps = getattr(config, 'TORQUE_CHANGE_STEPS', 20) # Fallback
        else:
            change_steps = torque_change_steps
        change_steps = max(1, int(change_steps)) # Ensure positive integer

        num_segments = (num_steps + change_steps - 1) // change_steps
        # Sample segment values uniformly within the range
        segment_values = np.random.uniform(torque_min, torque_max, num_segments)
        tau = np.repeat(segment_values, change_steps)[:num_steps] # Ensure correct length
    else:
        print(f"Warning: Unknown torque type '{type}'. Using zero torque.")
        # Already zeros

    # Clip the final torque sequence to the allowed range
    tau = np.clip(tau, torque_min, torque_max)
    return tau


# --- Simulation Execution (using solve_ivp) ---
def run_simulation(pendulum, t_span, dt, x0, tau_values, t_eval=None):
    """Runs the ODE simulation using solve_ivp."""
    if t_eval is None:
        t_eval = np.arange(t_span[0], t_span[1] + dt/2, dt) # Include endpoint approx
    if not isinstance(t_eval, np.ndarray) or t_eval.ndim != 1 or len(t_eval) == 0:
        print("Warning: Invalid t_eval in run_simulation.")
        return np.array([]), np.array([]), np.array([])

    # Ensure tau_values match t_eval length if possible, pad if necessary
    if len(tau_values) < len(t_eval):
        print(f"Warning: tau_values length ({len(tau_values)}) < t_eval length ({len(t_eval)}). Padding with last value.")
        padding = np.full(len(t_eval) - len(tau_values), tau_values[-1] if len(tau_values)>0 else 0)
        tau_values = np.concatenate((tau_values, padding))
    elif len(tau_values) > len(t_eval):
         print(f"Warning: tau_values length ({len(tau_values)}) > t_eval length ({len(t_eval)}). Truncating.")
         tau_values = tau_values[:len(t_eval)]


    effective_t_span = (t_eval[0], t_eval[-1])
    set_global_torque_params(tau_values, t_span[0], dt) # Set for potential ODE lookup

    try:
        sol = solve_ivp(
            fun=lambda t, x: pendulum.ode(t, x, current_tau=None), # ODE func
            t_span=effective_t_span,    # Time interval
            y0=x0,                      # Initial state
            method='RK45',              # Integrator method
            t_eval=t_eval,              # Times to store solution at
            dense_output=False          # Don't need dense output
            # rtol=1e-5, atol=1e-8      # Optional: Adjust tolerances
        )
        set_global_torque_params(None, 0, config.DT) # Reset globals after solve

        if sol.status != 0:
            print(f"Warning: ODE solver finished with status {sol.status}: {sol.message}")
            # Return empty arrays on solver failure, but maybe allow status 1 (root found)?
            # if sol.status < 0: # Only return empty for definite errors
            return np.array([]), np.array([]), np.array([])

        # Check if output length matches t_eval
        if len(sol.t) != len(t_eval):
            print(f"Warning: Solver output length ({len(sol.t)}) mismatch with t_eval length ({len(t_eval)}).")
            # Attempt to interpolate if lengths differ significantly? Or return empty?
            # For now, return what we got, but be aware.
            # Let's return empty if mismatch is drastic
            if abs(len(sol.t) - len(t_eval)) > 1:
                 print("  Mismatch too large, returning empty.")
                 return np.array([]), np.array([]), np.array([])

        # Return time, theta, theta_dot
        return sol.t, sol.y[0], sol.y[1]

    except Exception as e:
        print(f"Error during ODE solving: {e}")
        set_global_torque_params(None, 0, config.DT) # Ensure reset on error
        return np.array([]), np.array([]), np.array([])


# --- Original Simulation Data Generation (for single trajectories) ---
def generate_simulation_data(pendulum, t_span=config.T_SPAN, dt=config.DT, x0=None,
                            torque_type="highly_random", torque_change_steps=None):
    """
    Generates simulation data for a single trajectory with specified torque type.
    Used for generating evaluation segments.

    Args:
        pendulum: PendulumSystem instance.
        t_span: Simulation time span (start, end).
        dt: Time step.
        x0: Initial state [theta, theta_dot]. If None, random values are used.
        torque_type: Type of torque sequence to generate.
        torque_change_steps: Steps for torque change (used by 'highly_random').

    Returns:
        pd.DataFrame: DataFrame containing simulation data (time, theta, theta_dot, tau).
    """
    # Handle initial conditions
    if x0 is None:
        x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]
    elif len(x0) != 2:
        print("Warning: Invalid x0 provided to generate_simulation_data, using random.")
        x0 = [np.random.uniform(*config.THETA_RANGE), np.random.uniform(*config.THETA_DOT_RANGE)]

    # Generate time vector
    t_full = np.arange(t_span[0], t_span[1] + dt/2, dt)
    if len(t_full) == 0:
        return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    # Generate the torque sequence for this simulation
    tau_values = generate_torque_sequence(t_full, type=torque_type, torque_change_steps=torque_change_steps)

    # Run the simulation
    time_points, theta_values, theta_dot_values = run_simulation(
        pendulum, t_span, dt, x0, tau_values, t_eval=t_full
    )

    if len(time_points) == 0:
        return pd.DataFrame({'time': [], 'theta': [], 'theta_dot': [], 'tau': []})

    # Align torque values with actual output times from the solver
    set_global_torque_params(tau_values, t_span[0], dt)
    tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]
    set_global_torque_params(None, 0, config.DT)

    # Create DataFrame
    data = {
        'time': time_points,
        'theta': theta_values,
        'theta_dot': theta_dot_values,
        'tau': tau_at_output_times
    }
    df = pd.DataFrame(data)

    # Check data quality
    if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any():
        print("Warning: Simulation resulted in NaN or Inf values.")

    return df

# --- RK4 Step Function (can be used for alternative integration) ---
def rk4_step(pendulum, state_t, tau_t, dt):
    """ Performs a single RK4 integration step. """
    t_dummy = 0 # Time argument is not used in this specific ODE formulation
    try:
        # Calculate RK4 terms
        k1 = pendulum.ode(t_dummy, state_t,       current_tau=tau_t)
        k2 = pendulum.ode(t_dummy, state_t + 0.5*dt*k1, current_tau=tau_t)
        k3 = pendulum.ode(t_dummy, state_t + 0.5*dt*k2, current_tau=tau_t)
        k4 = pendulum.ode(t_dummy, state_t + dt*k3,     current_tau=tau_t)
        # Combine terms for the next state
        state_t_plus_dt = state_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Check for invalid results
        if not np.all(np.isfinite(state_t_plus_dt)):
            print(f"Warning: NaN/Inf detected during RK4 step.")
            return np.array([np.nan, np.nan]) # Return NaN state
        return state_t_plus_dt
    except Exception as e:
        print(f"Error during RK4 step calculation: {e}")
        return np.array([np.nan, np.nan]) # Return NaN state on error

