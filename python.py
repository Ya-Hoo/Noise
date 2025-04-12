import numpy as np
import matplotlib.pyplot as plt

# Constants and Initial Conditions
L = 20.0                  # Track length (m)
k_drag = 0.0005           # Tuned drag scaling factor (N·s²/m²)
k_lift = -0.0003          # Lift scaling factor (N·s²/m²) (negative = downforce)
Crr = 0.02                # Reduced rolling resistance coefficient (better bearings)
mveh = 0.050              # Vehicle mass (kg)
g = 9.81                  # Gravitational acceleration (m/s²)
Boost_time = 0.4          # Adjusted boost duration (s) → Shorter, higher thrust
initial_m_canister = 0.01  # Canister mass (kg)
dt = 0.0001               # Smaller time step for precision

# Thrust profile (peaks early, then decays)
def get_thrust(t):
    if t < 0.05:
        return 15.0        # Initial burst (N)
    elif t < Boost_time:
        return 12.0 * (1 - t / Boost_time)  # Decay phase
    else:
        return 0.0

# Initialize variables
time = 0.0
velocity = 0.0
distance = 0.0
current_m_canister = initial_m_canister

# Store results
time_history = []
velocity_history = []
distance_history = []

# Simulation loop
while distance < L and time < 2.0:  # Timeout at 2s
    Ft = get_thrust(time)
    
    # Update mass (linearly decreasing during boost)
    if time < Boost_time:
        current_m_canister = initial_m_canister * (1 - time / Boost_time)
    else:
        current_m_canister = 0.0
    total_mass = mveh + current_m_canister
    
    # Aerodynamic forces
    F_drag = k_drag * velocity**2
    F_lift = k_lift * velocity**2
    normal_force = total_mass * g + F_lift
    Fr = Crr * normal_force
    
    # Acceleration
    acceleration = (Ft - F_drag - Fr) / total_mass
    
    # Update velocity and distance
    velocity += acceleration * dt
    distance += velocity * dt
    time += dt
    
    # Store results
    time_history.append(time)
    velocity_history.append(velocity)
    distance_history.append(distance)

# Results
print(f"Final Time: {time:.3f} s")
print(f"Final Velocity: {velocity:.3f} m/s")
print(f"Distance: {distance:.3f} m")

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time_history, velocity_history, 'b-')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(time_history, distance_history, 'r-')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid()

plt.tight_layout()
plt.show()