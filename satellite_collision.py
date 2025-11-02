"""
satellite_collision.py

Combined satellite solar-energy simulation + simple collision prediction & avoidance.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------
# Configuration / constants
# --------------------------
MU_EARTH = 398600.4418  # km^3 / s^2
SOLAR_CONSTANT = 1361  # W/m^2
PANEL_AREA = 10        # m^2
EFFICIENCY = 0.3       # 30% efficiency
TRANSMISSION_EFF = 0.8 # 80% transmission
EARTH_RADIUS = 6371    # km

np.random.seed(1)

# Satellite parameters (3 satellites)
num_satellites = 3
orbital_radii = np.array([EARTH_RADIUS + 500, EARTH_RADIUS + 700, EARTH_RADIUS + 900])  # km

# Debris parameters
num_debris = 6
debris_radii = np.random.uniform(EARTH_RADIUS + 400, EARTH_RADIUS + 1000, size=num_debris)

# Prediction configuration
prediction_hours = 6.0
dt_minutes = 1.0
time_steps = int(prediction_hours * 60 / dt_minutes) + 1
time = np.linspace(0, prediction_hours * 60, time_steps)  # minutes

# Safety / uncertainty model
pos_uncertainty_sigma_km = 1.0
collision_radius_km = 0.1
prob_threshold = 0.001

# --------------------------
# Helper orbital functions
# --------------------------
def circular_velocity_km_per_s(radius_km):
    return np.sqrt(MU_EARTH / radius_km)

def orbital_period_minutes(radius_km):
    T_seconds = 2 * np.pi * np.sqrt(radius_km**3 / MU_EARTH)
    return T_seconds / 60.0

def mean_motion_rad_per_min(radius_km):
    T_min = orbital_period_minutes(radius_km)
    return 2 * np.pi / T_min

def propagate_circular(radius_km, initial_phase_rad, t_minutes):
    n = mean_motion_rad_per_min(radius_km)  # rad per minute
    phases = initial_phase_rad + n * t_minutes
    x = radius_km * np.cos(phases)
    y = radius_km * np.sin(phases)
    return x, y

# --------------------------
# Initialize satellites & debris
# --------------------------
sat_initial_phases = np.linspace(0, 2*np.pi, num_satellites, endpoint=False)
sat_radius = orbital_radii.copy()
debris_initial_phases = np.random.uniform(0, 2*np.pi, size=num_debris)

sat_x = np.zeros((num_satellites, time_steps))
sat_y = np.zeros((num_satellites, time_steps))
sat_power = np.zeros((num_satellites, time_steps))
sat_accum_energy = np.zeros((num_satellites, time_steps))  # Wh

deb_x = np.zeros((num_debris, time_steps))
deb_y = np.zeros((num_debris, time_steps))

for i in range(num_satellites):
    sx, sy = propagate_circular(sat_radius[i], sat_initial_phases[i], time)
    sat_x[i, :] = sx
    sat_y[i, :] = sy
    sunlight_factor = np.where(sx < 0, 0.2, 1.0)
    orientation_factor = 0.85 + 0.15 * np.sin(2 * np.pi * time / orbital_period_minutes(sat_radius[i]))
    sat_power[i, :] = SOLAR_CONSTANT * PANEL_AREA * EFFICIENCY * sunlight_factor * orientation_factor
    sat_accum_energy[i, :] = np.cumsum(sat_power[i, :]) * (time[1] - time[0]) / 60.0 * TRANSMISSION_EFF

for j in range(num_debris):
    dx, dy = propagate_circular(debris_radii[j], debris_initial_phases[j], time)
    deb_x[j, :] = dx
    deb_y[j, :] = dy

# --------------------------
# Collision prediction
# --------------------------
def collision_probability(min_dist_km, sigma_km, coll_radius_km):
    if min_dist_km <= coll_radius_km:
        return 0.9999
    delta = min_dist_km - coll_radius_km
    p = np.exp(- (delta**2) / (2 * sigma_km**2))
    return float(np.clip(p, 0.0, 0.9999))

alerts = []
for i in range(num_satellites):
    for j in range(num_debris):
        sep = np.sqrt((sat_x[i, :] - deb_x[j, :])**2 + (sat_y[i, :] - deb_y[j, :])**2)
        idx_min = int(np.argmin(sep))
        dmin = float(sep[idx_min])
        t_ca = float(time[idx_min])
        prob = collision_probability(dmin, pos_uncertainty_sigma_km, collision_radius_km)
        if prob > prob_threshold:
            alerts.append({
                'sat_index': i,
                'deb_index': j,
                'dmin_km': dmin,
                't_ca_min': t_ca,
                'prob': prob,
                'idx_time': idx_min
            })

# --------------------------
# Avoidance planner
# --------------------------
def plan_avoidance_delta_r(sat_radius_km, sat_phase_rad, t_ca_min, debris_radius_km, debris_phase_rad, safe_separation_km=1.0):
    sx0, sy0 = propagate_circular(sat_radius_km, sat_phase_rad, np.array([t_ca_min]))
    dx0, dy0 = propagate_circular(debris_radius_km, debris_phase_rad, np.array([t_ca_min]))
    baseline_sep = float(np.sqrt((sx0[0]-dx0[0])**2 + (sy0[0]-dy0[0])**2))

    dr_candidates = np.linspace(-5.0, 5.0, 401)
    best = None
    for dr in dr_candidates:
        new_r = sat_radius_km + dr
        n_new = mean_motion_rad_per_min(new_r)
        new_phase = sat_phase_rad + n_new * t_ca_min
        sx, sy = new_r * np.cos(new_phase), new_r * np.sin(new_phase)
        dx, dy = debris_radius_km * np.cos(debris_phase_rad + mean_motion_rad_per_min(debris_radius_km)*t_ca_min), \
                 debris_radius_km * np.sin(debris_phase_rad + mean_motion_rad_per_min(debris_radius_km)*t_ca_min)
        sep = np.sqrt((sx - dx)**2 + (sy - dy)**2)
        if sep >= safe_separation_km:
            v_old = circular_velocity_km_per_s(sat_radius_km)
            v_new = circular_velocity_km_per_s(new_r)
            dv_km_s = abs(v_new - v_old)
            dv_m_s = dv_km_s * 1000.0
            best = {'delta_r_km': float(dr), 'sep_km': float(sep), 'dv_m_s': float(dv_m_s)}
            break
    return best, baseline_sep

for a in alerts:
    i = a['sat_index']
    j = a['deb_index']
    plan, baseline_sep = plan_avoidance_delta_r(
        sat_radius_km = sat_radius[i],
        sat_phase_rad = sat_initial_phases[i],
        t_ca_min = a['t_ca_min'],
        debris_radius_km = debris_radii[j],
        debris_phase_rad = debris_initial_phases[j],
        safe_separation_km = max(1.0, a['dmin_km'] + 0.5)
    )
    a['baseline_sep_km'] = baseline_sep
    a['avoid_plan'] = plan

# Apply first avoidance (demo)
if len(alerts) > 0:
    primary = alerts[0]
    sat_to_manoeuvre = primary['sat_index']
    plan = primary['avoid_plan']
    print("ALERT: potential conjunction detected:")
    print(primary)
    if plan is not None:
        print(f"Planned avoidance for sat {sat_to_manoeuvre+1}: delta_r = {plan['delta_r_km']:.3f} km, "
              f"approx delta-v = {plan['dv_m_s']:.3f} m/s, predicted sep after = {plan['sep_km']:.3f} km")
        sat_radius[sat_to_manoeuvre] += plan['delta_r_km']
        sx, sy = propagate_circular(sat_radius[sat_to_manoeuvre], sat_initial_phases[sat_to_manoeuvre], time)
        sat_x[sat_to_manoeuvre, :] = sx
        sat_y[sat_to_manoeuvre, :] = sy
        sunlight_factor = np.where(sx < 0, 0.2, 1.0)
        orientation_factor = 0.85 + 0.15 * np.sin(2 * np.pi * time / orbital_period_minutes(sat_radius[sat_to_manoeuvre]))
        sat_power[sat_to_manoeuvre, :] = SOLAR_CONSTANT * PANEL_AREA * EFFICIENCY * sunlight_factor * orientation_factor
        sat_accum_energy[sat_to_manoeuvre, :] = np.cumsum(sat_power[sat_to_manoeuvre, :]) * (time[1] - time[0]) / 60.0 * TRANSMISSION_EFF
    else:
        print(f"No feasible avoidance found within search bounds for sat {sat_to_manoeuvre+1}.")
else:
    print("No conjunction alerts above probability threshold found in prediction window.")

# --------------------------
# Output summary
# --------------------------
print("\n--- Summary ---")
for i in range(num_satellites):
    print(f"Satellite {i+1}: final orbital radius = {sat_radius[i]:.3f} km, total transmitted energy (approx) = {sat_accum_energy[i,-1]:.1f} Wh")

if len(alerts) > 0:
    print("\nAlerts (first few):")
    for a in alerts[:5]:
        print(f"Sat {a['sat_index']+1} vs Deb {a['deb_index']+1}: dmin={a['dmin_km']:.3f} km at t+{a['t_ca_min']:.1f} min, P={a['prob']:.4f}, plan={a['avoid_plan']}")

# --------------------------
# Visualization (Plotly)
# --------------------------
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'scene'}, {'type':'xy'}]], column_widths=[0.6, 0.4])

# Earth marker (3D) — add to scene (row=1,col=1)
earth_trace = go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=30, color='blue'), name='Earth')
fig.add_trace(earth_trace, row=1, col=1)

# Draw orbits as lines and final satellite positions (3D traces added to scene via row/col in add_trace)
theta_full = np.linspace(0, 2*np.pi, 200)
for i in range(num_satellites):
    orbit_x = sat_radius[i] * np.cos(theta_full)
    orbit_y = sat_radius[i] * np.sin(theta_full)
    orbit_trace = go.Scatter3d(x=orbit_x, y=orbit_y, z=np.zeros_like(orbit_x), mode='lines', name=f'Sat {i+1} orbit')
    fig.add_trace(orbit_trace, row=1, col=1)

    final_trace = go.Scatter3d(x=[sat_x[i,-1]], y=[sat_y[i,-1]], z=[0], mode='markers+text',
                               marker=dict(size=6), text=[f"Sat {i+1}"], textposition='top center',
                               name=f'Sat {i+1} final')
    fig.add_trace(final_trace, row=1, col=1)

for j in range(num_debris):
    orbit_x = debris_radii[j] * np.cos(theta_full)
    orbit_y = debris_radii[j] * np.sin(theta_full)
    deb_orbit = go.Scatter3d(x=orbit_x, y=orbit_y, z=np.zeros_like(orbit_x), mode='lines', name=f'Deb {j+1}', opacity=0.4)
    fig.add_trace(deb_orbit, row=1, col=1)

    deb_final = go.Scatter3d(x=[deb_x[j,-1]], y=[deb_y[j,-1]], z=[0], mode='markers+text', marker=dict(size=4), text=[f"D{j+1}"], textposition='bottom center', name=f'Deb {j+1} final')
    fig.add_trace(deb_final, row=1, col=1)

# If alerts exist, mark predicted CA (3D)
for a in alerts:
    idx = a['idx_time']
    sx_ca = sat_x[a['sat_index'], idx]; sy_ca = sat_y[a['sat_index'], idx]
    dx_ca = deb_x[a['deb_index'], idx]; dy_ca = deb_y[a['deb_index'], idx]
    ca_trace = go.Scatter3d(x=[sx_ca, dx_ca], y=[sy_ca, dy_ca], z=[0,0], mode='markers+lines',
                            marker=dict(size=[6,4]), name=f'CA Sat{a["sat_index"]+1}-D{a["deb_index"]+1}',
                            hovertext=[f"Sat{a['sat_index']+1} at t+{a['t_ca_min']:.1f}m", f"Deb{a['deb_index']+1}"])
    fig.add_trace(ca_trace, row=1, col=1)

# Power vs time on right subplot (2D)
for i in range(num_satellites):
    fig.add_trace(go.Scatter(x=time, y=sat_power[i,:], mode='lines', name=f'Sat{i+1} Power (W)'), row=1, col=2)

fig.update_layout(title="Satellite Solar + Collision Prediction & Avoidance (demo)",
                scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)', aspectmode='data'),
                height=700)

import plotly.io as pio
pio.renderers.default = 'browser'
fig.write_html("orbit_output.html")
import webbrowser
webbrowser.open("orbit_output.html")

