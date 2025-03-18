# FlatWheels

## Data Description

The CSV file contains acceleration data from a SIMPACK simulation for testing frequency analysis. The simulation involves a wheel flat with a length of 30 mm at a constant speed of 20 km/h. The sampling frequency for the recorded results is 1 kHz. Additionally, the same scenario has been simulated without a wheel flat, considering only the inherent track irregularities.

### CSV Channels:
1. **Channel 1**: Time [s]
2. **Channel 2**: Vertical acceleration in the axle box (without wheel flat) [m/s²]
3. **Channel 3**: Vertical acceleration in the axle box (with wheel flat) [m/s²]

### Important Notes:
- The result logging does not start at t=0 to exclude the initial transient phase, allowing the vehicle to reach a quasi-static equilibrium.
- Recorded accelerations may include numerical artifacts from the integration method.
- The accelerations in the CSV are raw outputs from SIMPACK, without post-processing.

### File Naming:
- `TI_High` indicates high track irregularities considered in the simulation.
- `L` refers to the length of the flat in mm.
- `VEL` refers to the speed in km/h.
- `fs` refers to the sampling frequency in Hz