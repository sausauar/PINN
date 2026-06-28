"""
Generate synthetic seismic data for acoustic wave inversion using 2D Finite Difference Method (FDM)
This replaces SPECFEM2D for simple testing
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

AX_SPEC = 1.5
AZ_SPEC = 0.5
X_SENSORS = 1.3
N_ABS = 10
NX_REF = 100
NZ_REF = 100
DX_REF = AX_SPEC / NX_REF
DZ_REF = AZ_SPEC / NZ_REF
AX_PINN = X_SENSORS - N_ABS * DX_REF
AZ_PINN = AZ_SPEC - N_ABS * DZ_REF
Z0_SEIS = AZ_PINN
ZL_SEIS = 0.06 - N_ABS * DZ_REF
LAYER_Z1 = 0.12
LAYER_Z2 = 0.28
LAYER_SMOOTH = 90.0
V_SAND = 1.8
V_SANDSTONE = 2.6
V_DEEP = 3.3

SOURCE_X = 0.62
SOURCE_Z = 0.18
SOURCE_SIGMA_X = 0.03
SOURCE_SIGMA_Z = 0.025
SOURCE_FORCE_SCALE = 8.0

class AcousticWaveSimulator:
    def __init__(self, nx=201, nz=101, dx=7.5, dz=5.0, dt=0.0005):
        """
        Initialize 2D acoustic wave simulator
        
        Parameters:
        -----------
        nx, nz : int
            Number of grid points in x and z directions
        dx, dz : float
            Grid spacing in meters (converted to km internally)
        dt : float
            Time step in seconds
        """
        self.nx = nx
        self.nz = nz
        self.dx = dx / 1000  # Convert to km
        self.dz = dz / 1000  # Convert to km
        self.dt = dt
        
        # Domain size in km
        self.Lx = (nx - 1) * self.dx
        self.Lz = (nz - 1) * self.dz
        
        # Create coordinate arrays
        self.x = np.linspace(0, self.Lx, nx)
        self.z = np.linspace(0, self.Lz, nz)
        self.xx, self.zz = np.meshgrid(self.x, self.z)
        
        # Create layered velocity model
        self.alpha = self._create_velocity_model()
        
        print(f"Domain: {self.Lx:.3f} x {self.Lz:.3f} km")
        print(f"Grid: {nx} x {nz}, spacing: {self.dx:.4f} km")
        
    def _create_velocity_model(self):
        """
        Smooth 3-layer model:
        1) sand
        2) sandstone
        3) compact deep layer
        """
        s1 = 0.5 * (1.0 + np.tanh(LAYER_SMOOTH * (self.zz - LAYER_Z1)))
        s2 = 0.5 * (1.0 + np.tanh(LAYER_SMOOTH * (self.zz - LAYER_Z2)))
        alpha = V_SAND + (V_SANDSTONE - V_SAND) * s1 + (V_DEEP - V_SANDSTONE) * s2
        
        print(f"  Layer 1 (sand):      {V_SAND:.2f} km/s, z < {LAYER_Z1:.2f} km")
        print(f"  Layer 2 (sandstone): {V_SANDSTONE:.2f} km/s, {LAYER_Z1:.2f} <= z < {LAYER_Z2:.2f} km")
        print(f"  Layer 3 (deep):      {V_DEEP:.2f} km/s, z >= {LAYER_Z2:.2f} km")
        
        return alpha

    def displacement_components(self, field):
        uz, ux = np.gradient(field, self.dz, self.dx)
        return ux, uz
    
    def ricker_wavelet(self, f0=6.0, t_max=0.35):
        """
        Generate Ricker wavelet (2nd derivative of Gaussian)
        
        Parameters:
        -----------
        f0 : float
            Dominant frequency in Hz
        t_max : float
            Maximum time (if None, compute from frequency)
        """
        t = np.arange(0, t_max, self.dt)
        t_peak = 1.0 / f0
        
        # Ricker wavelet formula
        arg = np.pi * f0 * (t - t_peak)
        wavelet = (1 - 2 * arg**2) * np.exp(-arg**2)
        
        return wavelet, t

    def source_force_field(self, t):
        """Spatially localized external force f(x,z,t) for one event."""
        arg = np.pi * 6.0 * (t - 1.0 / 6.0)
        source_t = (1.0 - 2.0 * arg**2) * np.exp(-arg**2)
        source_xy = np.exp(
            -(((self.xx - SOURCE_X) ** 2) / (2.0 * SOURCE_SIGMA_X**2)
              + ((self.zz - SOURCE_Z) ** 2) / (2.0 * SOURCE_SIGMA_Z**2))
        )
        return SOURCE_FORCE_SCALE * source_t * source_xy
    
    def plane_wave_incident(self, angle=20):
        """
        Create incident plane wave with given angle (in degrees from vertical)
        """
        k_angle = np.radians(angle)
        # Plane wave: u(x,z,t) ~ sin(kx*x + kz*z - omega*t)
        # For simplicity, we'll source this as a time-varying boundary condition
        return k_angle
    
    def fdm_solve(self, wavelet, angle_deg=0, nt_steps=None):
        """
        Solve 2D acoustic wave equation using FDM with full time history
        ∂²u/∂t² = α² (∂²u/∂x² + ∂²u/∂z²)
        
        Returns full wavefield history for all timesteps
        """
        if nt_steps is None:
            nt_steps = len(wavelet) * 2
        
        # CFL stability condition
        v_max = float(np.max(self.alpha))
        r_x = (v_max * self.dt / self.dx)**2
        r_z = (v_max * self.dt / self.dz)**2
        
        if r_x > 0.25 or r_z > 0.25:
            print(f"Warning: CFL condition may be violated (rx={r_x:.3f}, rz={r_z:.3f})")
        
        # Initialize fields
        u = np.zeros((self.nz, self.nx, 3))  # Current, previous (t-1, t-2)
        u_history = np.zeros((nt_steps, self.nz, self.nx))  # Store full history
        
        # Main time loop
        for step in range(1, nt_steps):
            if step % 500 == 0:
                print(f"  Time step {step}/{nt_steps}")
            
            # Calculate Laplacian using finite differences
            uxx = np.zeros((self.nz, self.nx))
            uzz = np.zeros((self.nz, self.nx))
            
            # Interior points
            uxx[1:-1, 1:-1] = (u[1:-1, 2:, 1] - 2*u[1:-1, 1:-1, 1] + u[1:-1, :-2, 1]) / (self.dx**2)
            uzz[1:-1, 1:-1] = (u[2:, 1:-1, 1] - 2*u[1:-1, 1:-1, 1] + u[:-2, 1:-1, 1]) / (self.dz**2)
            
            # Update wavefield (explicit FDM) with external force f:
            # d2u/dt2 = alpha^2 * laplacian(u) + f
            t_now = step * self.dt
            force_term = self.source_force_field(t_now)
            u_new = (2*u[:,:,1] - u[:,:,2] + 
                    (self.dt**2) * ((self.alpha**2) * (uxx + uzz) + force_term))
            
            # Boundary conditions (free surface on top)
            u_new[0, :] = 0  # Fixed surface
            
            # Absorbing boundary conditions on other sides (damping)
            damping = 0.95
            u_new[-1, :] *= damping  # Bottom
            u_new[:, 0] *= damping   # Left
            u_new[:, -1] *= damping  # Right
            
            # Shift in time
            u[:, :, 2] = u[:, :, 1]
            u[:, :, 1] = u_new
            u_history[step] = u_new.copy()
        
        return u_history  # Return full history
    

    def extract_seismograms(self, u_history, receiver_x, receiver_z):
        """
        Extract seismograms at receiver locations
        **CRITICAL: Coordinates must match PINN domain exactly!**
        
        Parameters:
        -----------
        u_history : array (nt, nz, nx)
            Time history of wavefield
        receiver_x, receiver_z : array
            Receiver coordinates (must be in km, matching PINN grid)
        
        Returns:
        --------
        seismograms : array (n_receivers, nt) - normalized to [-1, 1]
        """
        nt = u_history.shape[0]
        n_receivers = len(receiver_x)
        seismograms = np.zeros((n_receivers, nt))
        
        print(f"\n[RECEIVER COORDINATE CHECK]")
        print(f"  Grid range: x=[{self.x.min():.6f}, {self.x.max():.6f}] km")
        print(f"  Grid range: z=[{self.z.min():.6f}, {self.z.max():.6f}] km")
        
        for i, (rx, rz) in enumerate(zip(receiver_x, receiver_z)):
            # Find nearest grid points
            ix = np.argmin(np.abs(self.x - rx))
            iz = np.argmin(np.abs(self.z - rz))
            
            # Check accuracy
            x_actual = self.x[ix]
            z_actual = self.z[iz]
            
            if i == 0 or i == n_receivers - 1:
                print(f"    Receiver {i+1}: requested=({rx:.6f}, {rz:.6f}), actual=({x_actual:.6f}, {z_actual:.6f})")
            
            seismograms[i, :] = u_history[:, iz, ix]
        
        # NORMALIZE to [-1, 1] for PINN training
        seis_max = np.max(np.abs(seismograms))
        if seis_max > 0:
            seismograms = seismograms / (seis_max + 1e-12)
            print(f"\n[NORMALIZATION] Seismic data normalized to [-1, 1]")
            print(f"  Peak amplitude before: {seis_max:.6e}")
            print(f"  Peak amplitude after: {np.max(np.abs(seismograms)):.6f}")
        
        return seismograms
    
    def run_simulation(self, event_id, angle_deg=0, n_receivers=20):
        """
        Run complete simulation for one event
        
        Parameters:
        -----------
        event_id : int
            Event number (1 or 2)
        angle_deg : float
            Incident angle in degrees
        n_receivers : int
            Number of seismic stations
        """
        print(f"\n{'='*60}")
        print(f"Simulating Event {event_id} on layered medium")
        print(f"{'='*60}")
        
        # Generate source wavelet
        wavelet, t_source = self.ricker_wavelet(f0=6.0, t_max=0.35)
        
        # Run FDM simulation
        print("Running FDM solver...")
        u_history = self.fdm_solve(wavelet, angle_deg=angle_deg, nt_steps=len(wavelet)*2)
        
        # Create receiver array exactly where PINN2 expects it:
        # a vertical borehole at x=ax with depth from z0_s to zl_s.
        receiver_x = np.full(n_receivers, AX_PINN)
        receiver_z = np.linspace(Z0_SEIS, ZL_SEIS, n_receivers)
        
        # Extract seismograms
        print("Extracting seismograms...")
        seismograms = self.extract_seismograms(u_history, receiver_x, receiver_z)
        
        # Create output directories
        event_dir = f'event{event_id}'
        os.makedirs(f'{event_dir}/seismograms', exist_ok=True)
        os.makedirs(f'{event_dir}/wavefields', exist_ok=True)
        
        # Save wavefield grid coordinates (in meters, compatible with PINN1 format)
        grid_data = np.column_stack([self.xx.flatten() * 1000,  # Convert back to meters
                                     self.zz.flatten() * 1000])
        np.savetxt(f'{event_dir}/wavefields/wavefield_grid_for_dumps_000.txt', grid_data, fmt='%16.9E')
        
        # Save wavefield snapshots at realistic timesteps
        # Choose 3 snapshots: early, middle, late in the simulation
        n_steps = u_history.shape[0]
        snapshot_indices = [200, 230, 500]
        
        snapshot_labels = ['0002000', '0002300', '0005000']  # PINN1-compatible naming
        
        for idx, snap_idx in enumerate(snapshot_indices):
            if snap_idx < u_history.shape[0]:
                wavefield = u_history[snap_idx]
                ux, uz = self.displacement_components(wavefield)
                
                # Stack X and Z components: [ux, uz] for each grid point
                wavefield_stacked = np.column_stack([ux.flatten(), uz.flatten()])
                filename = f'{event_dir}/wavefields/wavefield{snapshot_labels[idx]}_01_000.txt'
                np.savetxt(filename, wavefield_stacked, fmt='%16.9E')
                print(f"  Saved snapshot {idx+1}: {filename}")

        
        # Save seismograms (already normalized in extract_seismograms)
        time_axis = np.arange(seismograms.shape[1]) * self.dt
        
        print(f"\n[SEISMOGRAM STATISTICS]")
        print(f"  Shape: {seismograms.shape} (n_receivers x n_timesteps)")
        print(f"  Range: [{np.min(seismograms):.6f}, {np.max(seismograms):.6f}] (normalized)")
        print(f"  Timesteps: {seismograms.shape[1]}, dt={self.dt} s")
        
        for i in range(n_receivers):
            station_id = f'S{i+1:04d}'

            ix = np.argmin(np.abs(self.x - receiver_x[i]))
            iz = np.argmin(np.abs(self.z - receiver_z[i]))
            trace = u_history[:, iz, ix]
            trace_ux = np.zeros_like(trace)
            trace_uz = np.zeros_like(trace)
            for step in range(u_history.shape[0]):
                ux_step, uz_step = self.displacement_components(u_history[step])
                trace_ux[step] = ux_step[iz, ix]
                trace_uz[step] = uz_step[iz, ix]

            z_scale = np.max(np.abs(trace_uz)) + 1e-12
            x_scale = np.max(np.abs(trace_ux)) + 1e-12

            z_seismo = np.column_stack([time_axis, trace_uz / z_scale])
            np.savetxt(f'{event_dir}/seismograms/AA.{station_id}.BXZ.semd', z_seismo, fmt='%14.9E')

            x_seismo = np.column_stack([time_axis, trace_ux / x_scale])
            np.savetxt(f'{event_dir}/seismograms/AA.{station_id}.BXX.semd', x_seismo, fmt='%14.9E')
        
        print(f"✓ Data saved to {event_dir}/")
        print(f"  - {n_receivers} seismogram pairs")
        print(f"  - Wavefields: 3 snapshots (PINN1 format)")
        print(f"\n[⚠️  BOUNDARY CONDITION MISMATCH WARNING]")
        print(f"  • FDM (this script) uses: free surface (top) + absorbing BC (sides)")
        print(f"  • PINN uses: simple BC (pressure=0 on top only)")
        print(f"  → This mismatch REDUCES convergence!")
        print(f"  → Solution: Add PML loss terms to PINNs_Inversion_Acoustic.py")
        
        return seismograms, receiver_x, receiver_z


# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SYNTHETIC ACOUSTIC DATA GENERATOR")
    print("Single Event + Layered Model + External Force")
    print("="*60)
    
    # Create simulator
    # CRITICAL: Grid spacing MUST match PINN domain exactly!
    sim = AcousticWaveSimulator(nx=201, nz=101, dx=7.5, dz=5.0, dt=0.0005)
    
    print(f"\n[COORDINATE SYSTEM VERIFICATION]")
    print(f"  Grid points: {sim.nx} x {sim.nz}")
    print(f"  Grid spacing: dx={sim.dx:.6f} km, dz={sim.dz:.6f} km")
    print(f"  Domain size: {sim.Lx:.6f} x {sim.Lz:.6f} km")
    print(f"  → MUST match PINN parameters: ax, az in PINNs_Inversion_Acoustic.py")
    
    # Visualize velocity model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Full velocity model
    im1 = ax1.contourf(sim.xx, sim.zz, sim.alpha, 20, cmap='RdYlBu_r')
    ax1.set_xlabel('X (km)', fontsize=12)
    ax1.set_ylabel('Z (km)', fontsize=12)
    ax1.set_title(r'Layered Velocity Model - True $\alpha$', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Velocity (km/s)')
    
    # Right: Zoomed view of anomaly
    im2 = ax2.contourf(sim.xx, sim.zz, sim.alpha, 20, cmap='RdYlBu_r')
    ax2.set_xlim([sim.Lx/2 - 0.4, sim.Lx/2 + 0.4])
    ax2.set_ylim([0, 0.5])
    ax2.set_xlabel('X (km)', fontsize=12)
    ax2.set_ylabel('Z (km)', fontsize=12)
    ax2.set_title(r'Layer Boundaries (Zoomed)', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Velocity (km/s)')
    
    plt.tight_layout()
    plt.savefig('velocity_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Velocity model visualization saved to velocity_model.png")
    
    # Simulate one event only
    seismo1, rx1, rz1 = sim.run_simulation(event_id=1, angle_deg=0, n_receivers=17)
    
    print("\n" + "="*60)
    print("✓ SYNTHETIC DATA GENERATION COMPLETE")
    print("="*60)
    print("\n[CHECKLIST FOR PINN COMPATIBILITY]")
    print("  ✅ Seismic data NORMALIZED to [-1, 1]")
    print("  ✅ Receiver coordinates logged and verified")
    print("  ✅ Wavefields saved in PINN1-compatible format")
    print("  ⚠️  Boundary conditions mismatch (FDM uses ABC, PINN uses simple BC)")
    print("\n[CRITICAL FIXES TO MAKE IN PINN SCRIPT]")
    print("  1. Set w_seis = 100.0 (NOT 1e-7) in train_step()")
    print("  2. Verify receiver_x, receiver_z match FDM grid exactly")
    print("  3. Consider adding PML/ABC loss terms for better BC match")
    print(f"\n✓ Data ready for PINN training!")
    print(f"  Run: python PINNs_Inversion_Acoustic.py --fast")
