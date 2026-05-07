import numpy as np, os, sys, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

wf   = np.loadtxt('event1/wavefields/wavefield0002000_01_000.txt')
grid = np.loadtxt('event1/wavefields/wavefield_grid_for_dumps_000.txt')

print("=== WAVEFIELD FORMAT ===")
print(f"grid  shape: {grid.shape}  cols=[x, z]")
print(f"wavefield shape: {wf.shape}")
if wf.shape[1] == 2:
    print("  -> 2 cols = [ux, uz]  (coords in separate grid file) -- CORRECT")
elif wf.shape[1] == 4:
    print("  -> 4 cols = [x, z, ux, uz]  -- includes coords")
print(f"  ux: {wf[:,0].min():.3e} .. {wf[:,0].max():.3e}")
print(f"  uz: {wf[:,1].min():.3e} .. {wf[:,1].max():.3e}")
print(f"grid x: {grid[:,0].min():.1f} .. {grid[:,0].max():.1f}  (units: {'meters' if grid[:,0].max()>100 else 'km'})")
print(f"grid z: {grid[:,1].min():.1f} .. {grid[:,1].max():.1f}")
print()

print("=== SEISMOGRAM FORMAT ===")
sms = sorted([f for f in os.listdir('event1/seismograms') if 'BXZ' in f])
s = np.loadtxt(f'event1/seismograms/{sms[0]}')
print(f"shape: {s.shape}  file: {sms[0]}")
print(f"time:  {s[0,0]:.4f} .. {s[-1,0]:.4f} s")
peak_idx = np.argmax(np.abs(s[:,1]))
print(f"peak amplitude: {s[peak_idx,1]:.4e} m  at t={s[peak_idx,0]:.4f} s")
print()

print("=== ARRIVAL TIME CHECK ===")
# Source 1: x=400m, z=85m  |  Receiver 1: x=100m, z=415m
d1 = math.sqrt((400-100)**2 + (85-415)**2)
t_arr1 = 0.06 + d1/3000.0
print(f"Source1->Rcv1 distance: {d1:.1f} m")
print(f"Expected arrival (phys time): {t_arr1:.4f} s")
print(f"In PINN frame (t_st=0.10 s): {t_arr1-0.10:.4f} s")
print(f"Seismogram peak at (phys time): {s[peak_idx,0]:.4f} s")
delta = abs(s[peak_idx,0] - t_arr1)
print(f"Arrival match error: {delta:.4f} s  ({'OK' if delta < 0.05 else 'WARNING: mismatch!'})")
print()

print("=== IC SNAPSHOT TIMES ===")
# Check how many points are non-zero in IC1 (wave should have arrived by t=0.10s)
wf2 = np.loadtxt('event1/wavefields/wavefield0002000_01_000.txt')
nonzero = np.sum(np.abs(wf2[:,0]) > 1e-10)
total = wf2.shape[0]
print(f"IC1 (t=0.10s): nonzero ux points = {nonzero}/{total} ({100*nonzero/total:.1f}%)")
print(f"IC1 max|ux| = {np.abs(wf2[:,0]).max():.4e}")
print()

wf4 = np.loadtxt('event1/wavefields/wavefield0004000_01_000.txt')
nonzero4 = np.sum(np.abs(wf4[:,0]) > 1e-10)
print(f"IC2 (t=0.20s): nonzero ux points = {nonzero4}/{total} ({100*nonzero4/total:.1f}%)")
print(f"IC2 max|ux| = {np.abs(wf4[:,0]).max():.4e}")
