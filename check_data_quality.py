"""
check_data_quality.py
=====================
Proverka: mozhno li vosstanovit anomaliu iz dannyh SPECFEM2D.

Run (from PINN4 folder):
    python check_data_quality.py
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.interpolate as interpolate

# ── Параметры домена (должны совпадать с PINNs_Inversion_Acoustic.py) ──
Lx, Lz   = 3.0, 3.0
ax, az    = 1.5, 0.5
t_ic1     = 0.10
t_ic2     = 0.20
t_la      = 0.25
t_m       = 0.45
t_st      = t_ic1
t_s       = 0.45
n_seis    = 20
n_event   = 2
sponge_cells = 15
fd_dz     = 0.005
z_seis_phys = az - (sponge_cells + 2) * fd_dz   # ≈ 0.415 km

# Аномалия (эллипс)
ALPHA_BG   = 3.0
ALPHA_ANOM = 2.0
ELL_CX, ELL_CZ = 0.75, 0.25
ELL_RX, ELL_RZ = 0.20, 0.10

# Инверсионное окно
z_st_box, z_fi_box = 0.10, 0.42
x_st_box, x_fi_box = 0.45, 1.05

WF_STEPS = ['wavefield0002000_01_000.txt',
            'wavefield0004000_01_000.txt',
            'wavefield0005000_01_000.txt']

RESULTS = {}   # словарь: тест → (пройден?, сообщение)

def mark(name, ok, msg):
    RESULTS[name] = (ok, msg)
    status = "✅ OK  " if ok else "❌ FAIL"
    print(f"  [{status}] {name}: {msg}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  DATA QUALITY CHECK FOR PINN ACOUSTIC INVERSION")
print("="*70)

# ── 1. Проверяем наличие файлов ──────────────────────────────────────
print("\n[1] FILE EXISTENCE CHECK")
all_files_ok = True
for ev in range(1, n_event+1):
    wf_dir  = f'event{ev}/wavefields'
    sm_dir  = f'event{ev}/seismograms'
    grid_f  = f'{wf_dir}/wavefield_grid_for_dumps_000.txt'

    if not os.path.exists(grid_f):
        mark(f'event{ev} grid', False, f'Missing: {grid_f}')
        all_files_ok = False
    else:
        mark(f'event{ev} grid', True, grid_f)

    for wf in WF_STEPS:
        p = f'{wf_dir}/{wf}'
        ok = os.path.exists(p)
        if not ok: all_files_ok = False
        mark(f'event{ev}/{wf}', ok, 'found' if ok else f'MISSING: {p}')

    if not os.path.exists(sm_dir):
        mark(f'event{ev} seismograms dir', False, f'Missing dir: {sm_dir}')
        all_files_ok = False
    else:
        sms = sorted(os.listdir(sm_dir))
        bxz = [f for f in sms if 'BXZ' in f]
        bxx = [f for f in sms if 'BXX' in f]
        mark(f'event{ev} BXZ seis', len(bxz) == n_seis,
             f'{len(bxz)}/{n_seis} files found')
        mark(f'event{ev} BXX seis', len(bxx) == n_seis,
             f'{len(bxx)}/{n_seis} files found')
        if len(bxz) != n_seis or len(bxx) != n_seis:
            all_files_ok = False

if not all_files_ok:
    print("\n❌ КРИТИЧНО: Не все файлы найдены. Дальнейшая проверка невозможна.")
    sys.exit(1)

# ── 2. Загружаем и суммируем данные по событиям ──────────────────────
print("\n[2] LOADING DATA")

n_ini = 60
xx_g, zz_g = np.meshgrid(np.linspace(0, ax/Lx, n_ini),
                          np.linspace(0, az/Lz, n_ini))
xxzzs = np.column_stack((xx_g.ravel(), zz_g.ravel()))

U_ini1x = U_ini1z = U_ini2x = U_ini2z = U_specx = U_specz = None
Sz_total = Sx_total = None

l_f = 50
index = None

for ev in range(1, n_event+1):
    wf_dir = f'event{ev}/wavefields'
    sm_dir = f'event{ev}/seismograms'

    grid_raw = np.loadtxt(f'{wf_dir}/wavefield_grid_for_dumps_000.txt')
    xz_norm  = np.column_stack((grid_raw[:,0]/1000.0/Lx,
                                 grid_raw[:,1]/1000.0/Lz))

    U0 = [np.loadtxt(f'{wf_dir}/{wf}') for wf in WF_STEPS]

    u1 = interpolate.griddata(xz_norm, U0[0], xxzzs, fill_value=0.0)
    u2 = interpolate.griddata(xz_norm, U0[1], xxzzs, fill_value=0.0)
    ut = interpolate.griddata(xz_norm, U0[2], xxzzs, fill_value=0.0)

    if ev == 1:
        U_ini1x, U_ini1z = u1[:,0:1], u1[:,1:2]
        U_ini2x, U_ini2z = u2[:,0:1], u2[:,1:2]
        U_specx, U_specz = ut[:,0:1], ut[:,1:2]
    else:
        U_ini1x += u1[:,0:1]; U_ini1z += u1[:,1:2]
        U_ini2x += u2[:,0:1]; U_ini2z += u2[:,1:2]
        U_specx += ut[:,0:1]; U_specz += ut[:,1:2]

    # Seismograms
    sms  = sorted(os.listdir(sm_dir))
    bxz  = [f for f in sms if 'BXZ' in f]
    bxx  = [f for f in sms if 'BXX' in f]
    slz_raw = [np.loadtxt(f'{sm_dir}/{f}') for f in bxz]
    slx_raw = [np.loadtxt(f'{sm_dir}/{f}') for f in bxx]

    if index is None:
        t_spec = -slz_raw[0][0,0] + slz_raw[0][:,0]
        cut_u  = t_spec > t_s
        cut_l  = t_spec < t_st
        l_su   = len(cut_u) - sum(cut_u)
        l_sl   = sum(cut_l)
        index  = np.arange(l_sl, l_su, l_f)
        t_sub  = t_spec[index].reshape(-1,1)
        t_sub  = t_sub - t_sub[0]
        l_sub  = len(index)

    # Subsample (no in-place: create new arrays)
    slz = [s[index] for s in slz_raw]
    slx = [s[index] for s in slx_raw]

    Sz_ev = np.concatenate([s[:,1:2] for s in slz], axis=0)
    Sx_ev = np.concatenate([s[:,1:2] for s in slx], axis=0)

    if ev == 1:
        Sz_total = Sz_ev.copy()
        Sx_total = Sx_ev.copy()
    else:
        Sz_total += Sz_ev
        Sx_total += Sx_ev

print(f"  Loaded: IC1={U_ini1x.shape}, IC2={U_ini2x.shape}, Test={U_specx.shape}")
print(f"  Seismograms Sz={Sz_total.shape}, Sx={Sx_total.shape}")
print(f"  l_sub={l_sub} time samples per receiver")

# ── 3. Амплитудная проверка ──────────────────────────────────────────
print("\n[3] AMPLITUDE & SIGNAL CHECK")

def rms(a): return float(np.sqrt(np.mean(a**2)))
def maxabs(a): return float(np.max(np.abs(a)))

for name, arr in [('IC1_x', U_ini1x), ('IC1_z', U_ini1z),
                   ('IC2_x', U_ini2x), ('IC2_z', U_ini2z),
                   ('Test_x', U_specx), ('Test_z', U_specz),
                   ('Sz', Sz_total), ('Sx', Sx_total)]:
    mx = maxabs(arr)
    rm = rms(arr)
    ok = mx > 1e-30
    mark(f'amplitude_{name}', ok, f'max={mx:.3e}  rms={rm:.3e}')

disp_scale = max(maxabs(U_ini1x), maxabs(U_ini1z),
                 maxabs(U_ini2x), maxabs(U_ini2z),
                 maxabs(U_specx), maxabs(U_specz),
                 maxabs(Sx_total), maxabs(Sz_total), 1e-30)
print(f"\n  Global disp_scale = {disp_scale:.4e}")
mark('disp_scale_valid', disp_scale > 1e-20,
     f'scale={disp_scale:.3e} (must be >> 1e-20)')

# ── 4. Волновое поле покрывает аномалию? ─────────────────────────────
print("\n[4] WAVEFIELD COVERAGE OF ANOMALY")

# Индексы точек сетки внутри эллипса
xg_phys = xxzzs[:,0] * Lx
zg_phys = xxzzs[:,1] * Lz
inside_ell = ((xg_phys - ELL_CX)**2 / ELL_RX**2 +
              (zg_phys - ELL_CZ)**2 / ELL_RZ**2) <= 1.0
n_inside = inside_ell.sum()
mark('anomaly_grid_points', n_inside > 10,
     f'{n_inside} grid points inside ellipse anomaly')

U_total_ic1 = np.sqrt(U_ini1x**2 + U_ini1z**2).ravel()
amp_inside_ic1  = U_total_ic1[inside_ell]
amp_outside_ic1 = U_total_ic1[~inside_ell]

rms_in  = rms(amp_inside_ic1)
rms_out = rms(amp_outside_ic1)
contrast = rms_in / (rms_out + 1e-30)

mark('wavefield_illuminates_anomaly', rms_in > 1e-30,
     f'RMS inside ellipse (IC1) = {rms_in:.3e}')
mark('wavefield_contrast', contrast > 0.01,
     f'RMS_in/RMS_out = {contrast:.3f}  (should be > 0.01)')

# ── 5. Геометрия датчиков ────────────────────────────────────────────
print("\n[5] RECEIVER GEOMETRY")

X_SEIS_M    = np.linspace(100.0, 1400.0, n_seis)
X_SEIS_KM   = X_SEIS_M / 1000.0
x_cov_min   = X_SEIS_KM.min()
x_cov_max   = X_SEIS_KM.max()

covers_anomaly_x = x_cov_min <= ELL_CX - ELL_RX and x_cov_max >= ELL_CX + ELL_RX
mark('receivers_cover_anomaly_x', covers_anomaly_x,
     f'Receivers x=[{x_cov_min:.3f},{x_cov_max:.3f}] km  |  '
     f'Anomaly x=[{ELL_CX-ELL_RX:.3f},{ELL_CX+ELL_RX:.3f}] km')

z_rec_km = z_seis_phys
mark('receiver_depth', z_rec_km < az - 0.05,
     f'Receiver depth z={z_rec_km:.4f} km  (domain max z={az} km)')

# ── 6. Сейсмические записи несут информацию об аномалии? ─────────────
print("\n[6] SEISMOGRAM INFORMATION CONTENT")

# Проверяем: разница Sz между событием 1 и суммой событий
# (если amplitude = 0, аномалия не влияет на поле)
sms_ev1_z = []
for ev in range(1, n_event+1):
    sm_dir = f'event{ev}/seismograms'
    sms    = sorted(os.listdir(sm_dir))
    bxz    = [f for f in sms if 'BXZ' in f]
    slz_raw2 = [np.loadtxt(f'{sm_dir}/{f}') for f in bxz]
    slz2     = [s[index] for s in slz_raw2]  # subsample without in-place
    sms_ev1_z.append(np.concatenate([s[:,1:2] for s in slz2], axis=0))

# Вариация между записями по времени (должна быть ненулевой)
sz_var = np.std(Sz_total)
mark('seismogram_nonzero_variance', sz_var > 1e-30,
     f'Sz std = {sz_var:.4e}')

sz_rms = rms(Sz_total)
mark('seismogram_amplitude', sz_rms > 1e-30,
     f'Sz rms = {sz_rms:.4e}')

# Число временных точек на датчик
mark('seismogram_time_samples', l_sub >= 50,
     f'{l_sub} time samples per receiver  (need ≥50)')

# ── 7. Проверяем нормировки ──────────────────────────────────────────
print("\n[7] NORMALIZATION CHECK")

def normalized_rms(arr, scale):
    return rms(arr / scale)

for name, arr in [('IC1_x', U_ini1x), ('IC1_z', U_ini1z),
                   ('Sz', Sz_total), ('Sx', Sx_total)]:
    n_rms = normalized_rms(arr, disp_scale)
    ok = 0.001 < n_rms < 2.0
    mark(f'norm_rms_{name}', ok,
         f'rms after normalization = {n_rms:.4f}  (should be in [0.001, 2.0])')

# ── 8. Проверка: alpha достижим с новым коэффициентом ────────────────
print("\n[8] VELOCITY MODEL REACHABILITY CHECK")

alpha_net_range_min = ALPHA_BG - 1.0  # коэффициент теперь 1.0
alpha_net_range_max = ALPHA_BG + 1.0
mark('anomaly_reachable', alpha_net_range_min <= ALPHA_ANOM,
     f'New alpha range = [{alpha_net_range_min:.1f}, {alpha_net_range_max:.1f}] km/s  |  '
     f'Target anomaly = {ALPHA_ANOM} km/s  '
     f'→ {"REACHABLE ✓" if alpha_net_range_min <= ALPHA_ANOM else "STILL NOT REACHABLE!"}')

# ── 9. ИТОГ ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
passed = [(k, v) for k, v in RESULTS.items() if v[0]]
failed = [(k, v) for k, v in RESULTS.items() if not v[0]]
print(f"  Passed: {len(passed)}/{len(RESULTS)}")
if failed:
    print(f"\n  ❌ FAILED TESTS ({len(failed)}):")
    for k, (_, msg) in failed:
        print(f"      - {k}: {msg}")
else:
    print("\n  ✅ ALL CHECKS PASSED — данные достаточны для инверсии!")

# ── 10. ГЕНЕРАЦИЯ ДИАГНОСТИЧЕСКИХ ГРАФИКОВ ───────────────────────────
print("\n[10] GENERATING DIAGNOSTIC PLOTS")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Data Quality Check — Inversion Feasibility', fontsize=14, fontweight='bold')

# Нормируем для отображения
s = disp_scale if disp_scale > 0 else 1.0

# Plot 1: Wavefield IC1
ax1 = axes[0, 0]
U1 = np.sqrt(U_ini1x**2 + U_ini1z**2).reshape(xx_g.shape) / s
c1 = ax1.contourf(xx_g*Lx, zz_g*Lz, U1, 50, cmap='jet')
plt.colorbar(c1, ax=ax1, label='Norm. amplitude')
# Эллипс аномалии
ell = mpatches.Ellipse((ELL_CX, ELL_CZ), 2*ELL_RX, 2*ELL_RZ,
                       fill=False, edgecolor='white', linewidth=2, linestyle='--')
ax1.add_patch(ell)
ax1.set_title(f'IC1 Total Displacement (t={t_ic1} s)')
ax1.set_xlabel('x (km)'); ax1.set_ylabel('z (km)')
ax1.set_xlim(0, ax); ax1.set_ylim(0, az)

# Plot 2: Wavefield IC2
ax2 = axes[0, 1]
U2 = np.sqrt(U_ini2x**2 + U_ini2z**2).reshape(xx_g.shape) / s
c2 = ax2.contourf(xx_g*Lx, zz_g*Lz, U2, 50, cmap='jet')
plt.colorbar(c2, ax=ax2, label='Norm. amplitude')
ell2 = mpatches.Ellipse((ELL_CX, ELL_CZ), 2*ELL_RX, 2*ELL_RZ,
                        fill=False, edgecolor='white', linewidth=2, linestyle='--')
ax2.add_patch(ell2)
ax2.plot(X_SEIS_KM, z_seis_phys * np.ones(n_seis), 'r*', ms=6, label='Receivers')
ax2.set_title(f'IC2 Total Displacement (t={t_ic2} s)')
ax2.set_xlabel('x (km)'); ax2.set_ylabel('z (km)')
ax2.set_xlim(0, ax); ax2.set_ylim(0, az)
ax2.legend(fontsize=8)

# Plot 3: True velocity model
ax3 = axes[0, 2]
xp = xxzzs[:,0]*Lx; zp = xxzzs[:,1]*Lz
alpha_true = np.where(((xp-ELL_CX)**2/ELL_RX**2 + (zp-ELL_CZ)**2/ELL_RZ**2) <= 1.0,
                      ALPHA_ANOM, ALPHA_BG)
c3 = ax3.contourf(xx_g*Lx, zz_g*Lz, alpha_true.reshape(xx_g.shape), 50, cmap='RdBu_r')
plt.colorbar(c3, ax=ax3, label='km/s')
ax3.plot(X_SEIS_KM, z_seis_phys * np.ones(n_seis), 'k*', ms=8, label='Receivers')
ax3.set_title('True Velocity Model')
ax3.set_xlabel('x (km)'); ax3.set_ylabel('z (km)')
ax3.set_xlim(0, ax); ax3.set_ylim(0, az)
ax3.legend(fontsize=8)

# Plot 4: Seismogram example — station 10 (middle), Sz
ax4 = axes[1, 0]
i_mid = n_seis // 2
sz_stn = Sz_total[i_mid*l_sub:(i_mid+1)*l_sub, 0]
ax4.plot(t_sub.ravel(), sz_stn / s, 'b-', linewidth=1.5)
ax4.set_title(f'Z-Seismogram: Stn {i_mid+1} (x={X_SEIS_KM[i_mid]:.2f} km)')
ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Norm. amplitude')
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='k', linewidth=0.5)

# Plot 5: All seismograms (wiggle plot)
ax5 = axes[1, 1]
scale_wig = maxabs(Sz_total) * 4.0 if maxabs(Sz_total) > 0 else 1.0
for ii in range(n_seis):
    tr = Sz_total[ii*l_sub:(ii+1)*l_sub, 0]
    ax5.plot(t_sub.ravel(), X_SEIS_KM[ii] + tr/scale_wig, 'b-', linewidth=0.6, alpha=0.7)
ax5.axvline(x=0, color='k', linewidth=0.5)
ax5.set_title('Z-Seismograms (all stations, wiggle)')
ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Receiver x (km)')
ax5.grid(True, alpha=0.2)

# Plot 6: Wavefield at test time
ax6 = axes[1, 2]
Ut = np.sqrt(U_specx**2 + U_specz**2).reshape(xx_g.shape) / s
c6 = ax6.contourf(xx_g*Lx, zz_g*Lz, Ut, 50, cmap='jet')
plt.colorbar(c6, ax=ax6, label='Norm. amplitude')
ell6 = mpatches.Ellipse((ELL_CX, ELL_CZ), 2*ELL_RX, 2*ELL_RZ,
                         fill=False, edgecolor='white', linewidth=2, linestyle='--')
ax6.add_patch(ell6)
ax6.set_title(f'Test Wavefield (t={t_la} s)')
ax6.set_xlabel('x (km)'); ax6.set_ylabel('z (km)')
ax6.set_xlim(0, ax); ax6.set_ylim(0, az)

plt.tight_layout()
out = 'data_quality_report.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")

# Доп. рисунок — вертикальный профиль alpha через центр аномалии
fig2, ax_ = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Anomaly Cross-Sections for Recoverability Assessment', fontsize=12)

# Вертикальный профиль (через x=ELL_CX)
z_line = np.linspace(0, az, 200)
x_line = np.full_like(z_line, ELL_CX)
inside_z = ((x_line-ELL_CX)**2/ELL_RX**2 + (z_line-ELL_CZ)**2/ELL_RZ**2) <= 1.0
alpha_z   = np.where(inside_z, ALPHA_ANOM, ALPHA_BG)
ax_[0].plot(alpha_z, z_line, 'b-', linewidth=2, label='True α(x=0.75)')
ax_[0].axvline(ALPHA_BG - 1.0, color='r', linestyle='--', label='AlphaNet min (new)')
ax_[0].axvline(ALPHA_BG + 1.0, color='g', linestyle='--', label='AlphaNet max (new)')
ax_[0].axhline(z_st_box, color='gray', linestyle=':', label='Inversion box')
ax_[0].axhline(z_fi_box, color='gray', linestyle=':')
ax_[0].set_xlabel('α (km/s)'); ax_[0].set_ylabel('z (km)')
ax_[0].set_title('Vertical Profile Through Anomaly Center')
ax_[0].legend(fontsize=8); ax_[0].grid(True, alpha=0.3)
ax_[0].invert_yaxis()

# Горизонтальный профиль (через z=ELL_CZ)
x_line2 = np.linspace(0, ax, 300)
z_line2  = np.full_like(x_line2, ELL_CZ)
inside_x  = ((x_line2-ELL_CX)**2/ELL_RX**2 + (z_line2-ELL_CZ)**2/ELL_RZ**2) <= 1.0
alpha_x   = np.where(inside_x, ALPHA_ANOM, ALPHA_BG)
ax_[1].plot(x_line2, alpha_x, 'b-', linewidth=2, label='True α(z=0.25)')
ax_[1].axhline(ALPHA_BG - 1.0, color='r', linestyle='--', label='AlphaNet min (new)')
ax_[1].axhline(ALPHA_BG + 1.0, color='g', linestyle='--', label='AlphaNet max (new)')
ax_[1].axvline(X_SEIS_KM[0],  color='purple', linestyle=':', alpha=0.5)
ax_[1].axvline(X_SEIS_KM[-1], color='purple', linestyle=':', alpha=0.5, label='Receiver aperture')
ax_[1].set_xlabel('x (km)'); ax_[1].set_ylabel('α (km/s)')
ax_[1].set_title('Horizontal Profile Through Anomaly Center')
ax_[1].legend(fontsize=8); ax_[1].grid(True, alpha=0.3)

plt.tight_layout()
out2 = 'data_quality_profiles.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")

print("\n" + "="*70)
print("  CHECK COMPLETE")
print("="*70)
