#!/usr/bin/env python3
"""
convert_specfem_to_pinn.py
Конвертирует выходы SpecFem2D (src0/, src1/) в event1/, event2/ + .npy
Запускать из папки specfem2d_run/

ИСПРАВЛЕНИЯ v3:
  1. detect_grid_file()  — ищет и без суффикса _000, и с ним
  2. SNAP_STEPS теперь выбираются автоматически из реально доступных шагов
  3. save_npy() читает grid через detect_grid_file(), не хардкодит имя
  4. main() передаёт реальный путь к grid-файлу в write_event()
"""
import os, re, shutil
import numpy as np

# ── Параметры ────────────────────────────────────────────────────────────────
N_SEIS         = 20
NX_ELEM        = 150
NZ_ELEM        = 50
DX_M           = 10.0
DZ_M           = 10.0
LX_M           = 1500.0
LZ_M           = 500.0
X_SEIS         = np.linspace(100.0, 1400.0, 20)
Z_SEIS_VAL     = 415.0
VP_BG          = 3000.0
VP_ANOM        = 2000.0
ELL_CX, ELL_CZ = 750.0, 250.0
ELL_RX, ELL_RZ = 200.0, 100.0

# Целевые шаги для IC1, IC2, Test (шаги SpecFEM, по 5e-5 с/шаг)
# t_ic1=0.10s → step 2000, t_ic2=0.20s → step 4000, t_la=0.25s → step 5000
TARGET_STEPS = [2000, 4000, 5000]
# ─────────────────────────────────────────────────────────────────────────────


def detect_wf_suffix(out_dir):
    """Определяет суффикс файлов волновых полей: '_01_000.txt' или '_01.txt'."""
    for suf in ("_01_000.txt", "_01.txt"):
        files = [f for f in os.listdir(out_dir)
                 if f.startswith("wavefield") and f.endswith(suf)
                 and re.search(r"wavefield\d+", f)]
        if files:
            print(f"  [info] формат wavefield: *{suf}")
            return suf
    return None


def detect_grid_file(out_dir):
    """
    Ищет файл сетки в out_dir.
    Возможные имена (SpecFEM2D создаёт одно из двух):
      wavefield_grid_for_dumps_000.txt  (старый формат)
      wavefield_grid_for_dumps.txt      (новый формат, без _000)
    Возвращает полный путь или None.
    """
    candidates = [
        "wavefield_grid_for_dumps_000.txt",
        "wavefield_grid_for_dumps.txt",
    ]
    for name in candidates:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            print(f"  [info] grid-файл: {name}")
            return path
    # Широкий поиск на случай нестандартного имени
    for f in os.listdir(out_dir):
        if "grid_for_dumps" in f and f.endswith(".txt"):
            print(f"  [info] grid-файл (нестандартный): {f}")
            return os.path.join(out_dir, f)
    print(f"  [warn] grid-файл не найден в {out_dir}")
    return None


def collect_wavefields(src_dir):
    """
    Загружает первые n_snaps доступных волновых полей из src_dir/OUTPUT_FILES/.
    Возвращает dict {step: array} и список шагов.
    """
    out = os.path.join(src_dir, "OUTPUT_FILES")

    suf = detect_wf_suffix(out)
    if suf is None:
        raise FileNotFoundError(
            f"Нет файлов wavefield*.txt в {out}\n"
            f"  Убедись что в Par_file: output_wavefield_dumps = .true.\n"
            f"  Файлы в папке: {os.listdir(out)[:10]}"
        )

    # Собираем все доступные шаги (исключаем шаг 0000005 — это тест-дамп)
    all_wf = sorted([f for f in os.listdir(out)
                     if f.startswith("wavefield") and f.endswith(suf)])
    nums = []
    for f in all_wf:
        m = re.search(r"wavefield(\d+)", f)
        if m:
            step = int(m.group(1))
            if step > 0:          # пропускаем нулевой / инициализационный шаг
                nums.append(step)
    nums = sorted(set(nums))

    if not nums:
        raise FileNotFoundError(
            f"Не найдено ни одного wavefield-файла с ненулевым шагом в {out}"
        )

    # Выбираем ближайшие шаги к целевым TARGET_STEPS
    chosen = []
    for target in TARGET_STEPS:
        nearest = min(nums, key=lambda s: abs(s - target))
        chosen.append(nearest)
        if nearest != target:
            print(f"  [warn] шаг {target} не найден, использую ближайший {nearest}")

    print(f"  [info] доступные шаги (всего): {nums}")
    print(f"  [info] выбранные шаги: {chosen}")

    result = {}
    for step in chosen:
        fname = os.path.join(out, f"wavefield{step:07d}{suf}")
        result[step] = np.loadtxt(fname)
        print(f"  [wf] step {step}: {result[step].shape}")
    return result, chosen


def collect_seismograms(src_dir, n_seis):
    out = os.path.join(src_dir, "OUTPUT_FILES")
    all_semd = [f for f in os.listdir(out) if f.endswith(".semd")]

    def find_comp(tag_primary, tag_fallback):
        files = sorted([f for f in all_semd if tag_primary in f])
        if not files:
            files = sorted([f for f in all_semd if tag_fallback in f])
            if files:
                print(f"  [info] компонент '{tag_primary}' не найден, использую '{tag_fallback}'")
        return files

    bxz = find_comp("BXZ", "FXZ")
    bxx = find_comp("BXX", "FXX")

    if len(bxz) < n_seis or len(bxx) < n_seis:
        print(f"  [warn] найдено {len(bxz)} Z-компонент, {len(bxx)} X-компонент (ожидалось {n_seis})")
        print(f"  [info] все .semd файлы: {sorted(all_semd)[:6]} ...")

    sz = [np.loadtxt(os.path.join(out, f)) for f in bxz[:n_seis]]
    sx = [np.loadtxt(os.path.join(out, f)) for f in bxx[:n_seis]]
    return sx, sz


def write_event(event_dir, wf_dict, sx_list, sz_list, grid_src_path):
    """
    Записывает данные одного события в event_dir/.
    grid_src_path — полный путь к исходному grid-файлу (может быть None).
    """
    wf_dir  = os.path.join(event_dir, "wavefields")
    sei_dir = os.path.join(event_dir, "seismograms")
    os.makedirs(wf_dir,  exist_ok=True)
    os.makedirs(sei_dir, exist_ok=True)

    # Копируем grid под стандартным именем с суффиксом _000
    if grid_src_path and os.path.exists(grid_src_path):
        dst = os.path.join(wf_dir, "wavefield_grid_for_dumps_000.txt")
        shutil.copy2(grid_src_path, dst)
        print(f"  [grid] скопирован → {dst}")
    else:
        print(f"  [warn] grid-файл не скопирован: {grid_src_path}")

    # Wavefields
    for step, data in wf_dict.items():
        np.savetxt(os.path.join(wf_dir, f"wavefield{step:07d}_01_000.txt"),
                   data, fmt="%18.8e  %18.8e")

    # Seismograms
    for i, (sx, sz) in enumerate(zip(sx_list, sz_list)):
        np.savetxt(os.path.join(sei_dir, f"AA.S{i+1:04d}.BXX.semd"), sx, fmt="%18.8e  %18.8e")
        np.savetxt(os.path.join(sei_dir, f"AA.S{i+1:04d}.BXZ.semd"), sz, fmt="%18.8e  %18.8e")
    print(f"  [ok] {event_dir}: {len(sx_list)} seismograms, {len(wf_dict)} wavefields")


def save_npy(wf0, steps0, wf1, steps1, sx0, sz0, sx1, sz1):
    """Сохраняет .npy версию для patch_pinn_data_loader.py"""
    npy_dir = "data_2src_specfem"
    os.makedirs(npy_dir, exist_ok=True)

    # Читаем grid из event1 (уже скопирован под стандартным именем)
    grid_path = "event1/wavefields/wavefield_grid_for_dumps_000.txt"
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"Grid-файл не найден: {grid_path}\n"
            f"  write_event() должна была скопировать его туда."
        )
    grid = np.loadtxt(grid_path)
    xs = np.unique(grid[:, 0])
    zs = np.unique(grid[:, 1])
    nx, nz = len(xs), len(zs)
    print(f"  [npy] grid {nx} x {nz}")

    def to_snap(wf, step):
        d = wf[step]
        return np.stack([d[:, 0].reshape(nx, nz),
                         d[:, 1].reshape(nx, nz)], axis=-1)

    # Используем первые два реальных шага из каждого источника
    s0_0, s0_1 = steps0[0], steps0[1] if len(steps0) > 1 else steps0[0]
    s1_0, s1_1 = steps1[0], steps1[1] if len(steps1) > 1 else steps1[0]

    np.save(f"{npy_dir}/snap1_src0.npy", to_snap(wf0, s0_0))
    np.save(f"{npy_dir}/snap2_src0.npy", to_snap(wf0, s0_1))
    np.save(f"{npy_dir}/snap1_src1.npy", to_snap(wf1, s1_0))
    np.save(f"{npy_dir}/snap2_src1.npy", to_snap(wf1, s1_1))

    xx, zz = np.meshgrid(xs / 1000, zs / 1000, indexing="ij")   # в km
    np.save(f"{npy_dir}/coords_grid.npy", np.stack([xx, zz], axis=-1))
    np.save(f"{npy_dir}/seis_coords.npy",
            np.stack([X_SEIS / 1000, np.full(N_SEIS, Z_SEIS_VAL / 1000)], axis=-1))

    t_ref = sz0[0][:, 0]
    np.save(f"{npy_dir}/t_seism.npy", t_ref)

    for idx, (sx_l, sz_l) in enumerate([(sx0, sz0), (sx1, sz1)]):
        nt = sz_l[0].shape[0]
        seism = np.zeros((N_SEIS, nt, 2))
        for i in range(N_SEIS):
            seism[i, :, 0] = sx_l[i][:, 1]
            seism[i, :, 1] = sz_l[i][:, 1]
        np.save(f"{npy_dir}/seism_src{idx}.npy", seism)

    # alpha_true (в km/s)
    alpha = np.full((nx, nz), VP_BG / 1000)
    for ix, x in enumerate(xs):
        for iz, z in enumerate(zs):
            if ((x - ELL_CX)**2 / ELL_RX**2 + (z - ELL_CZ)**2 / ELL_RZ**2) <= 1.0:
                alpha[ix, iz] = VP_ANOM / 1000
    np.save(f"{npy_dir}/alpha_true.npy", alpha)
    print(f"  [npy] saved to {npy_dir}/")

    return s0_0, s0_1   # возвращаем шаги для финального отчёта


def main():
    print("=== convert_specfem_to_pinn.py ===")

    # Автоматически находим grid-файлы (с _000 или без)
    grid0 = detect_grid_file("src0/OUTPUT_FILES")
    grid1 = detect_grid_file("src1/OUTPUT_FILES")

    print("[src0] loading...")
    wf0, steps0 = collect_wavefields("src0")
    sx0, sz0 = collect_seismograms("src0", N_SEIS)
    write_event("event1", wf0, sx0, sz0, grid0)

    print("[src1] loading...")
    wf1, steps1 = collect_wavefields("src1")
    sx1, sz1 = collect_seismograms("src1", N_SEIS)
    write_event("event2", wf1, sx1, sz1, grid1)

    print("[npy] saving .npy copies...")
    t01_step, t02_step = save_npy(wf0, steps0, wf1, steps1, sx0, sz0, sx1, sz1)

    DT = 5e-5   # типичный DT для этих симуляций; скорректируй если нужно
    print()
    print("=== Done! ===")
    print("  event1/              — source 0 data")
    print("  event2/              — source 1 data")
    print("  data_2src_specfem/   — .npy format")
    print()
    print("В PINNs_Inversion_Acoustic.py установи: n_event = 2")
    print(f"  t01 = {t01_step} * DT = {t01_step * DT:.4f} s  (snap1)")
    print(f"  t02 = {t02_step} * DT = {t02_step * DT:.4f} s  (snap2)")


if __name__ == "__main__":
    main()