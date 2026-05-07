# SpecFem2D — PINN 2src Setup

## Параметры
- Домен: 1.5 x 0.5 km, сетка 150x50 (dx=10m)
- dt=5e-05s, Nstep=14000 → T=0.70s, f0=20.0Hz
- Источник 0: (400.0m, 85.0m)
- Источник 1: (1100.0m, 85.0m)
- Датчики: 20 шт, z=415.0m, x=[100..1400]m
- PML: 10 элементов
- Аномалия: vp=2000.0m/s, прямоугольник ix=[56..96], iz=[16..36]
- Снимки: шаги [2000, 2300, 5000]

## Запуск
```bash
export SPECFEM_BIN=/path/to/specfem2d/bin
bash run_specfem.sh
python3 convert_specfem_to_pinn.py
```
