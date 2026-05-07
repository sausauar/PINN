import os
# Force use the first GPU (if available). РЈСЃС‚Р°РЅРѕРІРёС‚Рµ CUDA_VISIBLE_DEVICES='0' РґР»СЏ RTX4060.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pickle
import json
import gc
import sys
import shutil
import tensorflow as tf
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


def _get_nvcc_version():
    try:
        import subprocess, re
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"release\s+([0-9]+\.[0-9]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def _get_cudnn_path():
    try:
        import subprocess
        out = subprocess.check_output(["where", "cudnn64_8.dll"], stderr=subprocess.STDOUT, text=True)
        return out.strip().splitlines()[0]
    except Exception:
        return None


def print_cuda_status():
    nvcc = _get_nvcc_version()
    cudnn = _get_cudnn_path()
    print(f"[CUDA STATUS] nvcc version={nvcc}, cudnn64_8 path={cudnn}")


print_cuda_status()

# Р”Р»СЏ TensorFlow 2.x РЅСѓР¶РЅРѕ РІРєР»СЋС‡РёС‚СЊ eager (GradientTape СЂР°Р±РѕС‚Р°РµС‚ С‚РѕР»СЊРєРѕ СЃ eager execution).
# tf.compat.v1.disable_eager_execution()  # РћС‚РєР»СЋС‡РµРЅРѕ СЃРїРµС†РёР°Р»СЊРЅРѕ РґР»СЏ TF2 СЃС‚РёР»Рµ.

# Placeholder-like РІС…РѕРґС‹ Р±РѕР»СЊС€Рµ РЅРµ РЅСѓР¶РЅС‹ Р·РґРµСЃСЊ, РјС‹ РїРµСЂРµРґР°РµРј РґР°РЅРЅС‹Рµ РЅР°РїСЂСЏРјСѓСЋ РІ train_step.
# x, z, t Р·Р°РґР°СЋС‚СЃСЏ РёР· РІС…РѕРґРЅРѕРіРѕ Р±Р°С‚С‡Р° РІРЅСѓС‚СЂРё train_step.

# РџСЂРѕРІРµСЂРєР° РґРѕСЃС‚СѓРїРЅРѕСЃС‚Рё GPU
# РЈР±РµРґРёС‚РµСЃСЊ, С‡С‚Рѕ `CUDA_VISIBLE_DEVICES` Р·Р°РґР°РЅ Р”Рћ РёРјРїРѕСЂС‚Р° tensorflow (РЅРёР¶Рµ СѓСЃС‚Р°РЅРѕРІР»РµРЅРѕ "0").
# Р’ TensorFlow 2.x РЅР° Windows РЅРµСЂРµРґРєРѕ РЅСѓР¶РµРЅ WSL2/DirectML РґР»СЏ СЂР°Р±РѕС‚С‹ РЅР° GPU.

def detect_and_configure_gpu():
    print(f"[TF STATUS] version=3, built_with_cuda={tf.test.is_built_with_cuda()}")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print('[TF STATUS] listed physical_devices GPU:', gpus)
    except Exception as e:
        print('РћС€РёР±РєР° РїСЂРё РїРѕР»СѓС‡РµРЅРёРё СЃРїРёСЃРєР° СѓСЃС‚СЂРѕР№СЃС‚РІ GPU:', e)
        gpus = []

    if gpus:
        print('Detected GPU devices:', gpus)
        print('Using GPU for training.')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f'РќРµ СѓРґР°Р»РѕСЃСЊ РІРєР»СЋС‡РёС‚СЊ СЂРѕСЃС‚ РїР°РјСЏС‚Рё РґР»СЏ GPU {gpu}:', e)
        try:
            tf.config.optimizer.set_jit(True)
            print('[TF STATUS] XLA JIT enabled.')
        except Exception as e:
            print(f'[TF STATUS] Could not enable XLA JIT: {e}')
        return gpus

    # GPU РЅРµ РЅР°Р№РґРµРЅ, СЃРѕРІРµС‚ РїРѕ CUDA/TF
    nvcc = _get_nvcc_version()
    if nvcc:
        if nvcc.startswith('11.'):
            print('Р РµРєРѕРјРµРЅРґРѕРІР°РЅРѕ TensorFlow 2.12 СЃ CUDA 11.8 + cuDNN 8.6.')
        else:
            print('РЈСЃС‚Р°РЅРѕРІР»РµРЅР° CUDA', nvcc, '- РґР»СЏ РїРѕРґРґРµСЂР¶РєРё GPU РѕР±С‹С‡РЅРѕ TF 2.15+ (tensorflow-cuda) С‚СЂРµР±СѓРµС‚СЃСЏ.')

    # Р”РѕРїРѕР»РЅРёС‚РµР»СЊРЅР°СЏ РїСЂРѕРІРµСЂРєР° РґР»СЏ TF1-СЃРѕРІРјРµСЃС‚РёРјРѕСЃС‚Рё
    try:
        if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
            print('tf.test.is_gpu_available() РіРѕРІРѕСЂРёС‚, С‡С‚Рѕ GPU РЅР°Р№РґРµРЅ.')
            return tf.config.list_physical_devices('GPU')
    except AttributeError:
        pass

    print('GPU not found, using CPU.')
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception as e:
        print('РќРµ СѓРґР°Р»РѕСЃСЊ СЃРєСЂС‹С‚СЊ СѓСЃС‚СЂРѕР№СЃС‚РІР° GPU (СЌС‚Рѕ РЅРѕСЂРјР°Р»СЊРЅРѕ РґР»СЏ CPU-only):', e)
    return []


gpu_devices = detect_and_configure_gpu()

import matplotlib.pyplot as plt
import time
import timeit
from SALib.sample import sobol_sequence
import scipy.interpolate as interpolate
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='PINN Acoustic Inversion Training')
parser.add_argument('--num-epochs', type=int, default=50001, help='Number of training epochs (full run, consistent with the paper-style setup)')
parser.add_argument('--batch-size', type=int, default=8000, help='Batch size for PDE samples')
parser.add_argument('--n-pde-total', type=int, default=60000, help='Total PDE Sobol points available across epochs')
parser.add_argument('--fast', action='store_true', help='Fast mode for testing (200 epochs)')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
parser.add_argument('--load-checkpoint', type=str, help='Path to load checkpoint from')
parser.add_argument('--resume-epoch', type=int, default=None, help='Optional absolute epoch to resume from (e.g. 8000). If omitted, inferred from checkpoint.')
parser.add_argument('--w-pde', type=float, default=0.05,
                    help='PDE loss weight. MUST be << w_ic/w_seis to avoid trivial phi=0 solution')
parser.add_argument('--w-ic', type=float, default=1.0, help='Initial-condition loss weight')
parser.add_argument('--w-seis', type=float, default=1.0, help='Seismogram loss weight')
parser.add_argument('--w-bc', type=float, default=0.15, help='Boundary-condition loss weight')
parser.add_argument('--gpu-id', type=str, default=None, help='Optional GPU id override, e.g. 0')
parser.add_argument('--mirror-dir', type=str, default=None, help='Optional directory where plots/results are mirrored')
parser.add_argument('--chunk-size', type=int, default=8000, help='Chunk size for second-derivative computation')
parser.add_argument('--warmup-epochs', type=int, default=1200, help='Epochs to prioritize IC + seismograms before full PINN weighting')
parser.add_argument('--log-every', type=int, default=100, help='Loss logging cadence in epochs')
parser.add_argument('--plot-every', type=int, default=400, help='Core plot cadence in epochs')
parser.add_argument('--detailed-every', type=int, default=800, help='Detailed comparison plot cadence in epochs')
parser.add_argument('--checkpoint-every', type=int, default=400, help='Checkpoint cadence in epochs')
parser.add_argument('--w-ic-phi', type=float, default=0.05, help='Auxiliary phi-matching weight inside IC loss')
parser.add_argument('--w-seis-phi', type=float, default=0.05, help='Auxiliary relative-phi-matching weight inside Seis loss')
parser.add_argument('--w-ic-grad', type=float, default=1.00, help='Gradient-matching weight inside IC loss (primary)')
parser.add_argument('--w-seis-grad', type=float, default=1.00, help='Gradient-matching weight inside Seis loss (primary)')
parser.add_argument('--grad-clip-norm', type=float, default=5.0, help='Global norm clipping for optimizer gradients (<=0 disables)')
parser.add_argument('--phase1-end', type=int, default=200, help='Curriculum phase 1 end epoch (IC-only)')
parser.add_argument('--phase2-end', type=int, default=1200, help='Curriculum phase 2 end epoch (IC+Seis ramp)')
parser.add_argument('--pde-ramp-epochs', type=int, default=2000, help='Epoch span for PDE/BC ramp after phase2')
args = parser.parse_args()
if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
os.makedirs(args.checkpoint_dir, exist_ok=True)
data_norm_info = {}

# Mirroring is OFF by default. It is enabled only if --mirror-dir is explicitly provided.
if args.mirror_dir:
    args.mirror_dir = os.path.abspath(args.mirror_dir)
    os.makedirs(args.mirror_dir, exist_ok=True)
    print(f"[OUTPUT MIRROR] Mirroring results to: {args.mirror_dir}")

def mirror_output(path):
    if not args.mirror_dir:
        return
    if not os.path.exists(path):
        return
    try:
        destination = os.path.join(args.mirror_dir, os.path.basename(path))
        shutil.copy2(path, destination)
    except Exception as exc:
        print(f"[WARNING] Could not mirror {path} -> {args.mirror_dir}: {exc}")

def save_figure(path, *args, **kwargs):
    plt.savefig(path, *args, **kwargs)
    mirror_output(path)


def save_run_parameters(path, args_obj, weights_obj, norm_obj=None):
    payload = {
        'num_epochs': int(args_obj.num_epochs),
        'batch_size': int(args_obj.batch_size),
        'n_pde_total': int(args_obj.n_pde_total),
        'chunk_size': int(args_obj.chunk_size),
        'warmup_epochs': int(args_obj.warmup_epochs),
        'log_every': int(args_obj.log_every),
        'plot_every': int(args_obj.plot_every),
        'detailed_every': int(args_obj.detailed_every),
        'checkpoint_every': int(args_obj.checkpoint_every),
        'loss_weights': {k: float(v) for k, v in weights_obj.items()}
    }
    if norm_obj:
        payload['normalization'] = norm_obj
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    mirror_output(path)


def infer_start_epoch_from_checkpoint_path(path, checkpoint_every):
    """Infer absolute resume epoch from ckpt-<N> naming convention."""
    if not path:
        return 0
    base = os.path.basename(path.rstrip('/'))
    if base.startswith('ckpt-'):
        try:
            step_id = int(base.split('-')[-1])
            cadence = int(checkpoint_every)
            # Guard against accidental huge test cadences passed on CLI during resume.
            if cadence <= 0 or cadence > 5000:
                cadence = 400
            return max(0, step_id * cadence)
        except Exception:
            return 0
    return 0


def load_loss_history_csv(path):
    if not os.path.exists(path):
        return []
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        if data.size == 0:
            return []
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data.tolist()
    except Exception:
        return []


def save_loss_history_csv(path, history_rows):
    header = 'epoch,total,pde,ic,seis,bc'
    if len(history_rows) == 0:
        np.savetxt(path, np.empty((0, 6), dtype=np.float32), delimiter=',', header=header, comments='')
    else:
        arr = np.array(history_rows, dtype=np.float64)
        np.savetxt(path, arr, delimiter=',', header=header, comments='', fmt=['%d', '%.10e', '%.10e', '%.10e', '%.10e', '%.10e'])
    mirror_output(path)

# Override defaults based on GPU availability and args
num_epoch = args.num_epochs
batch_size = args.batch_size
if not gpu_devices:
    print('Switching to reduced training number for CPU: 4000 epochs and smaller batch.')
    num_epoch = min(num_epoch, 4000)
    batch_size = min(batch_size, 2000)

chunk_size_runtime = max(256, int(args.chunk_size))
if gpu_devices:
    chunk_size_runtime = min(chunk_size_runtime, batch_size)
else:
    chunk_size_runtime = min(chunk_size_runtime, 1000, batch_size)

if args.fast:
    print('Fast mode ON, num_epoch=200 (РґР»СЏ Р±С‹СЃС‚СЂРѕР№ РїСЂРѕРІРµСЂРєРё)')
    num_epoch = 200

# ── Параметры домена (совпадают с generate_data_2src.py) ──────
DATA_DIR   = 'data_2src'   # папка с .npy данными от генератора
ax         = 1.5           # размер домена x (km)
az         = 0.5           # размер домена z (km)
t_ic1      = 0.10          # время первого снимка (s) = step 2000 * DT(5e-5) = 0.10 s
t_ic2      = 0.20          # время второго снимка (s) = step 4000 * DT(5e-5) = 0.20 s
t_la       = 0.25          # время тестового снимка  = step 5000 * DT(5e-5) = 0.25 s
t_m        = 0.45          # полное время PDE обучения (s)
t_st       = t_ic1         # PDE окно начинается с IC
t_s        = 0.45          # длина записи сейсмограмм
t01        = t_ic1
t02        = t_ic2
n_event    = 2             # два источника (суперпозиция)
n_seis     = 20            # число датчиков
sponge_cells = 15
fd_dz      = 0.005         # шаг FD сетки (km)
# z-координата датчиков (выше нижнего sponge, как в генераторе)
z_seis_phys = az - (sponge_cells + 2) * fd_dz   # ≈ 0.415 km
rho = 1.0



Lx=3.0;#this is for scaling the wavespeed in the PDE via saling x coordinate
Lz=3.0;#this is for scaling the wavespeed in the PDE via scaling z coordinate

# ── Границы инверсионного прямоугольника (вокруг эллипса) ────
# Эллипс: cx=0.75, cz=0.25, rx=0.20, rz=0.10
z_st = 0.10   # km
z_fi = 0.42   # km
x_st = 0.45   # km
x_fi = 1.05   # km
lld_smooth = 20.0

# ── Истинная скоростная модель (эллипс, только для графиков) ──
ALPHA_BG   = 3.0   # km/s фон
ALPHA_ANOM = 2.0   # km/s аномалия
ELL_CX, ELL_CZ = 0.75, 0.25
ELL_RX, ELL_RZ = 0.20, 0.10

def compute_alpha_true_numpy(xx, zz):
    x_vals = xx * Lx
    z_vals = zz * Lz
    alpha = np.full_like(x_vals, ALPHA_BG, dtype=np.float64)
    mask = ((x_vals - ELL_CX)**2 / ELL_RX**2 +
            (z_vals - ELL_CZ)**2 / ELL_RZ**2) <= 1.0
    alpha[mask] = ALPHA_ANOM
    return alpha

def source_force_tf(x, z, t):
    """Источник закодирован в IC — в PDE f=0."""
    return tf.zeros_like(x)

# alpha_true will be computed when needed for plotting
alpha_true = None  # Will be computed as numpy array for plotting

ub=np.array([ax/Lx,az/Lz,(t_m-t_st)], dtype=np.float32).reshape(-1,1).T# normalization of the input to the NN
ub0=np.array([ax/Lx,az/Lz], dtype=np.float32).reshape(-1,1).T#same for the inverse NN estimating the wave_speed



def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)


def neural_net(X, weights, biases):
    """Wave field network: (x,z,t) в†’ П†. Must be differentiable w.r.t. X."""
    num_layers = len(weights) + 1
    # Ensure ub is a TF tensor for proper gradient flow
    ub_tf = tf.cast(ub, tf.float32)
    H = 2.0 * (X / ub_tf) - 1.0  # Normalization to [-1, 1]
    
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


def neural_net0(X, weights, biases):
    """Velocity network: (x,z) → О±_star. Must be differentiable w.r.t. X."""
    num_layers = len(weights) + 1
    # Ensure ub0 is a TF tensor for proper gradient flow
    ub0_tf = tf.cast(ub0, tf.float32)
    H = 2.0 * (X / ub0_tf) - 1.0  # Normalization to [-1, 1]
    
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

layers=[3]+[100]*8+[1] # layers for the NN approximating the scalar acoustic potential (Case 4: 8 layers, 100 neurons)
layers0=[2]+[20]*5+[1] # layers for the second NN to approximate the wavespeed
# NOTE: weights and biases will be created inside PINNModel.__init__

learning_rate = 3.e-4  # More stable for inversion than the previous aggressive step

# Learning rate scheduler: exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=5000,   # was 2000 — softer decay to avoid LR dying too early
    decay_rate=0.985,   # was 0.97  — at epoch 50k: 0.985^10=0.86 vs 0.97^25=0.47
    staircase=False
)

# ========== ARCHITECTURE: TWO SEPARATE NETWORKS ==========
class WaveNet(tf.keras.Model):
    """
    Wave Field Network: (x, z, t) -> [ux(x,z,t), uz(x,z,t)]
    Directly predicts displacement components (FIRST-ORDER gradient for IC/seis!).
    Architecture: 3-input -> 8x100 hidden -> 2-output
    """
    def __init__(self, layers_list, ub_norm):
        super(WaveNet, self).__init__()
        self.ub_norm = tf.constant(ub_norm, dtype=tf.float32)  # Store as tensor constant
        self.W = []
        self.b = []
        for l in range(0, len(layers_list)-1):
            # Use self.add_weight() to ensure TensorFlow tracks these variables
            w = self.add_weight(shape=(layers_list[l], layers_list[l+1]),
                               initializer='glorot_uniform',
                               trainable=True, name=f"wave_w{l}")
            bias = self.add_weight(shape=(1, layers_list[l+1]),
                                 initializer='zeros',
                                 trainable=True, name=f"wave_b{l}")
            self.W.append(w)
            self.b.append(bias)

    def call(self, X):
        """Compute П†(x,z,t) with proper gradient tracking through constants."""
        H = 2.0 * (X / self.ub_norm) - 1.0
        
        # Hidden layers with tanh activation
        for i in range(len(self.W) - 1):
            H = tf.nn.tanh(tf.add(tf.matmul(H, self.W[i]), self.b[i]))
        
        # Last layer (linear output)
        return tf.add(tf.matmul(H, self.W[-1]), self.b[-1])


class AlphaNet(tf.keras.Model):
    """
    Velocity Inversion Network: (x, z) в†’ О±_star в€€ [-1, 1]
    Architecture: 2-input в†’ 5Г—20 hidden в†’ 1-output tanh
    Uses self.add_weight() for proper variable tracking by TensorFlow.
    """
    def __init__(self, layers_list, ub0_norm):
        super(AlphaNet, self).__init__()
        self.ub0_norm = tf.constant(ub0_norm, dtype=tf.float32)  # Store as tensor constant
        self.W = []
        self.b = []
        for l in range(0, len(layers_list)-1):
            # Use self.add_weight() to ensure TensorFlow tracks these variables
            w = self.add_weight(shape=(layers_list[l], layers_list[l+1]),
                               initializer='glorot_uniform',
                               trainable=True, name=f"alpha_w{l}")
            bias = self.add_weight(shape=(1, layers_list[l+1]),
                                 initializer='zeros',
                                 trainable=True, name=f"alpha_b{l}")
            self.W.append(w)
            self.b.append(bias)

    def call(self, X):
        """Compute О±_star(x,z) в€€ [-1, 1] with proper gradient tracking through constants."""
        H = 2.0 * (X / self.ub0_norm) - 1.0
        
        # Hidden layers with tanh activation
        for i in range(len(self.W) - 1):
            H = tf.nn.tanh(tf.add(tf.matmul(H, self.W[i]), self.b[i]))
        
        # Last layer: linear в†’ tanh for output в€€ [-1, 1]
        return tf.tanh(tf.add(tf.matmul(H, self.W[-1]), self.b[-1]))


def layered_velocity_tf(x, z):
    """Однородный фон 3.0 km/s. Заменить на слоистую модель при необходимости."""
    return tf.constant(3.0, dtype=tf.float32) * tf.ones_like(x)


def compute_velocity_with_mask(x, z, alpha_star, z_st, z_fi, x_st, x_fi, Lx, Lz, lld=1000.0):
    """
    CRITICAL: Apply mask OUTSIDE the neural networks.
    This prevents gradient artifacts at mask boundaries during PDE computation.
    
    Mathematical structure:
      О±(x,z) = 3.0 + 2.0 * О±_star(x,z) * mask(x,z)
    
    where:
      - 3.0 = background velocity (km/s)
      - О±_star в€€ [-1, 1] = network output (velocity perturbation)
      - mask в€€ [0, 1] = rectangular inversion box (smooth tanh transitions)
    
    This separation ensures CLEAN GRADIENTS for wave equation.
    
    Args:
        x, z: spatial coordinates (already normalized)
        alpha_star: output from AlphaNet в€€ [-1, 1]
        z_st, z_fi, x_st, x_fi: inversion box boundaries
        Lx, Lz: scaling factors
        lld: parameter controlling mask sharpness (default: 1000 from original)
    
    Returns:
        alpha: final velocity field О± в€€ [1, 5] km/s
        alpha_bound: mask в€€ [0, 1] for visualization
    """
    # Normalize boundaries to scaled coordinates
    z_st_norm = z_st / Lz
    z_fi_norm = z_fi / Lz
    x_st_norm = x_st / Lx
    x_fi_norm = x_fi / Lx
    
    # Compute box boundary masks (smooth tanh transitions)
    # Each mask is independent: 0 outside boundary, 1 inside
    z_low_mask = 0.5 * (1 + tf.tanh(lld * (z - z_st_norm)))
    z_high_mask = 0.5 * (1 + tf.tanh(lld * (-z + z_fi_norm)))
    x_low_mask = 0.5 * (1 + tf.tanh(lld * (x - x_st_norm)))
    x_high_mask = 0.5 * (1 + tf.tanh(lld * (-x + x_fi_norm)))
    
    # Combined mask: 1 inside box, 0 outside
    # This uses MULTIPLICATION of four smooth boundaries
    alpha_bound = z_low_mask * z_high_mask * x_low_mask * x_high_mask
    
    # Final velocity = layered background + bounded perturbation inside inversion box.
    # FIXED: coefficient 0.6 → 1.0 so alpha can reach 2.0 km/s (anomaly target).
    # With 0.6: range was [2.4, 3.6] km/s — anomaly at 2.0 was UNREACHABLE!
    # With 1.0: range is  [2.0, 4.0] km/s — correct.
    x_phys = x * Lx
    z_phys = z * Lz
    alpha_base = layered_velocity_tf(x_phys, z_phys)
    alpha = alpha_base + 1.0 * alpha_star * alpha_bound
    alpha = tf.clip_by_value(alpha, 1.2, 4.5)
    
    return alpha, alpha_bound


def compute_model_outputs(inputs):
    """Inference only - returns phi and alpha without gradient computation."""
    inputs = tf.cast(inputs, tf.float32)
    x, z, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    
    xzt = tf.concat((x, z, t), axis=1)
    phi = wave_net(xzt)
    
    xz = tf.concat((x, z), axis=1)
    alpha_star = alpha_net(xz)
    alpha, _ = compute_velocity_with_mask(x, z, alpha_star, z_st, z_fi, x_st, x_fi, Lx, Lz, lld=lld_smooth)
    
    return phi, alpha


def compute_model_outputs_with_derivatives(inputs):
    """
    Compute model outputs and derivatives for PINN loss calculation.
    Memory-efficient version for large batches using chunking for second derivatives.
    """
    inputs = tf.cast(inputs, tf.float32)
    batch_size = tf.shape(inputs)[0]
    
    # First pass: compute П† and first derivatives
    # CRITICAL: Extract x, z, t INSIDE tape context for proper gradient flow!
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(inputs)
        
        x = inputs[:, 0:1]
        z = inputs[:, 1:2]
        t = inputs[:, 2:3]
        
        xzt = tf.concat((x, z, t), axis=1)
        phi = wave_net(xzt)
        
        xz = tf.concat((x, z), axis=1)
        alpha_star = alpha_net(xz)
        alpha, _ = compute_velocity_with_mask(x, z, alpha_star, z_st, z_fi, x_st, x_fi, Lx, Lz, lld=lld_smooth)
    
    # Compute first derivatives
    grad_phi = tape1.gradient(phi, inputs)
    
    if grad_phi is None:
        raise RuntimeError("grad_phi is None! Check if inputs are properly connected to phi.")
    
    # Now extract components from grad_phi
    ux = grad_phi[:, 0:1]
    uz = grad_phi[:, 1:2]
    ut = grad_phi[:, 2:3]
    
    # Second derivatives - process in chunks to save memory
    # Key optimization: Compute each second derivative type SEPARATELY across all chunks
    # This ensures only ONE computation graph is in memory at a time (not three)
    # Process 8000 PDE samples in 8000-sample chunks (8000 / 8000 = 1 chunk)
    chunk_size = chunk_size_runtime
    if batch_size > chunk_size:
        # Delete original tape early to free first-derivative memory
        del tape1
        gc.collect()
        
        # ===== COMPUTE в€‚ВІП†/в€‚xВІ FOR ALL CHUNKS =====
        uxx_list = []
        for chunk_idx, i in enumerate(range(0, batch_size, chunk_size)):
            end_idx = min(i + chunk_size, batch_size)
            inputs_chunk = inputs[i:end_idx]
            
            # в€‚ВІП†/в€‚xВІ - nested tapes
            with tf.GradientTape() as tape_outer:
                tape_outer.watch(inputs_chunk)
                with tf.GradientTape() as tape_inner:
                    tape_inner.watch(inputs_chunk)
                    xzt_chunk = tf.concat((inputs_chunk[:, 0:1], inputs_chunk[:, 1:2], inputs_chunk[:, 2:3]), axis=1)
                    phi_chunk = wave_net(xzt_chunk)
                grad_phi_chunk = tape_inner.gradient(phi_chunk, inputs_chunk)
                ux_chunk = grad_phi_chunk[:, 0:1]
            
            uxx_chunk = tape_outer.gradient(ux_chunk, inputs_chunk)
            uxx_list.append(uxx_chunk[:, 0:1] if uxx_chunk is not None else tf.zeros_like(inputs_chunk[:, 0:1]))
            
            del tape_outer, tape_inner, grad_phi_chunk, ux_chunk, uxx_chunk, xzt_chunk, phi_chunk, inputs_chunk
            gc.collect()
        
        uxx = tf.concat(uxx_list, axis=0)
        del uxx_list
        gc.collect()
        gc.collect()  # Double collection to help fragmentation
        
        # ===== COMPUTE в€‚ВІП†/в€‚zВІ FOR ALL CHUNKS =====
        uzz_list = []
        for chunk_idx, i in enumerate(range(0, batch_size, chunk_size)):
            end_idx = min(i + chunk_size, batch_size)
            inputs_chunk = inputs[i:end_idx]
            
            # в€‚ВІП†/в€‚zВІ - nested tapes
            with tf.GradientTape() as tape_outer:
                tape_outer.watch(inputs_chunk)
                with tf.GradientTape() as tape_inner:
                    tape_inner.watch(inputs_chunk)
                    xzt_chunk = tf.concat((inputs_chunk[:, 0:1], inputs_chunk[:, 1:2], inputs_chunk[:, 2:3]), axis=1)
                    phi_chunk = wave_net(xzt_chunk)
                grad_phi_chunk = tape_inner.gradient(phi_chunk, inputs_chunk)
                uz_chunk = grad_phi_chunk[:, 1:2]
            
            uzz_chunk = tape_outer.gradient(uz_chunk, inputs_chunk)
            uzz_list.append(uzz_chunk[:, 1:2] if uzz_chunk is not None else tf.zeros_like(inputs_chunk[:, 1:2]))
            
            del tape_outer, tape_inner, grad_phi_chunk, uz_chunk, uzz_chunk, xzt_chunk, phi_chunk, inputs_chunk
            gc.collect()
        
        uzz = tf.concat(uzz_list, axis=0)
        del uzz_list
        gc.collect()
        gc.collect()  # Double collection to help fragmentation
        
        # ===== COMPUTE в€‚ВІП†/в€‚tВІ FOR ALL CHUNKS =====
        utt_list = []
        for chunk_idx, i in enumerate(range(0, batch_size, chunk_size)):
            end_idx = min(i + chunk_size, batch_size)
            inputs_chunk = inputs[i:end_idx]
            
            # в€‚ВІП†/в€‚tВІ - nested tapes
            with tf.GradientTape() as tape_outer:
                tape_outer.watch(inputs_chunk)
                with tf.GradientTape() as tape_inner:
                    tape_inner.watch(inputs_chunk)
                    xzt_chunk = tf.concat((inputs_chunk[:, 0:1], inputs_chunk[:, 1:2], inputs_chunk[:, 2:3]), axis=1)
                    phi_chunk = wave_net(xzt_chunk)
                grad_phi_chunk = tape_inner.gradient(phi_chunk, inputs_chunk)
                ut_chunk = grad_phi_chunk[:, 2:3]
            
            utt_chunk = tape_outer.gradient(ut_chunk, inputs_chunk)
            utt_list.append(utt_chunk[:, 2:3] if utt_chunk is not None else tf.zeros_like(inputs_chunk[:, 2:3]))
            
            del tape_outer, tape_inner, grad_phi_chunk, ut_chunk, utt_chunk, xzt_chunk, phi_chunk, inputs_chunk
            gc.collect()
        
        utt = tf.concat(utt_list, axis=0)
        del utt_list
        gc.collect()
    else:
        # Original code for small batches (batch <= 200) - use nested tapes without chunking
        # в€‚ВІП†/в€‚xВІ
        with tf.GradientTape() as tape_outer_ux:
            tape_outer_ux.watch(inputs)
            with tf.GradientTape() as tape_inner_ux:
                tape_inner_ux.watch(inputs)
                xzt2 = tf.concat((inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]), axis=1)
                phi2 = wave_net(xzt2)
            grad_phi2 = tape_inner_ux.gradient(phi2, inputs)
            ux2 = grad_phi2[:, 0:1]
        
        uxx = tape_outer_ux.gradient(ux2, inputs)
        uxx = uxx[:, 0:1] if uxx is not None else tf.zeros_like(inputs[:, 0:1])
        
        # в€‚ВІП†/в€‚zВІ
        with tf.GradientTape() as tape_outer_uz:
            tape_outer_uz.watch(inputs)
            with tf.GradientTape() as tape_inner_uz:
                tape_inner_uz.watch(inputs)
                xzt3 = tf.concat((inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]), axis=1)
                phi3 = wave_net(xzt3)
            grad_phi3 = tape_inner_uz.gradient(phi3, inputs)
            uz3 = grad_phi3[:, 1:2]
        
        uzz = tape_outer_uz.gradient(uz3, inputs)
        uzz = uzz[:, 1:2] if uzz is not None else tf.zeros_like(inputs[:, 1:2])
        
        # в€‚ВІП†/в€‚tВІ
        with tf.GradientTape() as tape_outer_ut:
            tape_outer_ut.watch(inputs)
            with tf.GradientTape() as tape_inner_ut:
                tape_inner_ut.watch(inputs)
                xzt4 = tf.concat((inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]), axis=1)
                phi4 = wave_net(xzt4)
            grad_phi4 = tape_inner_ut.gradient(phi4, inputs)
            ut4 = grad_phi4[:, 2:3]
        
        utt = tape_outer_ut.gradient(ut4, inputs)
        utt = utt[:, 2:3] if utt is not None else tf.zeros_like(inputs[:, 2:3])
    
    # PDE with external force:
    #   О±ВІв€‡ВІП† + f = в€‚ВІП†/в€‚tВІ  ->  residual = в€‚ВІП†/в€‚tВІ - О±ВІв€‡ВІП† - f
    P = (1.0/Lx)**2 * uxx + (1.0/Lz)**2 * uzz
    forcing = source_force_tf(x, z, t)
    eq = utt - (alpha**2) * P - forcing
    
    return phi, alpha, grad_phi, ux, uz, ut, uxx, uzz, utt, eq, P

# Create optimizer with learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# ========== CREATE THE TWO SEPARATE NETWORKS ==========
# Pass normalization constants to networks for proper gradient tracking
wave_net = WaveNet(layers, ub)
alpha_net = AlphaNet(layers0, ub0)

# Combine trainable variables from both networks
all_trainable_vars = lambda: wave_net.trainable_variables + alpha_net.trainable_variables

num_wave_vars = len(wave_net.trainable_variables)
num_alpha_vars = len(alpha_net.trainable_variables)
num_weights_total = num_wave_vars + num_alpha_vars

print(f"[DEBUG] Networks created successfully:")
print(f"  * WaveNet (phi):   {num_wave_vars} trainable variables (8 layers x 100 neurons)")
print(f"  * AlphaNet (alpha):  {num_alpha_vars} trainable variables (5 layers x 20 neurons)")
print(f"  * TOTAL:         {num_weights_total} trainable variables")

if num_weights_total == 0:
    raise RuntimeError("FATAL: Total trainable variables = 0! Something is wrong.")

# Checkpoint manager
training_epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False, name='training_epoch')
checkpoint = tf.train.Checkpoint(wave_net=wave_net, alpha_net=alpha_net, optimizer=optimizer, training_epoch=training_epoch_var)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=3)

def safe_save_checkpoint(tag):
    try:
        checkpoint_manager.save()
        print(f'Checkpoint saved ({tag})')
    except Exception as exc:
        print(f'[WARNING] Checkpoint save skipped at {tag}: {exc}')

# Load checkpoint if specified
start_epoch = max(0, int(args.resume_epoch)) if args.resume_epoch is not None else 0
if args.load_checkpoint:
    status = checkpoint.restore(args.load_checkpoint)
    status.expect_partial()
    print(f"Loaded checkpoint from {args.load_checkpoint}")

    restored_epoch = int(training_epoch_var.numpy())
    inferred_epoch = infer_start_epoch_from_checkpoint_path(args.load_checkpoint, args.checkpoint_every)
    if args.resume_epoch is not None:
        start_epoch = max(0, int(args.resume_epoch))
        print(f"[RESUME] Using explicit --resume-epoch={start_epoch}")
    elif restored_epoch > 0:
        start_epoch = restored_epoch
        print(f"[RESUME] Restored epoch from checkpoint variable: {start_epoch}")
    elif inferred_epoch > 0:
        start_epoch = inferred_epoch
        print(f"[RESUME] Inferred epoch from checkpoint filename: {start_epoch}")
    else:
        print("[RESUME] Could not infer resume epoch; starting from 0.")

# Training data preparation (same as before)
### PDE residuals
n_pde=max(args.n_pde_total, batch_size)
print('batch_size',':',batch_size)
print('chunk_size_runtime',':',chunk_size_runtime)
print(f'n_pde: {n_pde} (generating Sobol samples...)')
X_pde = sobol_sequence.sample(n_pde+1, 3)[1:,:]
X_pde[:,0] = X_pde[:,0] * ax/Lx
X_pde[:,1] = X_pde[:,1] * az/Lz
X_pde[:,2] = X_pde[:,2] * (t_m-t_st)

# ── Сетка для IC (n_ini × n_ini равномерно по домену) ─────────
n_ini = 60  # was 40 → 3600 IC points (was 1600) for better domain coverage
from scipy.interpolate import RegularGridInterpolator

xx, zz = np.meshgrid(np.linspace(0, ax/Lx, n_ini),
                     np.linspace(0, az/Lz, n_ini))
xxzz  = np.concatenate((xx.reshape((-1,1)), zz.reshape((-1,1))), axis=1)
xxzzs = xxzz.copy()   # та же сетка (нет поглощающих регионов в .npy данных)

X_init1 = np.concatenate((xx.reshape((-1,1)), zz.reshape((-1,1)),
           0.0           * np.ones((n_ini**2, 1), dtype=np.float32)), axis=1)
X_init2 = np.concatenate((xx.reshape((-1,1)), zz.reshape((-1,1)),
           (t02 - t01)   * np.ones((n_ini**2, 1), dtype=np.float32)), axis=1)


# =====================================================
# NORMALIZATION CRITICAL FOR PINN CONVERGENCE
# =====================================================
# Neural networks are initialized to work with inputs and outputs in [-1, 1] range.
# Without normalization, gradients have different orders of magnitude:
#  - в€‚u/в€‚t vs в€‚u/в€‚x have wildly different scales в†’ unstable optimization
#  - Loss contributions become unbalanced
#
# Strategy:
# 1. INPUT SCALING: x, z, t are scaled to O(1)
# 2. OUTPUT SCALING: displacement targets are normalized to O(1)
# 3. Loss weights stay simple and interpretable, similar to case 3:
#    data terms dominate, PDE/BC regularize
# 4. Physics scaling: when computing PDE, account for coordinate transformation
#    (derivatives are в€‚/в€‚x_norm, not в€‚/в€‚x_physical)
# =====================================================
import os

# Загружаем координатную сетку SpecFEM2D (x, z для каждой точки)
grid_raw = np.loadtxt('event1/wavefields/wavefield_grid_for_dumps_000.txt')
xz = grid_raw[:, 0:2]   # колонки 0=x, 1=z (в метрах → нормируем ниже)
# Переводим физические координаты сетки из метров в km и затем нормируем на Lx, Lz
xz_norm = np.column_stack((xz[:, 0] / 1000.0 / Lx,
                            xz[:, 1] / 1000.0 / Lz))

# Загружаем wavefield-снапшоты: IC1=step2000(t=0.10s), IC2=step4000(t=0.20s), Test=step5000(t=0.25s)
# Есть 5 файлов — выбираем ровно три нужных по номеру шага
WF_STEPS = ['wavefield0002000_01_000.txt',   # t_ic1 = 0.10 s
             'wavefield0004000_01_000.txt',   # t_ic2 = 0.20 s
             'wavefield0005000_01_000.txt']   # t_la  = 0.25 s (test snapshot)
wfs = [f for f in WF_STEPS if os.path.exists('event1/wavefields/'+f)]
if len(wfs) != 3:
    raise FileNotFoundError(f'Expected 3 wavefield files (steps 2000/4000/5000) in event1/wavefields, got {wfs}')
print(f'[DATA] Wavefield files for event1: {wfs}')
U0_raw = [np.loadtxt('event1/wavefields/'+f) for f in wfs]
U0 = U0_raw
xz = xz_norm  # нормированные координаты [0..ax/Lx] x [0..az/Lz]

U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
U_ini1x = U_ini1[:,0:1]
U_ini1z = U_ini1[:,1:2]


U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
U_ini2x = U_ini2[:,0:1]
U_ini2z = U_ini2[:,1:2]

U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)#Test data
U_specx = U_spec[:,0:1]
U_specz = U_spec[:,1:2]





#the first event's data has been uploaded above and below
#the rest of the n-1 events will be added
for ii in range(n_event-1):
    ev_dir = 'event'+str(ii+2)+'/wavefields'
    # Загрузка сетки для текущего события (если отличается от event1)
    grid_raw_ev = np.loadtxt(ev_dir+'/wavefield_grid_for_dumps_000.txt')
    xz_ev = np.column_stack((grid_raw_ev[:,0]/1000.0/Lx,
                              grid_raw_ev[:,1]/1000.0/Lz))
    wfs_ev = [f for f in WF_STEPS if os.path.exists(ev_dir+'/'+f)]
    if len(wfs_ev) != 3:
        raise FileNotFoundError(f'Expected 3 wavefield files in {ev_dir}, got {wfs_ev}')
    U0 = [np.loadtxt(ev_dir+'/'+f) for f in wfs_ev]

    U_ini1 = interpolate.griddata(xz_ev, U0[0], xxzzs, fill_value=0.0)
    U_ini1x += U_ini1[:,0:1]
    U_ini1z += U_ini1[:,1:2]


    U_ini2 = interpolate.griddata(xz_ev, U0[1], xxzzs, fill_value=0.0)
    U_ini2x += U_ini2[:,0:1]
    U_ini2z += U_ini2[:,1:2]

    U_spec = interpolate.griddata(xz_ev, U0[2], xxzzs, fill_value=0.0)
    U_specx += U_spec[:,0:1]
    U_specz += U_spec[:,1:2]
#U_ini=U_ini.reshape(-1,1)



################### plots of inputs for sum of the events
fig = plt.figure()
plt.contourf(xx*Lx, zz*Lz, np.sqrt(U_ini1x**2+U_ini1z**2).reshape(xx.shape),100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Scaled I.C total disp. input specfem t='+str(t01))
plt.colorbar()
plt.axis('scaled')
save_figure('Ini_total_disp_spec_sumEvents.png', dpi=400)
# plt.show()

fig = plt.figure()
plt.contourf(xx*Lx, zz*Lz, np.sqrt(U_ini2x**2+U_ini2z**2).reshape(xx.shape),100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Scaled sec I.C total disp. input specfem t='+str(round(t02, 4)))
plt.colorbar()
plt.axis('scaled')
save_figure('sec_wavefield_input_spec_sumEvents.png', dpi=400)
# plt.show()

fig = plt.figure()
U_spec_test_mag = np.sqrt(U_specx**2 + U_specz**2).reshape(xx.shape)
disp_plot_vmin = float(np.min(U_spec_test_mag))
disp_plot_vmax = float(np.max(U_spec_test_mag))

plt.contourf(xx*Lx, zz*Lz, U_spec_test_mag, 100, cmap='jet', vmin=disp_plot_vmin, vmax=disp_plot_vmax)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Test data: Total displacement specfem t='+str(round((t_la-t01), 4)))
plt.colorbar()
plt.axis('scaled')
save_figure('total_disp_spec_testData_sumEvents.png', dpi=400)
# plt.show()
###############################################################



################# ----Z component seismograms
#################input seismograms for the first event


import os
sms = sorted(os.listdir('event1/seismograms'))
smsz = [f for f in sms if 'BXZ' in f]  # Z-компонента: AA.S000N.BXZ.semd
seismo_listz = [np.loadtxt('event1/seismograms/'+f) for f in smsz]  # Z cmp seismos
print(f'[DATA] Z-seismogram files found for event1: {len(smsz)} stations')

t_spec=-seismo_listz[0][0,0]+seismo_listz[0][:,0]#specfem's time doesn't start from zero for the seismos, so we shift it forward to zero
cut_u=t_spec>t_s#here we include only part of the seismograms from specfem that are within PINNs' training time domain which is [t_st t_m]
cut_l=t_spec<t_st#Cutting the seismograms to only after the time the first snapshot from specfem is used for PINNs
l_su=len(cut_u)-sum(cut_u)#this is the index of the time axis in specfem after which t>t_m
l_sl=sum(cut_l)




l_f=50  # was 100 — now 2x more time points per receiver for denser seismic constraint
index = np.arange(l_sl,l_su,l_f) #subsampling every l_s time steps from specfem in the training interval
l_sub=len(index)
t_spec_sub=t_spec[index].reshape((-1,1))#subsampled time axis of specfem for the seismograms

t_spec_sub=t_spec_sub-t_spec_sub[0]#shifting the time axis back to zero. length of t_spec_sub must be equal to t_m-t_st




for ii in range(len(seismo_listz)):
    seismo_listz[ii]=seismo_listz[ii][index]



Sz=seismo_listz[0][:,1].reshape(-1,1)
for ii in range(len(seismo_listz)-1):
    Sz=np.concatenate((Sz,seismo_listz[ii+1][:,1].reshape(-1,1)),axis=0)


#################################################################
#######input seismograms for the rest of the events added to the first event

for ii in range(n_event-1):
    sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms'))
    smsz = [f for f in sms if 'BXZ' in f]  # Z-компонента
    seismo_listz = [np.loadtxt('event'+str(ii+2)+'/seismograms/'+f) for f in smsz]

    for jj in range(len(seismo_listz)):
        seismo_listz[jj]=seismo_listz[jj][index]


    Sze=seismo_listz[0][:,1].reshape(-1,1)
    for jj in range(len(seismo_listz)-1):
       Sze=np.concatenate((Sze,seismo_listz[jj+1][:,1].reshape(-1,1)),axis=0)

    Sz +=Sze
###########################################################


print(f"\n[INFO] Z-component seismogram raw loading complete. Sz shape: {Sz.shape}")

# X_S: координаты датчиков в пространстве-времени.
# Датчики расположены по оси X (linspace 100..1400 м), глубина z постоянная = z_seis_phys.
# z0_s, zl_s — нормированная z-координата линейки датчиков (все на одной глубине).
z0_s = z_seis_phys / Lz   # нормированная z-координата датчиков (≈ 0.1383)
zl_s = z0_s               # все датчики на одной глубине → d_s = 0

# x-координаты датчиков: равномерно от 100 м до 1400 м (в km → нормируем на Lx)
X_SEIS_M = np.linspace(100.0, 1400.0, n_seis)  # в метрах
X_SEIS_NORM = X_SEIS_M / 1000.0 / Lx           # нормированные x

X_S = np.empty([int(np.size(Sz)), 3])

for i in range(len(seismo_listz)):
    X_S[i*l_sub:(i+1)*l_sub, :] = np.concatenate((
        X_SEIS_NORM[i] * np.ones((l_sub, 1), dtype=np.float32),
        z0_s           * np.ones((l_sub, 1), dtype=np.float32),
        t_spec_sub
    ), axis=1)






################# ----X component seismograms
#################input seismograms for the first event


import os
sms = sorted(os.listdir('event1/seismograms'))
smsx = [f for f in sms if 'BXX' in f]  # X-компонента: AA.S000N.BXX.semd
seismo_listx = [np.loadtxt('event1/seismograms/'+f) for f in smsx]  # X cmp seismos
print(f'[DATA] X-seismogram files found for event1: {len(smsx)} stations')


for ii in range(len(seismo_listx)):
    seismo_listx[ii]=seismo_listx[ii][index]



Sx=seismo_listx[0][:,1].reshape(-1,1)
for ii in range(len(seismo_listx)-1):
    Sx=np.concatenate((Sx,seismo_listx[ii+1][:,1].reshape(-1,1)),axis=0)

#################################################################
#######input seismograms for the rest of the events added to the first event

for ii in range(n_event-1):
    sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms'))
    smsx = [f for f in sms if 'BXX' in f]  # X-компонента
    seismo_listx = [np.loadtxt('event'+str(ii+2)+'/seismograms/'+f) for f in smsx]

    for jj in range(len(seismo_listx)):
        seismo_listx[jj]=seismo_listx[jj][index]



    Sxe=seismo_listx[0][:,1].reshape(-1,1)
    for jj in range(len(seismo_listx)-1):
       Sxe=np.concatenate((Sxe,seismo_listx[jj+1][:,1].reshape(-1,1)),axis=0)

    Sx +=Sxe
###########################################################


print(f"[INFO] X-component seismogram raw loading complete. Sx shape: {Sx.shape}")


def _max_abs(arr):
    return float(np.max(np.abs(arr))) if arr.size else 0.0

def _rms(arr):
    return float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0


# FIXED: Normalize by RMS not max.
# Problem with max-norm: wave arrivals are sparse → RMS/max ≈ 0.03 → targets ≈ 0
# → gradient signal too weak → network stuck at trivial phi=0.
# RMS-norm: target RMS = 1.0 → 30x stronger gradient → network escapes zero.
all_nonzero = np.concatenate([
    U_ini1x.ravel(), U_ini1z.ravel(),
    U_ini2x.ravel(), U_ini2z.ravel(),
    Sx.ravel(), Sz.ravel()
])
disp_scale = max(_rms(all_nonzero[np.abs(all_nonzero) > 0]), 1e-12)

print(f"[NORM] Using RMS normalization. disp_scale (RMS of nonzero)={disp_scale:.6e}")
print(f"[NORM] Max/RMS ratio = {_max_abs(all_nonzero)/disp_scale:.1f}  "
      f"(was: max-norm gave RMS≈0.03, now RMS≈1.0)")

U_ini1x = U_ini1x / disp_scale
U_ini1z = U_ini1z / disp_scale
U_ini2x = U_ini2x / disp_scale
U_ini2z = U_ini2z / disp_scale
U_specx = U_specx / disp_scale
U_specz = U_specz / disp_scale
Sx = Sx / disp_scale
Sz = Sz / disp_scale

data_norm_info = {
    'target_scale': 'rms_nonzero',
    'disp_scale': float(disp_scale),
    'applies_to': ['U_ini1', 'U_ini2', 'U_spec', 'Sx', 'Sz']
}

print(f"[OK] Unified target normalization applied. disp_scale={disp_scale:.6e}")
print(f"[OK] Z-component seismogram normalized. Sz shape: {Sz.shape}")
print(f"[OK] X-component seismogram normalized. Sx shape: {Sx.shape}")

####  BCs: Free stress on top and no BC for other sides (absorbing)
bcxn=100
bctn=50
num_bc_points = 5000  # Case 4: 5,000 BC data points
bc_samples_per_grid = num_bc_points // (bcxn * bctn)  # Repeat sampling to get 5000 points
x_vec = np.random.rand(bcxn,1)*ax/Lx
t_vec = np.random.rand(bctn,1)*(t_m-t_st)
xxb, ttb = np.meshgrid(x_vec, t_vec)
X_BC_base = np.concatenate((xxb.reshape((-1,1)),az/Lz*np.ones((xxb.reshape((-1,1)).shape[0],1)),ttb.reshape((-1,1))),axis=1)
# Replicate to reach 5000 points
if len(X_BC_base) < num_bc_points:
    num_repeats = (num_bc_points // len(X_BC_base)) + 1
    X_BC_t = np.vstack([X_BC_base for _ in range(num_repeats)])[:num_bc_points]
else:
    X_BC_t = X_BC_base[:num_bc_points]
print(f'BC points: {X_BC_t.shape[0]}')

def sample_bc_batch():
    x_vec = np.random.rand(bcxn,1)*ax/Lx
    t_vec = np.random.rand(bctn,1)*(t_m-t_st)
    xxb, ttb = np.meshgrid(x_vec, t_vec)
    X_BC_base = np.concatenate(
        (
            xxb.reshape((-1,1)),
            az/Lz*np.ones((xxb.reshape((-1,1)).shape[0],1)),
            ttb.reshape((-1,1))
        ),
        axis=1
    )
    if len(X_BC_base) < num_bc_points:
        num_repeats = (num_bc_points // len(X_BC_base)) + 1
        return np.vstack([X_BC_base for _ in range(num_repeats)])[:num_bc_points]
    return X_BC_base[:num_bc_points]

# Define data sizes BEFORE using them
N1 = batch_size
N2 = X_init1.shape[0]
N3 = X_init2.shape[0]
N4 = X_S.shape[0]

print(f"\nData sizes defined:")
print(f"  N1 (batch_size): {N1}")
print(f"  N2 (X_init1): {N2}")
print(f"  N3 (X_init2): {N3}")
print(f"  N4 (X_S): {N4}")

# Prepare separate data arrays for each component
print(f"\n[...] Preparing training data...")

# Split the data into separate numpy arrays
XX_pde_np = X_pde[0:N1]
XX_ic1_np = X_init1
XX_ic2_np = X_init2
XX_seis_np = X_S
XX_bc_np = X_BC_t

print(f"[OK] Training data prepared:")
print(f"  PDE={XX_pde_np.shape}, IC1={XX_ic1_np.shape}, IC2={XX_ic2_np.shape}, Seis={XX_seis_np.shape}, BC={XX_bc_np.shape}")

def get_pde_batch(epoch):
    """Random PDE batch — avoids correlated sequential sweeps through Sobol sequence."""
    if n_pde <= batch_size:
        return X_pde[:batch_size]
    idx = np.random.choice(n_pde, batch_size, replace=False)
    return X_pde[idx]

def build_training_batch(epoch):
    XX_pde_epoch = get_pde_batch(epoch)
    XX_bc_epoch = sample_bc_batch()
    return np.concatenate((XX_pde_epoch, XX_ic1_np, XX_ic2_np, XX_seis_np, XX_bc_epoch), axis=0)

# Training data - current epoch batch plus static IC / seismic / BC blocks
XX_unified = build_training_batch(start_epoch)

# Combine IC data (IC1 and IC2 concatenated) - kept for PDE/seis reference
U_ic1x_combined = tf.constant(np.concatenate([U_ini1x, U_ini2x], axis=0), dtype=tf.float32)
U_ic1z_combined = tf.constant(np.concatenate([U_ini1z, U_ini2z], axis=0), dtype=tf.float32)

# Seismogram data as constants
U_seisx = tf.constant(Sx, dtype=tf.float32)
U_seisz = tf.constant(Sz, dtype=tf.float32)

# ─── DIRECT PHI TARGETS via integration (FIRST-ORDER gradient!) ─────────────
# WHY: IC loss = MSE(∂φ/∂x - ux_data) requires SECOND-ORDER gradient w.r.t.
#      network weights through 8 tanh layers → vanishing gradient → loss stuck.
# FIX: integrate ux_data over x to get phi_data, then use MSE(phi - phi_data)
#      which only needs FIRST-ORDER gradient → 30× stronger signal.
# Integration: φ(x,z) ≈ φ(0,z) + ∫₀ˣ ux(x',z) dx'  (cumulative trapezoid)
dx_norm = (ax/Lx) / max(n_ini - 1, 1)   # grid spacing in normalized x

def integrate_phi_from_ux(ux_2d, dz_norm_unused):
    """Integrate ux over x-axis (axis=1) to get phi. Shape: (nz, nx) -> (nz, nx)."""
    return np.cumsum(ux_2d, axis=1) * dx_norm  # cumulative sum × dx

# IC1 phi target
U_ini1x_grid = U_ini1x.reshape(n_ini, n_ini)   # (nz, nx)
phi_ic1_np   = integrate_phi_from_ux(U_ini1x_grid, dx_norm).reshape(-1, 1)

# IC2 phi target
U_ini2x_grid = U_ini2x.reshape(n_ini, n_ini)
phi_ic2_np   = integrate_phi_from_ux(U_ini2x_grid, dx_norm).reshape(-1, 1)

phi_ic1_tf = tf.constant(phi_ic1_np, dtype=tf.float32)
phi_ic2_tf = tf.constant(phi_ic2_np, dtype=tf.float32)
print(f"[IC-PHI] phi_ic1 range: [{phi_ic1_np.min():.3e}, {phi_ic1_np.max():.3e}]")
print(f"[IC-PHI] phi_ic2 range: [{phi_ic2_np.min():.3e}, {phi_ic2_np.max():.3e}]")
# ─────────────────────────────────────────────────────────────────────────────

# ─── SEISMOGRAM PHI TARGETS via spatial integration (FIRST-ORDER gradient!) ──
# 20 receivers at different x, same z=z_seis, give ux(x_i, z_r, t_j) = Sx[i*l_sub+j]
# Integrate Sx over x: phi_rel(x_i, t) = integral_{x_0}^{x_i} ux dx
# This is RELATIVE phi (phi[i,t] - phi[0,t]) → first-order, no vanishing gradient!
# NOTE: absolute phi at x_0 is anchored by IC loss; seis loss constrains spatial variation.
dx_seis_norm = float(np.mean(np.diff(X_SEIS_NORM)))   # spacing between receivers (normalized)
Sx_grid_np   = Sx.reshape(n_seis, l_sub)               # (n_seis, l_sub)

# Trapezoidal integration: phi_rel[0]=0, phi_rel[i] = sum of trapezoids
phi_seis_rel_np = np.zeros_like(Sx_grid_np)            # shape (n_seis, l_sub)
for i in range(1, n_seis):
    phi_seis_rel_np[i] = (phi_seis_rel_np[i-1]
                          + 0.5 * (Sx_grid_np[i-1] + Sx_grid_np[i]) * dx_seis_norm)

phi_seis_rel_tf = tf.constant(phi_seis_rel_np.reshape(-1, 1), dtype=tf.float32)  # (N4,1)
print(f"[SEIS-PHI] phi_seis_rel range: [{phi_seis_rel_np.min():.3e}, {phi_seis_rel_np.max():.3e}]")
# ─────────────────────────────────────────────────────────────────────────────

# Generate initial plots (True wavespeed and Initial guess)
print(f"\n[INIT-VIZ] Generating initial reference plots...")
try:
    # True wavespeed with seismometer locations marked
    x_grid = xxzzs[:, 0:1]
    z_grid = xxzzs[:, 1:2]
    alpha_true_grid = compute_alpha_true_numpy(x_grid, z_grid).reshape(xx.shape)
    
    fig = plt.figure(figsize=(10, 8))
    plt.contourf(xx*Lx, zz*Lz, alpha_true_grid, 100, cmap='jet')
    plt.xlabel('x (km)')
    plt.ylabel('z (km)')
    plt.title('True Acoustic Wave Speed (alpha)')
    plt.colorbar(label='Wave Speed (km/s)')
    # Рисуем 20 уникальных позиций датчиков: x меняется (100..1400 м), z постоянная
    plt.plot(X_SEIS_M / 1000.0, z_seis_phys * np.ones(n_seis), 'r*', markersize=8, label='Seismometers')
    plt.axis('scaled')
    plt.legend()
    save_figure('True_wavespeed.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [INIT-VIZ] True_wavespeed.png saved")
    
    # Initial alpha guess (before training)
    feed_dict_init = np.concatenate((xxzzs[:, 0:1], xxzzs[:, 1:2], 0.0*np.ones((xxzzs.shape[0], 1), dtype=np.float32)), axis=1)
    _, alpha_init, _, _, _, _, _, _, _, _, _ = compute_model_outputs_with_derivatives(tf.convert_to_tensor(feed_dict_init, dtype=tf.float32))
    alpha_init_grid = alpha_init.numpy().reshape(xx.shape)
    
    fig = plt.figure(figsize=(10, 8))
    plt.contourf(xx*Lx, zz*Lz, alpha_init_grid, 100, cmap='jet')
    plt.xlabel('x (km)')
    plt.ylabel('z (km)')
    plt.title('Initial Guess: Wave Speed (alpha) - Epoch 0')
    plt.colorbar(label='Wave Speed (km/s)')
    plt.axis('scaled')
    save_figure('Ini_guess_wavespeed.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [INIT-VIZ] Ini_guess_wavespeed.png saved")
except Exception as e:
    print(f"  [WARNING] Initial visualization error: {e}")

# Training step with normalized loss weighting
# CORRECT PINN implementation: separate networks with proper gradient flow
def train_step(inputs_pde, inputs_ic1, inputs_ic2, inputs_seis, inputs_bc, epoch_weights):
    """
    Restructured train_step with SEPARATE tapes for each loss type.

    ROOT CAUSE of previous failure:
      The single combined tape deleted tape1 (for memory) BEFORE tape_weights
      could compute d(grad_phi)/dW → second-order gradient = 0 → loss stuck.

    FIX (mirrors original TF1 symbolic approach):
      - IC tape: kept alive so tape_weights can compute d(∂phi/∂x)/dW
      - Seis tape: same
      - PDE tape: chunked (can delete early, only needs alpha grads)
    """
    inputs_pde  = tf.cast(inputs_pde,  tf.float32)
    inputs_ic1  = tf.cast(inputs_ic1,  tf.float32)
    inputs_ic2  = tf.cast(inputs_ic2,  tf.float32)
    inputs_seis = tf.cast(inputs_seis, tf.float32)
    inputs_bc   = tf.cast(inputs_bc,   tf.float32)

    all_vars = wave_net.trainable_variables + alpha_net.trainable_variables

    with tf.GradientTape() as tape_weights:

        # ── 1-2. IC losses: primary first-order phi matching + auxiliary grad matching ──
        with tf.GradientTape() as tape_ic1:
            tape_ic1.watch(inputs_ic1)
            phi_ic1_pred = wave_net(inputs_ic1)
        grad_ic1 = tape_ic1.gradient(phi_ic1_pred, inputs_ic1)
        loss_ic1_phi = tf.reduce_mean(tf.square(phi_ic1_pred - phi_ic1_tf))
        loss_ic1_grad = (tf.reduce_mean(tf.square(grad_ic1[:, 0:1] - U_ini1x)) +
                         tf.reduce_mean(tf.square(grad_ic1[:, 1:2] - U_ini1z)))
        loss_ic1 = float(args.w_ic_phi) * loss_ic1_phi + float(args.w_ic_grad) * loss_ic1_grad

        with tf.GradientTape() as tape_ic2:
            tape_ic2.watch(inputs_ic2)
            phi_ic2_pred = wave_net(inputs_ic2)
        grad_ic2 = tape_ic2.gradient(phi_ic2_pred, inputs_ic2)
        loss_ic2_phi = tf.reduce_mean(tf.square(phi_ic2_pred - phi_ic2_tf))
        loss_ic2_grad = (tf.reduce_mean(tf.square(grad_ic2[:, 0:1] - U_ini2x)) +
                         tf.reduce_mean(tf.square(grad_ic2[:, 1:2] - U_ini2z)))
        loss_ic2 = float(args.w_ic_phi) * loss_ic2_phi + float(args.w_ic_grad) * loss_ic2_grad

        loss_ic = loss_ic1 + loss_ic2

        # ── 3. Seis losses: relative phi matching across receivers + auxiliary grad matching ──
        with tf.GradientTape() as tape_seis:
            tape_seis.watch(inputs_seis)
            phi_seis_pred = wave_net(inputs_seis)
        grad_seis = tape_seis.gradient(phi_seis_pred, inputs_seis)
        phi_seis_grid = tf.reshape(phi_seis_pred, [n_seis, l_sub])
        phi_seis_rel_pred = phi_seis_grid - phi_seis_grid[0:1, :]
        phi_seis_rel_true = tf.reshape(phi_seis_rel_tf, [n_seis, l_sub])
        loss_seis_phi = tf.reduce_mean(tf.square(phi_seis_rel_pred - phi_seis_rel_true))
        loss_seis_grad = (tf.reduce_mean(tf.square(grad_seis[:, 0:1] - U_seisx)) +
                          tf.reduce_mean(tf.square(grad_seis[:, 1:2] - U_seisz)))
        loss_seis = float(args.w_seis_phi) * loss_seis_phi + float(args.w_seis_grad) * loss_seis_grad

        # ── 4. PDE loss: chunked (tape can be reused per chunk) ──
        if epoch_weights['w_pde'] > 0.0:
            phi_p, alpha_p, _, _, _, _, uxx_p, uzz_p, utt_p, eq_p, _ = \
                compute_model_outputs_with_derivatives(inputs_pde)
            loss_pde = tf.reduce_mean(tf.square(eq_p))
        else:
            loss_pde = tf.constant(0.0, dtype=tf.float32)

        # ── 5. BC loss: MSE(P at top boundary) ──
        if epoch_weights['w_bc'] > 0.0:
            with tf.GradientTape() as tape_bc2:
                tape_bc2.watch(inputs_bc)
                with tf.GradientTape() as tape_bc1:
                    tape_bc1.watch(inputs_bc)
                    phi_bc = wave_net(inputs_bc)
                gbc = tape_bc1.gradient(phi_bc, inputs_bc)
                ux_bc = gbc[:, 0:1]
            uxx_bc = tape_bc2.gradient(ux_bc, inputs_bc)
            uxx_bc = uxx_bc[:, 0:1] if uxx_bc is not None else tf.zeros_like(inputs_bc[:, 0:1])

            with tf.GradientTape() as tape_bc4:
                tape_bc4.watch(inputs_bc)
                with tf.GradientTape() as tape_bc3:
                    tape_bc3.watch(inputs_bc)
                    phi_bc2 = wave_net(inputs_bc)
                gbc2 = tape_bc3.gradient(phi_bc2, inputs_bc)
                uz_bc = gbc2[:, 1:2]
            uzz_bc = tape_bc4.gradient(uz_bc, inputs_bc)
            uzz_bc = uzz_bc[:, 1:2] if uzz_bc is not None else tf.zeros_like(inputs_bc[:, 1:2])

            P_bc = (1.0/Lx)**2 * uxx_bc + (1.0/Lz)**2 * uzz_bc
            loss_bc = tf.reduce_mean(tf.square(P_bc))
        else:
            loss_bc = tf.constant(0.0, dtype=tf.float32)

        total_loss = (epoch_weights['w_ic']   * loss_ic +
                      epoch_weights['w_seis']  * loss_seis +
                      epoch_weights['w_pde']   * loss_pde +
                      epoch_weights['w_bc']    * loss_bc)

    grads_all = tape_weights.gradient(total_loss, all_vars)
    grads_all = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads_all, all_vars)]
    if args.grad_clip_norm and args.grad_clip_norm > 0:
        grads_all, _ = tf.clip_by_global_norm(grads_all, args.grad_clip_norm)
    optimizer.apply_gradients(zip(grads_all, all_vars))

    return total_loss, loss_pde, loss_ic, loss_seis, loss_bc

# Visualization function for core predictions
def generate_visualizations(epoch, loss_history):
    """Generate prediction plots at current training state."""
    print(f"  [VIZ] Generating visualizations for epoch {epoch}...")
    try:
        # Use the same test-time snapshot as detailed comparisons, so plots are consistent.
        t_eval = (t_la - t01)
        feed_dict_eval = np.concatenate((xxzzs[:, 0:1], xxzzs[:, 1:2], t_eval*np.ones((xxzzs.shape[0], 1), dtype=np.float32)), axis=1)
        feed_tensor = tf.convert_to_tensor(feed_dict_eval, dtype=tf.float32)
        phi, alpha, grad_phi, ux_pred, uz_pred, ut, uxx, uzz, utt, eq, P = compute_model_outputs_with_derivatives(feed_tensor)
        ux_pred = ux_pred.numpy()
        uz_pred = uz_pred.numpy()
        alpha_pred = alpha.numpy()
        U_PINN = np.sqrt(ux_pred**2 + uz_pred**2).reshape(xx.shape)
        
        fig = plt.figure(figsize=(10, 8))
        plt.contourf(xx*Lx, zz*Lz, U_PINN, 100, cmap='jet', vmin=disp_plot_vmin, vmax=disp_plot_vmax)
        plt.xlabel('x (km)')
        plt.ylabel('z (km)')
        plt.title(f'PINN Predicted Displacement at t={t_la:.3f}s (Epoch {epoch})')
        plt.colorbar(label='Displacement')
        plt.axis('scaled')
        save_figure('predicted_displacement.png', dpi=150, bbox_inches='tight')  # Overwrites same file
        plt.close(fig)
        
        fig = plt.figure(figsize=(10, 8))
        plt.contourf(xx*Lx, zz*Lz, alpha_pred.reshape(xx.shape), 100, cmap='jet')
        plt.xlabel('x (km)')
        plt.ylabel('z (km)')
        plt.title(f'Inverted Wave Speed (Epoch {epoch})')
        plt.colorbar(label='Wave Speed (km/s)')
        plt.axis('scaled')
        save_figure('inverted_alpha.png', dpi=150, bbox_inches='tight')  # Overwrites same file
        plt.close(fig)
        
        if len(loss_history) > 0:
            fig = plt.figure(figsize=(12, 6))
            epochs_list = np.array([h[0] for h in loss_history])
            losses = np.array([h[1] for h in loss_history])
            loss_pde = np.array([h[2] for h in loss_history])
            loss_ic = np.array([h[3] for h in loss_history])
            loss_seis = np.array([h[4] for h in loss_history])
            loss_bc = np.array([h[5] for h in loss_history])
            plt.semilogy(epochs_list, losses, 'y--', label='Total Loss', linewidth=2)
            plt.semilogy(epochs_list, loss_pde, 'r-', label='PDE Loss')
            plt.semilogy(epochs_list, loss_ic, 'b-', label='IC Loss')
            plt.semilogy(epochs_list, loss_seis, 'c-', label='Seismic Loss')
            plt.semilogy(epochs_list, loss_bc, 'k-', label='BC Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.title(f'Training Loss History (Epoch {epoch})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            save_figure('loss_history.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f"  [VIZ] Visualizations saved for epoch {epoch}")
    except Exception as e:
        print(f"  [WARNING] Visualization error at epoch {epoch}: {e}")

# Detailed comparison visualizations (every 200 epochs)
def generate_detailed_visualizations(epoch, loss_history):
    """Generate detailed comparison plots: displacement error, alpha misfit, seismograms."""
    print(f"  [VIZ-DETAIL] Generating detailed comparisons for epoch {epoch}...")
    try:
        # Compute predictions on different time snapshots
        feed_dict01 = np.concatenate((xxzzs[:, 0:1], xxzzs[:, 1:2], 0.0*np.ones((xxzzs.shape[0], 1), dtype=np.float32)), axis=1)
        feed_dict02 = np.concatenate((xxzzs[:, 0:1], xxzzs[:, 1:2], (t02-t01)*np.ones((xxzzs.shape[0], 1), dtype=np.float32)), axis=1)
        feed_dictt = np.concatenate((xxzzs[:, 0:1], xxzzs[:, 1:2], (t_la-t01)*np.ones((xxzzs.shape[0], 1), dtype=np.float32)), axis=1)
        
        phi01, alpha01, _, ux01, uz01, _, _, _, _, _, _ = compute_model_outputs_with_derivatives(tf.convert_to_tensor(feed_dict01, dtype=tf.float32))
        phi02, alpha02, _, ux02, uz02, _, _, _, _, _, _ = compute_model_outputs_with_derivatives(tf.convert_to_tensor(feed_dict02, dtype=tf.float32))
        phit, alphat, _, uxt, uzt, _, _, _, _, _, _ = compute_model_outputs_with_derivatives(tf.convert_to_tensor(feed_dictt, dtype=tf.float32))
        
        U_PINN01 = np.sqrt(ux01.numpy()**2 + uz01.numpy()**2).reshape(xx.shape)
        U_PINN02 = np.sqrt(ux02.numpy()**2 + uz02.numpy()**2).reshape(xx.shape)
        U_PINNt = np.sqrt(uxt.numpy()**2 + uzt.numpy()**2).reshape(xx.shape)
        U_diff = np.sqrt(U_specx**2 + U_specz**2).reshape(xx.shape) - U_PINNt
        
        # True alpha for comparison
        x_grid = xxzzs[:, 0:1]
        z_grid = xxzzs[:, 1:2]
        alpha_true_grid = compute_alpha_true_numpy(x_grid, z_grid).reshape(xx.shape)
        alpha_pred = alphat.numpy().reshape(xx.shape)
        
        # Displacement error: SPECFEM - PINN
        fig = plt.figure(figsize=(10, 8))
        plt.contourf(xx*Lx, zz*Lz, U_diff, 100, cmap='jet')
        plt.xlabel('x (km)')
        plt.ylabel('z (km)')
        plt.title(f'Displacement Error: SPECFEM - PINN (Epoch {epoch})')
        plt.colorbar(label='Error')
        plt.axis('scaled')
        save_figure('pointwise_Error_spec_minus_PINNs.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Alpha misfit: true - inverted
        fig = plt.figure(figsize=(10, 8))
        plt.contourf(xx*Lx, zz*Lz, alpha_true_grid - alpha_pred, 100, cmap='jet')
        plt.xlabel('x (km)')
        plt.ylabel('z (km)')
        plt.title(f'Wave Speed Misfit: True - Inverted (Epoch {epoch})')
        plt.colorbar(label='Misfit (km/s)')
        plt.axis('scaled')
        save_figure('alpha_misfit.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Seismogram comparison at the ACTUAL receiver coordinates X_S.
        # IMPORTANT: station 0 is the reference for relative-phi seismic loss,
        # so its trace can look artificially near-zero. Plot a non-reference station.
        if len(Sz) >= l_sub:
            seis_input = tf.constant(X_S.astype(np.float32))
            with tf.GradientTape() as tape_viz:
                tape_viz.watch(seis_input)
                phi_viz = wave_net(seis_input)
            grad_viz = tape_viz.gradient(phi_viz, seis_input)
            ux_seism_pred = grad_viz[:, 0].numpy()
            uz_seism_pred = grad_viz[:, 1].numpy()

            station_idx = min(max(1, n_seis // 2), n_seis - 1)  # middle non-reference station
            i0 = station_idx * l_sub
            i1 = (station_idx + 1) * l_sub

            fig = plt.figure(figsize=(12, 5))
            plt.plot(t_spec_sub.flatten(), Sz[i0:i1].flatten(), 'ko', mfc='none', markersize=5, label='Input (SPECFEM)')
            plt.plot(t_spec_sub.flatten(), uz_seism_pred[i0:i1], 'r-', linewidth=2, label='PINN Prediction')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Z-Seismogram Comparison - Stn {station_idx+1} at receiver X_S (Epoch {epoch})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            save_figure('ZSeismograms_compare.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(12, 5))
            plt.plot(t_spec_sub.flatten(), Sx[i0:i1].flatten(), 'ko', mfc='none', markersize=5, label='Input (SPECFEM)')
            plt.plot(t_spec_sub.flatten(), ux_seism_pred[i0:i1], 'r-', linewidth=2, label='PINN Prediction')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'X-Seismogram Comparison - Stn {station_idx+1} at receiver X_S (Epoch {epoch})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            save_figure('XSeismograms_compare.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"  [VIZ-DETAIL] Detailed comparisons saved for epoch {epoch}")
    except Exception as e:
        print(f"  [WARNING] Detailed visualization error at epoch {epoch}: {e}")

# Training loop with dynamic weight balancing
if start_epoch >= num_epoch:
    raise ValueError(f"start_epoch={start_epoch} must be smaller than num_epochs={num_epoch}.")
print(f"\n[OK] Starting training from epoch {start_epoch} to {num_epoch-1}...")
print(f"\n[LOSS SETUP] Fixed data-first weights with staged PDE/BC warmup...\n")

# ========== LOSS WEIGHTS FROM ORIGINAL PAPER ==========
# CRITICAL: These weights control the balance between physics and observations
# Actual raw losses observed at epoch 0:
# w_pde = 0.01      - VERY weak physics constraint (raw loss ~129.65 Г— 0.01 = 1.30)
# w_ic  = 0.005     - VERY weak initial condition weight (raw loss ~204.30 Г— 0.005 = 1.02)
# w_seis = 1.5      - Seismic observations constraint (raw loss ~3.43 Г— 1.5 = 5.14)
# w_bc  = 3.0       - Boundary condition weight (raw loss ~2.21 Г— 3.0 = 6.63)
#
# Physics: All loss components are balanced to have similar magnitude (~1-6).
# This helps network learn from ALL constraints simultaneously without one dominating.

# Fixed normalized defaults for this TF2 setup:
# IC and seismograms lead the inversion, while PDE and BC stay soft regularizers.
loss_weights = {
    'w_pde': args.w_pde,
    'w_ic': args.w_ic,
    'w_seis': args.w_seis,
    'w_bc': args.w_bc
}

def get_epoch_loss_weights(epoch):
    """
    3-PHASE CURRICULUM to escape static-blob local minimum.

    Phase 1 (0..800):    IC only
      Forces network to learn NON-TRIVIAL phi from wavefield snapshots.
    Phase 2 (800..2400): IC + Seismograms
      Time-varying seis enforces t-dependence, breaks static-blob trap.
    Phase 3 (2400+):     Full PINN (IC + Seis + PDE + BC)
      PDE ramps slowly to avoid collapsing back to phi=0.
    """
    phase1_end = max(1, int(args.phase1_end))
    phase2_end = max(phase1_end + 1, int(args.phase2_end))
    pde_ramp_epochs = max(1, int(args.pde_ramp_epochs))

    if epoch < phase1_end:
        return {'w_pde': 0.0, 'w_ic': 1.0, 'w_seis': 0.0, 'w_bc': 0.0}

    elif epoch < phase2_end:
        seis_ramp = min((epoch - phase1_end) / float(phase2_end - phase1_end), 1.0)
        return {'w_pde': 0.0, 'w_ic': 1.0, 'w_seis': seis_ramp, 'w_bc': 0.0}

    else:
        pde_ramp = min((epoch - phase2_end) / float(pde_ramp_epochs), 1.0)
        return {
            'w_pde':  loss_weights['w_pde']  * pde_ramp,
            'w_ic':   loss_weights['w_ic'],
            'w_seis': loss_weights['w_seis'],
            'w_bc':   loss_weights['w_bc']   * pde_ramp
        }


save_run_parameters('run_parameters.json', args, loss_weights, data_norm_info)
print('[OK] Run parameters saved to run_parameters.json')

weights_computed = False

start_time = timeit.default_timer()
first_step_debug = True
loss_history_csv = 'loss_history_data.csv'
loss_history = load_loss_history_csv(loss_history_csv)
if len(loss_history) > 0:
    loss_history = [row for row in loss_history if int(row[0]) < start_epoch]
    print(f"[RESUME] Loaded {len(loss_history)} historical loss rows from {loss_history_csv}")

# Compute initial losses to determine scaling reference
print(f"[COMPUTING] Initial loss magnitudes for diagnostic purposes (may take a moment)...")
inputs_sample = XX_unified[:min(2000, len(XX_unified))]
sample_size = len(inputs_sample)
print(f"[COMPUTING] Sample shape: {inputs_sample.shape}, computing derivatives...")

try:
    phi_t, alpha_t, grad_phi_t, ux_t, uz_t, ut_t, uxx_t, uzz_t, utt_t, eq_t, P_t = compute_model_outputs_with_derivatives(tf.cast(inputs_sample, tf.float32))
    print(f"[COMPUTING] Derivatives computed successfully!")
    
    # For diagnostic sample, use safe slicing that respects actual sample boundaries
    # Calculate how many points of each type are in the sample (proportional distribution)
    n_pde_sample = min(int(sample_size * N1 / (N1 + N2 + N3 + N4 + 5000)), sample_size)
    n_ic_sample = min(int(sample_size * (N2 + N3) / (N1 + N2 + N3 + N4 + 5000)), sample_size - n_pde_sample)
    n_seis_sample = min(int(sample_size * N4 / (N1 + N2 + N3 + N4 + 5000)), sample_size - n_pde_sample - n_ic_sample)
    n_bc_sample = sample_size - n_pde_sample - n_ic_sample - n_seis_sample
    
    # Safe indexing within sample bounds
    loss_pde_init = tf.reduce_mean(tf.square(eq_t[0:n_pde_sample])) if n_pde_sample > 0 else tf.constant(0.0)
    
    if n_ic_sample > 0:
        loss_ic_init = (tf.reduce_mean(tf.square(grad_phi_t[n_pde_sample:n_pde_sample+n_ic_sample, 0:1] - 
                                                  U_ic1x_combined[0:min(n_ic_sample, len(U_ic1x_combined))])) +
                        tf.reduce_mean(tf.square(grad_phi_t[n_pde_sample:n_pde_sample+n_ic_sample, 1:2] - 
                                                  U_ic1z_combined[0:min(n_ic_sample, len(U_ic1z_combined))])))
    else:
        loss_ic_init = tf.constant(0.0)
    
    if n_seis_sample > 0:
        loss_seis_init = (tf.reduce_mean(tf.square(grad_phi_t[n_pde_sample+n_ic_sample:n_pde_sample+n_ic_sample+n_seis_sample, 0:1] - 
                                                    U_seisx[0:min(n_seis_sample, len(U_seisx))])) +
                          tf.reduce_mean(tf.square(grad_phi_t[n_pde_sample+n_ic_sample:n_pde_sample+n_ic_sample+n_seis_sample, 1:2] - 
                                                    U_seisz[0:min(n_seis_sample, len(U_seisz))])))
    else:
        loss_seis_init = tf.constant(0.0)
    
    if n_bc_sample > 0:
        loss_bc_init = tf.reduce_mean(tf.square(P_t[n_pde_sample+n_ic_sample+n_seis_sample:]))
    else:
        loss_bc_init = tf.constant(0.0)
    
    print(f"[COMPUTING] Loss components computed successfully!")
except Exception as e:
    print(f"[ERROR] During initial loss computation: {e}")
    import traceback
    traceback.print_exc()
    raise

loss_pde_val_init = loss_pde_init.numpy()
loss_ic_val_init = loss_ic_init.numpy()
loss_seis_val_init = loss_seis_init.numpy()
loss_bc_val_init = loss_bc_init.numpy()

print("\n[LOSS WEIGHTS] Using fixed normalized weights:")
print(f"  PDE  = {loss_weights['w_pde']:.4f}")
print(f"  IC   = {loss_weights['w_ic']:.4f}")
print(f"  Seis = {loss_weights['w_seis']:.4f}")
print(f"  BC   = {loss_weights['w_bc']:.4f}")
print(f"  Warmup epochs = {args.warmup_epochs}")

print(f"\n[LOSS ANALYSIS] Pre-training loss magnitudes:")
print(f"  Raw loss_pde  = {loss_pde_val_init:.6e}  x  w_pde={loss_weights['w_pde']:.4f}  -> Contribution = {loss_weights['w_pde']*loss_pde_val_init:.6e}")
print(f"  Raw loss_ic   = {loss_ic_val_init:.6e}  x  w_ic={loss_weights['w_ic']:.4f}  -> Contribution = {loss_weights['w_ic']*loss_ic_val_init:.6e}")
print(f"  Raw loss_seis = {loss_seis_val_init:.6e}  x  w_seis={loss_weights['w_seis']:.4f}  -> Contribution = {loss_weights['w_seis']*loss_seis_val_init:.6e}")
print(f"  Raw loss_bc   = {loss_bc_val_init:.6e}  x  w_bc={loss_weights['w_bc']:.4f}  -> Contribution = {loss_weights['w_bc']*loss_bc_val_init:.6e}")

total_init = (loss_weights['w_pde']*loss_pde_val_init + 
              loss_weights['w_ic']*loss_ic_val_init + 
              loss_weights['w_seis']*loss_seis_val_init + 
              loss_weights['w_bc']*loss_bc_val_init)
print(f"  в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ")
print(f"  Total weighted loss = {total_init:.6e}")
print(f"\n[EXPECTED BEHAVIOR with your tuned weights]")
print(f"  OK PDE uses w_pde={loss_weights['w_pde']:.4f}")
print(f"  OK IC uses w_ic={loss_weights['w_ic']:.4f}")
print(f"  OK Seismic uses w_seis={loss_weights['w_seis']:.4f}")
print(f"  OK BC uses w_bc={loss_weights['w_bc']:.4f}")
print(f"  OK Data terms dominate, PDE/BC regularize\n")

for epoch in range(start_epoch, num_epoch):
    pde_batch = get_pde_batch(epoch)
    bc_batch  = sample_bc_batch()
    epoch_weights = get_epoch_loss_weights(epoch)

    # Pass each loss group separately so their tapes are NEVER deleted
    inputs_pde_tf  = tf.constant(pde_batch.astype(np.float32))
    inputs_ic1_tf  = tf.constant(XX_ic1_np.astype(np.float32))
    inputs_ic2_tf  = tf.constant(XX_ic2_np.astype(np.float32))
    inputs_seis_tf = tf.constant(XX_seis_np.astype(np.float32))
    inputs_bc_tf   = tf.constant(bc_batch.astype(np.float32))

    loss_val, loss_pde_val, loss_ic_val, loss_seis_val, loss_bc_val = train_step(
        inputs_pde_tf, inputs_ic1_tf, inputs_ic2_tf,
        inputs_seis_tf, inputs_bc_tf, epoch_weights)
    training_epoch_var.assign(epoch + 1)
    
    if first_step_debug and epoch == start_epoch:
        first_step_debug = False
        print(f"\n[DEBUG] POST-EPOCH {epoch} DIAGNOSTICS:")
        print(f"  Loss components with staged fixed weights:")
        print(f"    PDE:      {loss_pde_val.numpy():.6f}  (weight={epoch_weights['w_pde']:.4f}  -> weighted = {epoch_weights['w_pde']*loss_pde_val.numpy():.6f})")
        print(f"    IC:       {loss_ic_val.numpy():.6f}  (weight={epoch_weights['w_ic']:.4f}  -> weighted = {epoch_weights['w_ic']*loss_ic_val.numpy():.6f})")
        print(f"    Seis:     {loss_seis_val.numpy():.6f}  (weight={epoch_weights['w_seis']:.4f}  -> weighted = {epoch_weights['w_seis']*loss_seis_val.numpy():.6f})")
        print(f"    BC:       {loss_bc_val.numpy():.6f}  (weight={epoch_weights['w_bc']:.4f}  -> weighted = {epoch_weights['w_bc']*loss_bc_val.numpy():.6f})")
        print(f"  Total weighted loss: {loss_val.numpy():.6f}")
        print(f"\n  Active weights: PDE={epoch_weights['w_pde']:.4f}, IC={epoch_weights['w_ic']:.4f}, Seis={epoch_weights['w_seis']:.4f}, BC={epoch_weights['w_bc']:.4f}")
        print(f"    Warmup phase keeps PDE/BC soft while data terms shape the field first.")
    
    if epoch == start_epoch:
        print(f"\n[DEBUG] Epoch {epoch} - Weighted loss decomposition:")
        pde_weighted = epoch_weights['w_pde'] * loss_pde_val.numpy()
        ic_weighted = epoch_weights['w_ic'] * loss_ic_val.numpy()
        seis_weighted = epoch_weights['w_seis'] * loss_seis_val.numpy()
        bc_weighted = epoch_weights['w_bc'] * loss_bc_val.numpy()
        total_check = pde_weighted + ic_weighted + seis_weighted + bc_weighted
        
        print(f"  {pde_weighted:12.6f}  <- PDE (raw {loss_pde_val.numpy():.6f} x {epoch_weights['w_pde']:.4f})")
        print(f"  {ic_weighted:12.6f}  <- IC (raw {loss_ic_val.numpy():.6f} x {epoch_weights['w_ic']:.4f})")
        print(f"  {seis_weighted:12.6f}  <- Seismic (raw {loss_seis_val.numpy():.6f} x {epoch_weights['w_seis']:.4f})")
        print(f"  {bc_weighted:12.6f}  <- BC (raw {loss_bc_val.numpy():.6f} x {epoch_weights['w_bc']:.4f})")
        print(f"  в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ")
        print(f"  {total_check:12.6f}  <- Total (reported as {loss_val.numpy():.6f})")
        print(f"\n  Physics-informed behavior:")
        print(f"     * PDE remains a soft regularizer")
        print(f"     * Seismic and IC targets drive the inversion with consistent scaling")
        print(f"     * Result: alpha updates are no longer distorted by mixed target magnitudes\n")
    
    loss_history.append([epoch, loss_val.numpy(), loss_pde_val.numpy(), loss_ic_val.numpy(), loss_seis_val.numpy(), loss_bc_val.numpy()])
    
    # Loss reporting
    if epoch % args.log_every == 0:
        elapsed = timeit.default_timer() - start_time
        done = epoch - start_epoch + 1
        eta = (elapsed / done) * (num_epoch - epoch - 1) if done > 0 else 0
        print(f'Epoch: {epoch}, Total Loss: {loss_val.numpy():.6f}, PDE: {loss_pde_val.numpy():.6f}, IC: {loss_ic_val.numpy():.6f}, Seis: {loss_seis_val.numpy():.6f}, BC: {loss_bc_val.numpy():.6f}')
        print(f'  Time elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s')
        save_loss_history_csv(loss_history_csv, loss_history)

    if epoch % args.detailed_every == 0 or epoch == num_epoch - 1:
        generate_detailed_visualizations(epoch, loss_history)

    # Core visualizations should be less frequent than the optimization steps.
    if epoch % args.plot_every == 0 or epoch == num_epoch - 1:
        generate_visualizations(epoch, loss_history)

    if epoch % args.checkpoint_every == 0 and epoch > 0:
        safe_save_checkpoint(f'epoch {epoch}')
        save_loss_history_csv(loss_history_csv, loss_history)

# Save final checkpoint
safe_save_checkpoint('final')
save_loss_history_csv(loss_history_csv, loss_history)

# Save weights
with open('recorded_weights.pickle', 'wb') as f:
    pickle.dump({
        'wave_weights': [w.numpy() for w in wave_net.W],
        'wave_biases': [b.numpy() for b in wave_net.b],
        'alpha_weights': [w.numpy() for w in alpha_net.W],
        'alpha_biases': [b.numpy() for b in alpha_net.b],
        'loss_weights': loss_weights
    }, f)
mirror_output('recorded_weights.pickle')
print('Weights saved to recorded_weights.pickle')

# Visualization
print('Generating final plots...')
generate_visualizations(num_epoch-1, loss_history)

# Generate comparison plot with true alpha if available
try:
    print('Generating final plots...')
    feed_dict01 = np.concatenate((xxzzs[:, 0:1], xxzzs[:, 1:2], 0.0*np.ones((xxzzs.shape[0], 1), dtype=np.float32)), axis=1)
    feed_tensor = tf.convert_to_tensor(feed_dict01, dtype=tf.float32)
    phi, alpha, grad_phi, ux_pred, uz_pred, ut, uxx, uzz, utt, eq, P = compute_model_outputs_with_derivatives(feed_tensor)
    alpha_pred = alpha.numpy()
    
    # Compute true alpha on the grid
    x_grid = xxzzs[:, 0:1]
    z_grid = xxzzs[:, 1:2]
    alpha_true_grid = compute_alpha_true_numpy(x_grid, z_grid).reshape(xx.shape)
    
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.contourf(xx*Lx, zz*Lz, alpha_true_grid, 100, cmap='jet')
    plt.xlabel('x (km)')
    plt.ylabel('z (km)')
    plt.title('True Wave Speed')
    plt.colorbar(label='Wave Speed (km/s)')
    plt.axis('scaled')
    
    plt.subplot(1, 3, 2)
    plt.contourf(xx*Lx, zz*Lz, alpha_pred.reshape(xx.shape), 100, cmap='jet')
    plt.xlabel('x (km)')
    plt.ylabel('z (km)')
    plt.title('Inverted Wave Speed')
    plt.colorbar(label='Wave Speed (km/s)')
    plt.axis('scaled')
    
    plt.subplot(1, 3, 3)
    plt.contourf(xx*Lx, zz*Lz, np.abs(alpha_true_grid - alpha_pred.reshape(xx.shape)), 100, cmap='jet')
    plt.xlabel('x (km)')
    plt.ylabel('z (km)')
    plt.title('Wave Speed Misfit (|True - Inverted|)')
    plt.colorbar(label='Misfit (km/s)')
    plt.axis('scaled')
    
    plt.tight_layout()
    save_figure('alpha_comparison_final.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[OK] Alpha comparison plot saved!')
except Exception as e:
    print(f'[WARNING] Could not generate alpha comparison: {e}')
