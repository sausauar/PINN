"""
Minimal test: can tape_weights compute d(grad_phi/dx)/dW ?
This is the second-order gradient needed for IC/Seis loss.
"""
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

# Simple 2-layer network
class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.W1 = self.add_weight(shape=(3,20), initializer='glorot_uniform', trainable=True, name='W1')
        self.b1 = self.add_weight(shape=(1,20), initializer='zeros', trainable=True, name='b1')
        self.W2 = self.add_weight(shape=(20,1), initializer='glorot_uniform', trainable=True, name='W2')
        self.b2 = self.add_weight(shape=(1,1), initializer='zeros', trainable=True, name='b2')
    def call(self, X):
        H = tf.nn.tanh(tf.matmul(X, self.W1) + self.b1)
        return tf.matmul(H, self.W2) + self.b2

net = tf.keras.optimizers.Adam(1e-3)
model = Net()
target_grad = tf.constant(np.random.randn(50,1).astype(np.float32))  # target for ∂phi/∂x
inputs = tf.constant(np.random.randn(50,3).astype(np.float32))

print("=== TEST: d(MSE(∂phi/∂x - target))/dW ===")

# Method A: tape_ic inside tape_weights (our new approach)
with tf.GradientTape() as tape_w:
    with tf.GradientTape() as tape_ic:
        tape_ic.watch(inputs)
        phi = model(inputs)
    grad_phi = tape_ic.gradient(phi, inputs)
    loss_ic = tf.reduce_mean(tf.square(grad_phi[:, 0:1] - target_grad))

grads_A = tape_w.gradient(loss_ic, model.trainable_variables)
norms_A = [tf.norm(g).numpy() if g is not None else 0.0 for g in grads_A]
print(f"Method A (tape inside tape_weights):")
print(f"  loss = {loss_ic.numpy():.4f}")
print(f"  grad norms W1={norms_A[0]:.4e}  b1={norms_A[1]:.4e}  W2={norms_A[2]:.4e}  b2={norms_A[3]:.4e}")
print(f"  Note: b2 gradient is expected to be ~0 for dphi/dx-based loss")
print()

# Method B: direct MSE(phi - integrated_target) - first-order
target_phi = tf.constant(np.random.randn(50,1).astype(np.float32))
with tf.GradientTape() as tape_w2:
    phi2 = model(inputs)
    loss_phi = tf.reduce_mean(tf.square(phi2 - target_phi))

grads_B = tape_w2.gradient(loss_phi, model.trainable_variables)
norms_B = [tf.norm(g).numpy() if g is not None else 0.0 for g in grads_B]
print(f"Method B (direct phi MSE - first order):")
print(f"  loss = {loss_phi.numpy():.4f}")
print(f"  grad norms W1={norms_B[0]:.4e}  b1={norms_B[1]:.4e}  W2={norms_B[2]:.4e}  b2={norms_B[3]:.4e}")
print()

# Method C: tape EXITING before gradient (potential issue)
with tf.GradientTape() as tape_w3:
    with tf.GradientTape() as tape_ic3:
        tape_ic3.watch(inputs)
        phi3 = model(inputs)
    # tape_ic3 exits here
    grad_phi3 = tape_ic3.gradient(phi3, inputs)  # called INSIDE tape_w3 but AFTER tape_ic3 exits
    loss_ic3 = tf.reduce_mean(tf.square(grad_phi3[:, 0:1] - target_grad))

grads_C = tape_w3.gradient(loss_ic3, model.trainable_variables)
norms_C = [tf.norm(g).numpy() if g is not None else 0.0 for g in grads_C]
print(f"Method C (tape exited before .gradient() call - current code):")
print(f"  loss = {loss_ic3.numpy():.4f}")
print(f"  grad norms W1={norms_C[0]:.4e}  b1={norms_C[1]:.4e}  W2={norms_C[2]:.4e}  b2={norms_C[3]:.4e}")
print(f"  Note: b2 gradient is expected to be ~0 for dphi/dx-based loss")
