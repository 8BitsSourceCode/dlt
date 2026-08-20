import tensorflow as tf
import numpy as np

# 1. Creating tensors of different ranks

scalar = tf.constant(7)

vector = tf.constant([10, 20, 30])

matrix = tf.constant([[1, 2], [3, 4]])

tensor3d = tf.constant([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("Scalar:", scalar, "| Rank:", scalar.ndim)

print("Vector:", vector, "| Shape:", vector.shape)

print("Matrix:\n", matrix, "| Shape:", matrix.shape)

print("3D Tensor Shape:", tensor3d.shape)


# 2. Basic arithmetic operations

a = tf.constant([[1, 2], [3, 4]])

b = tf.constant([[5, 6], [7, 8]])

print("\nAddition:")
print(tf.add(a, b).numpy())

print("Subtraction:")
print(tf.subtract(a, b).numpy())

print("Element-wise Multiplication:")
print(tf.multiply(a, b).numpy())

print("Matrix Multiplication:")
print(tf.matmul(a, b).numpy())


# 3. Reshaping

c = tf.constant([1, 2, 3, 4, 5, 6])

reshaped = tf.reshape(c, [2, 3])

print("\nReshaped Tensor:")
print(reshaped.numpy())


# 4. Indexing and Slicing

d = tf.constant([10, 20, 30, 40, 50])

print("\nFirst Element:", d[0].numpy())

print("Slice [1:4]:", d[1:4].numpy())


# 5. Broadcasting

e = tf.constant([[1], [2], [3]])

f = tf.constant([10, 20, 30])

print("\nBroadcasted Addition:")
print((e + f).numpy())


# 6. Tensor <-> NumPy Conversion

np_array = np.array([1, 2, 3])

tensor_from_np = tf.convert_to_tensor(np_array)

back_to_np = tensor_from_np.numpy()

print("\nTensor from NumPy:")
print(tensor_from_np)

print("Back to NumPy:")
print(back_to_np)