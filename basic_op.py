import tensorflow as tf

print("TensorFlow Basic Operations")

# 1. Tensor Creation
a = tf.constant([100, 200, 300])

print("\nTensor:", a.numpy())
print("Shape:", a.shape)
print("Datatype:", a.dtype)

# 2. Addition
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])

print("\nAddition:", tf.add(x, y).numpy())

# 3. Subtraction
print("Subtraction:", tf.subtract(y, x).numpy())

# 4. Multiplication
print("Multiplication:", tf.multiply(x, y).numpy())

# 5. Division
print("Division:", tf.divide(y, x).numpy())

# 6. Reshape
r = tf.constant([1, 2, 3, 4])

reshape = tf.reshape(r, (2,2))

print("\nReshaped Tensor:\n", reshape.numpy())

# 7. Square
s = tf.constant([2, 3, 4])

print("\nSquare:", tf.square(s).numpy())

# 8. Broadcasting
b = tf.constant([[1,2],[3,4]])

print("\nBroadcasting:\n", (b + 5).numpy())

# 9. Concatenation
c1 = tf.constant([[1,2]])
c2 = tf.constant([[3,4]])

print("\nConcatenation:\n", tf.concat([c1, c2], axis=0).numpy())

# 10. Advanced Operations
p = tf.constant([1, 2, 3])
q = tf.constant([3, 2, 1])

print("\nMaximum:", tf.maximum(p,q).numpy())
print("Minimum:", tf.minimum(p,q).numpy())

n = tf.constant([-1, -2, 3])

print("Absolute:", tf.abs(n).numpy())

f = tf.constant([1.0, 2.0])

print("Log:", tf.math.log(f).numpy())
print("Exponential:", tf.exp(f).numpy())

print("\nProgram Completed")
