# Logic Gates & XOR Neural Network in C

This project implements basic logic gates (AND, OR, NAND) and builds an XOR gate using a small neural network written entirely in C, without any machine-learning libraries.


![img](https://i.imgur.com/Md5gF4F.png)


The project is purely educational and demonstrates how neural networks work internally: cost functions, sigmoid activation, numerical gradients, and gradient descent.

---

## Core Idea

Each logic gate is represented as a single neuron:

$$
y = \sigma(w_1 x_1 + w_2 x_2 + b)
$$

Where:
- w1, w2 are weights
- b is the bias
- sigmoid provides non-linearity

XOR is not linearly separable, so it cannot be learned by a single neuron. Instead, XOR is constructed using multiple learned gates:

$$
XOR(x_1, x_2) = AND(OR(x_1, x_2),\, NAND(x_1, x_2))
$$

---

## Sigmoid Activation Function

The sigmoid function is used for all neurons:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

It maps values into the range (0, 1), making it suitable for binary logic outputs.

---

## Training Data

Each gate is trained on its corresponding truth table.

Each gate is trained on its truth table:

### AND
| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 0 |
| 1  | 0  | 0 |
| 1  | 1  | 1 |

### OR
| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 1 |

### NAND
| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 1 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

---

## Cost Function

Training minimizes Mean Squared Error (MSE):

$$
MSE = \frac{1}{N} \sum (y_{pred} - y_{true})^2
$$

Lower cost means the neuron behaves closer to the correct logic gate.

---

## Gradient Descent (Finite Differences)

Instead of analytical derivatives, gradients are computed using finite differences:

$$
\frac{\partial C}{\partial w} \approx \frac{C(w + \varepsilon) - C(w)}{\varepsilon}
$$

Weights are updated using gradient descent:

$$
w := w - \eta \cdot \nabla C
$$

This approach is slow but very clear and educational.

---

## Gate Structure

Each logic gate is stored as:
```
typedef struct Gate {
    float w1;
    float w2;
    float b;
} Gate;
```
Each gate is trained independently.

---

## XOR Construction

After training AND, OR, and NAND gates, XOR is built in two steps:

1. Compute intermediate outputs:
   OR(x1, x2)
   NAND(x1, x2)

2. Train a final neuron:
   XOR = AND(OR, NAND)

This demonstrates a simple multi-layer neural network.

---

## Program Output

For each gate, the program prints:

Input: (x1, x2), Predicted: ŷ, Actual: y

Predicted values are close to:
- 0.0 for false
- 1.0 for true

---

## Training Parameters

- Activation: Sigmoid
- Cost Function: Mean Squared Error
- Gradient Method: Finite Differences
- Learning Rate: 0.1
- Epsilon: 0.1
- Iterations: 100,000
- Random weight initialization

---

## How to Compile and Run

gcc logic_gates.c -lm -o logic_gates  
./logic_gates

The -lm flag is required for math.h.

---

## Learning Goals

This project helps understand:
- How neurons compute outputs
- Why XOR requires multiple layers
- How gradient descent works internally
- How logic gates map to neural networks
- The importance of non-linearity

No frameworks. No abstractions. Just math and C.

---
