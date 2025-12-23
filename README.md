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

After training AND, OR, and NAND gates, XOR is built using the gates' hidden pre-activations and a final neuron trained on those features:

1. Compute intermediate pre-activations:
   p_or = x1 * w_or1 + x2 * w_or2 + b_or
   p_nand = x1 * w_nand1 + x2 * w_nand2 + b_nand

2. Clip pre-activations into [-1, 1]:
   or_clipped = clipped_linear(p_or)
   nand_clipped = clipped_linear(p_nand)

3. Train a final neuron on (or_clipped, nand_clipped) with sigmoid activation so XOR = AND(or_clipped, nand_clipped).

This demonstrates a simple two-layer neural network that uses clipped hidden features for better separability.

---

## Program Output

For each gate, the program prints:

Input: (x1, x2), Predicted: ŷ, Actual: y

For the XOR section, each line prints the input, the *clipped* hidden pre-activations for OR and NAND, and the final predicted output, e.g.:

Input: (0, 1) -> Hidden(OR:1.00, NAND:1.00) -> Predicted: 0.95 (Actual: 1)

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
- Iterations: 3,000 (default in `main.c`; experiments here also used 3,000)
- Random weight initialization (seeded with `time(0)` in `main.c`)

---

## Clipped Linear for Hidden Layers ✅

To improve the hidden-layer signal quality we add a small helper:

```
double clipped_linear(float x)
{
    return MAX(-1.0, MIN(1.0, x));
}
```

This clamps hidden neuron pre-activations to the range [-1, 1] before feeding them into the final sigmoid. Doing so produces stronger, more binary-like hidden features (values near -1 or 1) which the final sigmoid maps to outputs closer to 0 or 1. In practice this yields faster and more stable convergence for the XOR composition (AND(OR, NAND)) because the hidden representation is more separable.

Proof (experimental results, 3000 iterations):

### With clipped linear combined with `sigmoidf()`

```
NAND Gate Parameters:
Input: (0.00, 0.00), Predicted: 0.991256 (~1.0), Actual: 1
Input: (0.00, 1.00), Predicted: 0.834819 (~0.8), Actual: 1
Input: (1.00, 0.00), Predicted: 0.834827 (~0.8), Actual: 1
Input: (1.00, 1.00), Predicted: 0.183899 (~0.2), Actual: 0

AND Gate Parameters:
Input: (0.00, 0.00), Predicted: 0.007574 (~0.0), Actual: 0
Input: (0.00, 1.00), Predicted: 0.149979 (~0.1), Actual: 0
Input: (1.00, 0.00), Predicted: 0.149975 (~0.1), Actual: 0
Input: (1.00, 1.00), Predicted: 0.803105 (~0.8), Actual: 1

OR Gate Parameters:
Input: (0.00, 0.00), Predicted: 0.154763 (~0.2), Actual: 0
Input: (0.00, 1.00), Predicted: 0.897510 (~0.9), Actual: 1
Input: (1.00, 0.00), Predicted: 0.897975 (~0.9), Actual: 1
Input: (1.00, 1.00), Predicted: 0.997630 (~1.0), Actual: 1

XOR Gate Parameters:
Input: (0, 0) -> Hidden(OR:-1.00, NAND:1.00) -> Predicted: 0.064314(~0.1) (Actual: 0)
Input: (0, 1) -> Hidden(OR:1.00, NAND:1.00) -> Predicted: 0.948731(~0.9) (Actual: 1)
Input: (1, 0) -> Hidden(OR:1.00, NAND:1.00) -> Predicted: 0.948731(~0.9) (Actual: 1)
Input: (1, 1) -> Hidden(OR:1.00, NAND:-1.00) -> Predicted: 0.064314(~0.1) (Actual: 0)
```

### With `sigmoidf()` only

```
NAND Gate Parameters:
Input: (0.00, 0.00), Predicted: 0.990794 (~1.0), Actual: 1
Input: (0.00, 1.00), Predicted: 0.832493 (~0.8), Actual: 1
Input: (1.00, 0.00), Predicted: 0.832471 (~0.8), Actual: 1
Input: (1.00, 1.00), Predicted: 0.186640 (~0.2), Actual: 0

AND Gate Parameters:
Input: (0.00, 0.00), Predicted: 0.007660 (~0.0), Actual: 0
Input: (0.00, 1.00), Predicted: 0.150445 (~0.2), Actual: 0
Input: (1.00, 0.00), Predicted: 0.150448 (~0.2), Actual: 0
Input: (1.00, 1.00), Predicted: 0.802480 (~0.8), Actual: 1

OR Gate Parameters:
Input: (0.00, 0.00), Predicted: 0.152884 (~0.2), Actual: 0
Input: (0.00, 1.00), Predicted: 0.899228 (~0.9), Actual: 1
Input: (1.00, 0.00), Predicted: 0.898579 (~0.9), Actual: 1
Input: (1.00, 1.00), Predicted: 0.997722 (~1.0), Actual: 1

XOR Gate Parameters:
Input: (0, 0) -> Hidden(OR:0.15, NAND:0.99) -> Predicted: 0.254623(~0.3) (Actual: 0)
Input: (0, 1) -> Hidden(OR:0.90, NAND:0.83) -> Predicted: 0.744160(~0.7) (Actual: 1)
Input: (1, 0) -> Hidden(OR:0.90, NAND:0.83) -> Predicted: 0.743695(~0.7) (Actual: 1)
Input: (1, 1) -> Hidden(OR:1.00, NAND:0.19) -> Predicted: 0.285562(~0.3) (Actual: 0)
```

These experiments were run at **3000 iterations** to make the convergence speed and hidden-layer separability easier to observe. The clipped-linear hidden representation produces more extreme hidden activations (approx. -1 or 1) which the final sigmoid maps to outputs closer to 0 or 1, improving XOR accuracy and convergence speed.

---

## How to Compile and Run

gcc main.c -lm -o main  
./main

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
