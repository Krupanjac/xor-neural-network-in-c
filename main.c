#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define rand_float() ((float)rand() / (float)RAND_MAX)


float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Structure to hold gate parameters
typedef struct Gate {
    float w1;
    float w2;
    float b;
} Gate;



float cost(float w1, float w2, float b, float train_data[][3], size_t train_data_size) {
    float result = 0.0f;
    for (size_t i = 0; i < train_data_size; ++i) {
        float x1 = train_data[i][0];
        float x2 = train_data[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + b);
        float d = y - train_data[i][2];
        result += d * d;
    }
    return result / (float)train_data_size; // Mean Squared Error
}

Gate finite_difference(Gate g, float eps, float train_data[][3], size_t train_data_size) {
    Gate grad;
    float c = cost(g.w1, g.w2, g.b, train_data, train_data_size);
    grad.w1 = (cost(g.w1 + eps, g.w2, g.b, train_data, train_data_size) - c) / eps;
    grad.w2 = (cost(g.w1, g.w2 + eps, g.b, train_data, train_data_size) - c) / eps;
    grad.b = (cost(g.w1, g.w2, g.b + eps, train_data, train_data_size) - c) / eps;
    return grad;
}

Gate rate_gradient(Gate g, Gate grad, float rate) {
    Gate result;
    result.w1 = g.w1 - rate * grad.w1;
    result.w2 = g.w2 - rate * grad.w2;
    result.b = g.b - rate * grad.b;
    return result;
}

void test_model(Gate g, float train_data[][3], size_t train_data_size) {
    for (size_t i = 0; i < train_data_size; ++i) {
        float x1 = train_data[i][0];
        float x2 = train_data[i][1];
        float y_pred = sigmoidf(x1 * g.w1 + x2 * g.w2 + g.b);
        printf("Input: (%.2f, %.2f), Predicted: %f (~%.1f), Actual: %.0f\n", x1, x2, y_pred, y_pred, train_data[i][2]);
    }
}

void print_gate(Gate g) {
    printf("w1: %f, w2: %f, b: %f\n", sigmoidf(g.w1), sigmoidf(g.w2), sigmoidf(g.b));
}

void train_model(Gate *g, float eps, float rate, size_t iterations, float train_data[][3], size_t train_data_size) {
    for (size_t i = 0; i < iterations; ++i) {
        Gate grad = finite_difference(*g, eps, train_data, train_data_size);
        *g = rate_gradient(*g, grad, rate);
    }
}


int main () {
    //nand gate
float train_data_nand[][3] = {
    {0.0, 0.0, 1.0},
    {0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    {1.0, 1.0, 0.0},
};

//and gate
float train_data_and[][3] = {
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {1.0, 0.0, 0.0},
    {1.0, 1.0, 1.0},
};

//or gate
float train_data_or[][3] = {
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    {1.0, 1.0, 1.0},
};

//xor gate
float train_data_xor[][3] = {
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    {1.0, 1.0, 0.0},
};

#define train_data_size_nand (sizeof(train_data_nand) / sizeof(train_data_nand[0]))
#define train_data_size_and (sizeof(train_data_and) / sizeof(train_data_and[0]))
#define train_data_size_or (sizeof(train_data_or) / sizeof(train_data_or[0]))
#define train_data_size_xor (sizeof(train_data_xor) / sizeof(train_data_xor[0]))


srand(time(0));
float eps = 0.1f;
float rate = 0.1f;
size_t iterations = 220000;

Gate g_nand = {rand_float(), rand_float(), rand_float()};
Gate g_and = {rand_float(), rand_float(), rand_float()};
Gate g_or = {rand_float(), rand_float(), rand_float()};

train_model(&g_nand, eps, rate, iterations, train_data_nand, train_data_size_nand);
train_model(&g_and, eps, rate, iterations, train_data_and, train_data_size_and);
train_model(&g_or, eps, rate, iterations, train_data_or, train_data_size_or);

printf("NAND Gate Parameters:\n");
//print_gate(g_nand);
test_model(g_nand, train_data_nand, train_data_size_nand);

printf("\nAND Gate Parameters:\n");
//print_gate(g_and);
test_model(g_and, train_data_and, train_data_size_and);

printf("\nOR Gate Parameters:\n");
//print_gate(g_or); 
test_model(g_or, train_data_or, train_data_size_or);


//xor is  (OR AND NOT(AND))
Gate g_xor_final = {rand_float(), rand_float(), rand_float()};

float train_data_xor_combined[4][3];
for (size_t i = 0; i < train_data_size_xor; ++i) {
    float x1 = train_data_xor[i][0];
    float x2 = train_data_xor[i][1];
    // Compute intermediate values
    float nand_out = sigmoidf(x1 * g_nand.w1 + x2 * g_nand.w2 + g_nand.b);
    float and_out = sigmoidf(x1 * g_and.w1 + x2 * g_and.w2 + g_and.b);
    float or_out = sigmoidf(x1 * g_or.w1 + x2 * g_or.w2 + g_or.b);
    // XOR output is AND(OR, NAND)
    train_data_xor_combined[i][0] = or_out;
    train_data_xor_combined[i][1] = nand_out;
    train_data_xor_combined[i][2] = train_data_xor[i][2];
}

train_model(&g_xor_final, eps, rate, iterations, train_data_xor_combined, train_data_size_xor);
printf("\nALO XOR Gate Parameters:\n");
for (size_t i = 0; i < train_data_size_xor; ++i) {
        float x1 = train_data_xor[i][0];
        float x2 = train_data_xor[i][1];

        float nand_out = sigmoidf(x1 * g_nand.w1 + x2 * g_nand.w2 + g_nand.b);
        float or_out   = sigmoidf(x1 * g_or.w1   + x2 * g_or.w2   + g_or.b);
        
        float xor_out = sigmoidf(or_out * g_xor_final.w1 + nand_out * g_xor_final.w2 + g_xor_final.b);
        
        printf("Input: (%.0f, %.0f) -> Hidden(OR:%.2f, NAND:%.2f) -> Predicted: %f(~%.1f) (Actual: %.0f)\n",
               x1, x2, or_out, nand_out, xor_out, xor_out, train_data_xor[i][2]);
    }




    return 0;
}
