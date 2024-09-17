package gonn

import (
    "math"
    "math/rand"
    "sync"
    "time"

    "github.com/elskow/goNN/matops"
)

type ActivationFunc func(float64) float64
type LossFunc func([][]float64, [][]float64) float64

type NeuralNetworkInterface interface {
    ForwardPropagation(X [][]float64) ([][]float64, Cache)
    BackwardPropagation(cache Cache, X, Y [][]float64) Parameters
    UpdateParameters(grads Parameters, learningRate float64)
    ComputeCost(A2, Y [][]float64) float64
}

type NeuralNetwork struct {
    Parameters      Parameters
    ActivationFunc  ActivationFunc
    LossFunc        LossFunc
    ActivationPrime ActivationFunc
}

type Parameters struct {
    W1, b1, W2, b2 [][]float64
}

type Cache struct {
    Z1, A1, Z2, A2 [][]float64
}

func NewNeuralNetwork(n_x, n_h, n_y int, activationFunc, activationPrime ActivationFunc, lossFunc LossFunc) *NeuralNetwork {
    rand.Seed(time.Now().UnixNano())
    return &NeuralNetwork{
        Parameters:      initializeParameters(n_x, n_h, n_y),
        ActivationFunc:  activationFunc,
        LossFunc:        lossFunc,
        ActivationPrime: activationPrime,
    }
}

func Sigmoid(z float64) float64 {
    return 1 / (1 + math.Exp(-z))
}

func SigmoidPrime(z float64) float64 {
    s := Sigmoid(z)
    return s * (1 - s)
}

func Relu(z float64) float64 {
    if z > 0 {
        return z
    }
    return 0
}

func ReluPrime(z float64) float64 {
    if z > 0 {
        return 1
    }
    return 0
}

func MSE(A2, Y [][]float64) float64 {
    m := float64(len(Y[0]))
    diff := matops.Sub(A2, Y)
    squaredDiff := matops.MulElem(diff, diff)
    return matops.Sum(squaredDiff) / (2 * m)
}

func CrossEntropy(A2, Y [][]float64) float64 {
    m := float64(len(Y[0]))
    logprobs := matops.Add(
        matops.MulElem(Y, matops.Log(A2)),
        matops.MulElem(matops.SubScalar(1, Y), matops.Log(matops.SubScalar(1, A2))),
    )
    return -matops.Sum(logprobs) / m
}

func initializeMatrix(rows, cols int, scale float64) [][]float64 {
    matrix := make([][]float64, rows)
    for i := range matrix {
        matrix[i] = make([]float64, cols)
        for j := range matrix[i] {
            matrix[i][j] = rand.NormFloat64() * scale
        }
    }
    return matrix
}

func initializeParameters(n_x, n_h, n_y int) Parameters {
    return Parameters{
        W1: initializeMatrix(n_h, n_x, 0.01),
        b1: initializeMatrix(n_h, 1, 0),
        W2: initializeMatrix(n_y, n_h, 0.01),
        b2: initializeMatrix(n_y, 1, 0),
    }
}

func applyActivation(Z [][]float64, activationFunc ActivationFunc) [][]float64 {
    A := make([][]float64, len(Z))
    var wg sync.WaitGroup
    for i := range Z {
        A[i] = make([]float64, len(Z[i]))
        wg.Add(1)
        go func(i int) {
            defer wg.Done()
            for j := range Z[i] {
                A[i][j] = activationFunc(Z[i][j])
            }
        }(i)
    }
    wg.Wait()
    return A
}

func (nn *NeuralNetwork) ForwardPropagation(X [][]float64) ([][]float64, Cache) {
    W1, b1 := nn.Parameters.W1, nn.Parameters.b1
    W2, b2 := nn.Parameters.W2, nn.Parameters.b2

    Z1 := matops.Add(matops.Mul(W1, X), matops.Broadcast(b1, len(X[0])))
    A1 := applyActivation(Z1, nn.ActivationFunc)

    Z2 := matops.Add(matops.Mul(W2, A1), matops.Broadcast(b2, len(A1[0])))
    A2 := applyActivation(Z2, nn.ActivationFunc)

    cache := Cache{
        Z1: Z1,
        A1: A1,
        Z2: Z2,
        A2: A2,
    }

    return A2, cache
}

func (nn *NeuralNetwork) ComputeCost(A2, Y [][]float64) float64 {
    return nn.LossFunc(A2, Y)
}

func (nn *NeuralNetwork) BackwardPropagation(cache Cache, X, Y [][]float64) Parameters {
    m := float64(len(X[0]))

    W2 := nn.Parameters.W2
    A1, A2 := cache.A1, cache.A2

    dZ2 := matops.Sub(A2, Y)
    dW2 := matops.Scale(matops.Mul(dZ2, matops.Transpose(A1)), 1/m)
    db2 := matops.Scale(matops.SumCols(dZ2), 1/m)

    dZ1 := matops.MulElem(
        matops.Mul(matops.Transpose(W2), dZ2),
        matops.MulElem(A1, matops.SubScalar(1, A1)),
    )
    dW1 := matops.Scale(matops.Mul(dZ1, matops.Transpose(X)), 1/m)
    db1 := matops.Scale(matops.SumCols(dZ1), 1/m)

    return Parameters{
        W1: dW1,
        b1: db1,
        W2: dW2,
        b2: db2,
    }
}

func (nn *NeuralNetwork) UpdateParameters(grads Parameters, learningRate float64) {
    nn.Parameters.W1 = matops.Sub(nn.Parameters.W1, matops.Scale(grads.W1, learningRate))
    nn.Parameters.b1 = matops.Sub(nn.Parameters.b1, matops.Scale(grads.b1, learningRate))
    nn.Parameters.W2 = matops.Sub(nn.Parameters.W2, matops.Scale(grads.W2, learningRate))
    nn.Parameters.b2 = matops.Sub(nn.Parameters.b2, matops.Scale(grads.b2, learningRate))
}