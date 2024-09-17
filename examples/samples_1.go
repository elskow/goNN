package main

import (
	"fmt"
	"math"

	gonn "github.com/elskow/goNN"
)

func layerSizes(X, Y [][]float64) (int, int) {
	n_x := len(X)
	n_y := len(Y)

	return n_x, n_y
}

func normalizeData(X [][]float64) [][]float64 {
	n := len(X)
	m := len(X[0])
	mean := make([]float64, n)
	std := make([]float64, n)

	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < m; j++ {
			sum += X[i][j]
		}
		mean[i] = sum / float64(m)
	}

	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < m; j++ {
			sum += (X[i][j] - mean[i]) * (X[i][j] - mean[i])
		}
		std[i] = math.Sqrt(sum / float64(m))
	}

	X_norm := make([][]float64, n)
	for i := 0; i < n; i++ {
		X_norm[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			X_norm[i][j] = (X[i][j] - mean[i]) / std[i]
		}
	}

	return X_norm
}

func nnModel(X, Y [][]float64, n_h, numIterations int, learningRate float64, printCost bool, activationFunc, activationPrime gonn.ActivationFunc, lossFunc gonn.LossFunc) *gonn.NeuralNetwork {
	n_x, n_y := layerSizes(X, Y)

	nn := gonn.NewNeuralNetwork(n_x, n_h, n_y, activationFunc, activationPrime, lossFunc)

	for i := 0; i < numIterations; i++ {
		A2, cache := nn.ForwardPropagation(X)

		cost := nn.ComputeCost(A2, Y)

		grads := nn.BackwardPropagation(cache, X, Y)

		nn.UpdateParameters(grads, learningRate)

		if printCost && i%1000 == 0 {
			fmt.Printf("Cost after iteration %d: %f\n", i, cost)
		}
	}

	return nn
}

func predict(nn *gonn.NeuralNetwork, X [][]float64) [][]float64 {
	A2, _ := nn.ForwardPropagation(X)
	predictions := make([][]float64, len(A2))
	for i := range A2 {
		predictions[i] = make([]float64, len(A2[i]))
		for j := range A2[i] {
			if A2[i][j] > 0.5 {
				predictions[i][j] = 1
			} else {
				predictions[i][j] = 0
			}
		}
	}
	return predictions
}

func main() {
	X1 := [][]float64{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 8, 10, 2, 3, 16},
	}
	Y1 := [][]float64{
		{1, 0, 0, 1, 1, 0, 0, 1},
	}

	n_h := 10
	numIterations := 10000
	learningRate := 0.1
	printCost := true

	// Use sigmoid activation and cross-entropy loss for classification
	nn1 := nnModel(X1, Y1, n_h, numIterations, learningRate, printCost, gonn.Sigmoid, gonn.SigmoidPrime, gonn.CrossEntropy)

	testX1 := [][]float64{
		{2, 3, 4, 6, 8, 21},
		{4, 6, 8, 11, 16, 42},
	}
	testY1 := [][]float64{
		{1, 1, 1, 0, 1, 1},
	}

	fmt.Println("Predictions for first neural network:")
	predictions1 := predict(nn1, testX1)
	for i := range predictions1[0] {
		fmt.Printf("Input: [%.0f, %.0f] Output: %.0f Expected: %.0f\n", testX1[0][i], testX1[1][i], predictions1[0][i], testY1[0][i])
	}

	n_h = 10
	numIterations = 10000
	learningRate = 0.1
	printCost = true

	// Example for regression
	X2 := [][]float64{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 8, 10, 2, 3, 16},
	}
	Y2 := [][]float64{
		{2, 6, 12, 32, 50, 12, 21, 128},
	}

	nn2 := nnModel(X2, Y2, n_h, numIterations, learningRate, printCost, gonn.Relu, gonn.ReluPrime, gonn.MSE)

	testX2 := [][]float64{
		{2, 3, 4, 6, 8, 21},
		{4, 6, 8, 11, 16, 42},
	}
	testY2 := [][]float64{
		{8, 18, 32, 66, 128, 882},
	}

	fmt.Println("Predictions for second neural network (regression):")
	predictions2, _ := nn2.ForwardPropagation(testX2)
	for i := range predictions2[0] {
		fmt.Printf("Input: [%.0f, %.0f] Output: %.2f Expected: %.2f\n", testX2[0][i], testX2[1][i], predictions2[0][i], testY2[0][i])
	}
}
