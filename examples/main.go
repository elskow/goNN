package main

import (
	"fmt"

	gonn "github.com/elskow/goNN"
)

func layerSizes(X, Y [][]float64) (int, int) {
	n_x := len(X)
	n_y := len(Y)

	return n_x, n_y
}

func nnModel(X, Y [][]float64, n_h, numIterations int, learningRate float64, printCost bool) *gonn.NeuralNetwork {
	n_x, n_y := layerSizes(X, Y)

	nn := gonn.NewNeuralNetwork(n_x, n_h, n_y)

	for i := 0; i < numIterations; i++ {
		A2, cache := nn.ForwardPropagation(X)

		cost := gonn.ComputeCost(A2, Y)

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
	numIterations := 1000000
	learningRate := 0.1
	printCost := true

	nn1 := nnModel(X1, Y1, n_h, numIterations, learningRate, printCost)

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
}
