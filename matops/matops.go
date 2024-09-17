package matops

import (
    "math"
    "sync"
)

func Sub(A, B [][]float64) [][]float64 {
    n, m := len(A), len(A[0])
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, m)
    }
    for i := range A {
        for j := range A[i] {
            result[i][j] = A[i][j] - B[i][j]
        }
    }
    return result
}

func Scale(A [][]float64, scalar float64) [][]float64 {
    n, m := len(A), len(A[0])
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, m)
    }
    for i := range A {
        for j := range A[i] {
            result[i][j] = A[i][j] * scalar
        }
    }
    return result
}

func Transpose(A [][]float64) [][]float64 {
    n, m := len(A), len(A[0])
    result := make([][]float64, m)
    for i := range result {
        result[i] = make([]float64, n)
    }
    for i := range A {
        for j := range A[i] {
            result[j][i] = A[i][j]
        }
    }
    return result
}

func MulElem(A, B [][]float64) [][]float64 {
    n, m := len(A), len(A[0])
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, m)
    }
    for i := range A {
        for j := range A[i] {
            result[i][j] = A[i][j] * B[i][j]
        }
    }
    return result
}

func SubScalar(scalar float64, matrix [][]float64) [][]float64 {
    n, m := len(matrix), len(matrix[0])
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, m)
    }
    for i := range matrix {
        for j := range matrix[i] {
            result[i][j] = scalar - matrix[i][j]
        }
    }
    return result
}

func SumCols(A [][]float64) [][]float64 {
    n := len(A)
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, 1)
    }
    for i := range A {
        sum := 0.0
        for j := range A[i] {
            sum += A[i][j]
        }
        result[i][0] = sum
    }
    return result
}

func Mul(W, X [][]float64) [][]float64 {
    n, m := len(W), len(X[0])
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, m)
    }
    var wg sync.WaitGroup
    for i := range W {
        wg.Add(1)
        go func(i int) {
            defer wg.Done()
            for j := range result[i] {
                for k := range X {
                    result[i][j] += W[i][k] * X[k][j]
                }
            }
        }(i)
    }
    wg.Wait()
    return result
}

func Add(W, X [][]float64) [][]float64 {
    n, m := len(W), len(W[0])
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, m)
    }
    for i := range W {
        for j := range W[i] {
            result[i][j] = W[i][j] + X[i][j]
        }
    }
    return result
}

func Sum(W [][]float64) float64 {
    sum := 0.0
    for _, row := range W {
        for _, val := range row {
            sum += val
        }
    }
    return sum
}

func Broadcast(matrix [][]float64, cols int) [][]float64 {
    n := len(matrix)
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, cols)
    }
    for i := range matrix {
        for j := range result[i] {
            result[i][j] = matrix[i][0]
        }
    }
    return result
}

func Log(matrix [][]float64) [][]float64 {
    n, m := len(matrix), len(matrix[0])
    result := make([][]float64, n)
    for i := range result {
        result[i] = make([]float64, m)
    }
    for i := range matrix {
        for j := range matrix[i] {
            result[i][j] = math.Log(matrix[i][j])
        }
    }
    return result
}