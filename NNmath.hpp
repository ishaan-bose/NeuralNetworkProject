#include <random>
#include <chrono>
#include <immintrin.h>
#include "rapidcsv.h"
#include <iomanip>


// Include the BLAS header
#include <cblas.h>

// ---------------------------------------------------------
// 1. CUSTOM AVX-512 SIMD KERNELS (O(N) operations)
// ---------------------------------------------------------

/*
███╗░░██╗░█████╗░████████╗███████╗██╗
████╗░██║██╔══██╗╚══██╔══╝██╔════╝╚═╝
██╔██╗██║██║░░██║░░░██║░░░█████╗░░░░░
██║╚████║██║░░██║░░░██║░░░██╔══╝░░░░░
██║░╚███║╚█████╔╝░░░██║░░░███████╗██╗
╚═╝░░╚══╝░╚════╝░░░░╚═╝░░░╚══════╝╚═╝

the below commented out functions are not going to be used as
the compiler's auto vectorization is infact better than naively using SIMD
because it also unrolls loops


██████╗░██╗░░░██╗████████╗░░░
██╔══██╗██║░░░██║╚══██╔══╝░░░
██████╦╝██║░░░██║░░░██║░░░░░░
██╔══██╗██║░░░██║░░░██║░░░██╗
██████╦╝╚██████╔╝░░░██║░░░╚█║
╚═════╝░░╚═════╝░░░░╚═╝░░░░╚╝

the functions such as tanh, the sech^2 from tanh, outer product (actually im unsure of this one), etc.
IS FASTER WITH EXPLICIT SIMD INSTRUCTIONS because the compiler cant figure out im trynna
do pade approximations, etc.

*/



void avx512_add(const double* a, const double* b, double* res, size_t n) {
    //here auto vectorization is better or else i will have to manually unroll the loops n shi but i think link time optimization is needed
    for (size_t i = 0; i < n; i++)
    {
        res[i] = a[i] + b[i];
    }
}


void avx512_sub(const double* a, const double* b, double* res, size_t n) {
    //here auto vectorization is better or else i will have to manually unroll the loops n shi but i think link time optimization is needed
    for (size_t i = 0; i < n; i++)
    {
        res[i] = a[i] - b[i];
    }
}

void avx512_div(const double* a, const double* b, double* res, size_t n) {
    //here auto vectorization is better or else i will have to manually unroll the loops n shi but i think link time optimization is needed
    for (size_t i = 0; i < n; i++)
    {
        res[i] = a[i] / b[i];
    }
}

void avx512_add_scalar(const double* a, double val, double* res, size_t n) {
    //here auto vectorization is better or else i will have to manually unroll the loops n shi but i think link time optimization is needed
    for (size_t i = 0; i < n; i++)
    {
        res[i] = a[i] + val;
    }
}

void avx512_sub_scalar(const double* a, double val, double* res, size_t n) {
    //here auto vectorization is better or else i will have to manually unroll the loops n shi but i think link time optimization is needed
    for (size_t i = 0; i < n; i++)
    {
        res[i] = a[i] - val;
    }
}


void avx512_mul_scalar(const double* a, double val, double* res, size_t n) {
    //here auto vectorization is better or else i will have to manually unroll the loops n shi but i think link time optimization is needed
    for (size_t i = 0; i < n; i++)
    {
        res[i] = a[i] * val;
    }
}

inline double avx512_dot_product(const double* a, const double* b, size_t len)
{
    // cblas_ddot: Returns the dot product of two double-precision vectors
    // len: Number of elements
    // a, b: Pointers to the vectors
    // 1: The stride (increment) between elements for both vectors
    return cblas_ddot((int)len, a, 1, b, 1);
}

inline void avx512_hadamard_product(const double* a, const double* b, double* res, size_t len) {
    //here auto vectorization is better or else i will have to manually unroll the loops n shi but i think link time optimization is needed
    for (size_t i = 0; i < len; i++)
    {
        res[i] = a[i] * b[i];
    }
}

//there is a very similar thing in BLAS called daxpy or smth like that idk
//but i think this is better, i 
inline void avx512_fma(const double* a, const double* b, double scalar, double* res, size_t len) {
    __m512d v_scalar = _mm512_set1_pd(scalar);
    size_t vec_len = len & ~7ULL; // Process in chunks of 8
    size_t i = 0;

    for (; i < vec_len; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        
        // _mm512_fmadd_pd(a, b, c) performs: (a * b) + c
        // Here: (va * v_scalar) + vb
        __m512d v_res = _mm512_fmadd_pd(va, v_scalar, vb);
        
        _mm512_storeu_pd(res + i, v_res);
    }

    // Scalar cleanup
    for (; i < len; ++i) {
        res[i] = (a[i] * scalar) + b[i];
    }
}

// ---------------------------------------------------------
// 2. MATRIX WRAPPERS
// ---------------------------------------------------------

// General M x N Matrix by N x 1 Vector Multiplication
void matrix_vector_multiply_blas(const double* A, const double* x, double* y, int M, int N) {
    // cblas_dgemv: y = alpha * A * x + beta * y
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                M, N,           // Dimensions of A
                1.0,            // alpha
                A, N,           // A, LDA (N columns)
                x, 1,           // x, incX
                0.0,            // beta (0.0 means overwrite y)
                y, 1);          // y, incY
}

inline void vector_matrix_multiply(const double* v, const double* A, double* res, int N, int M) {
    // We use dgemv: y = alpha * op(A) * x + beta * y
    // To do (1xN) * (NxM), we treat A as Transposed (MxN) and multiply by the vector.
    cblas_dgemv(
        CblasRowMajor, 
        CblasTrans,   // Transpose A to match (1xN) * (NxM) logic
        N,            // Original rows of A
        M,            // Original columns of A
        1.0,          // alpha
        A,            // Matrix A
        M,            // LDA (Number of columns in RowMajor)
        v,            // Vector x
        1,            // incX
        0.0,          // beta
        res,          // Result vector y
        1             // incY
    );
}


/**
 * Modifies an array in-place by applying the tanh approximation.
 * Uses AVX-512 to process 8 elements at a time.
 * Formula: x * (1 + (1/9)x^2 + (1/945)x^4) / (1 + (4/9)x^2 + (1/63)x^4)
 * * param data Pointer to the double array to be modified.
 * param len Number of elements in the array.
 */
inline void avx512_tanh_pade(double* data, size_t len) {
    // Define constants for the Padé approximation coefficients
    const __m512d v_one = _mm512_set1_pd(1.0);
    const __m512d v_c1_num = _mm512_set1_pd(1.0 / 9.0);      // 1/9
    const __m512d v_c2_num = _mm512_set1_pd(1.0 / 945.0);    // 1/945
    const __m512d v_c1_den = _mm512_set1_pd(4.0 / 9.0);      // 4/9
    const __m512d v_c2_den = _mm512_set1_pd(1.0 / 63.0);     // 1/63

    size_t vec_len = len & ~7ULL;
    size_t i = 0;

    for (; i < vec_len; i += 8) {
        // Load 8 doubles
        __m512d v_x = _mm512_loadu_pd(data + i);

        // Calculate x^2 and x^4 once to reuse
        __m512d v_x2 = _mm512_mul_pd(v_x, v_x);
        __m512d v_x4 = _mm512_mul_pd(v_x2, v_x2);

        // Numerator: 1 + (1/9)x^2 + (1/945)x^4
        // Using FMA: (v_c2_num * v_x4) + ((v_c1_num * v_x2) + 1.0)
        __m512d v_num_terms = _mm512_fmadd_pd(v_c1_num, v_x2, v_one);
        __m512d v_num = _mm512_fmadd_pd(v_c2_num, v_x4, v_num_terms);
        
        // Final numerator: x * (numerator_poly)
        v_num = _mm512_mul_pd(v_x, v_num);

        // Denominator: 1 + (4/9)x^2 + (1/63)x^4
        __m512d v_den_terms = _mm512_fmadd_pd(v_c1_den, v_x2, v_one);
        __m512d v_den = _mm512_fmadd_pd(v_c2_den, v_x4, v_den_terms);

        // Result: Numerator / Denominator
        __m512d v_res = _mm512_div_pd(v_num, v_den);

        // Store back in place
        _mm512_storeu_pd(data + i, v_res);
    }

    // Scalar cleanup for remaining elements
    for (; i < len; ++i) {
        double x = data[i];
        double x2 = x * x;
        double x4 = x2 * x2;
        
        double num = x * (1.0 + (1.0/9.0)*x2 + (1.0/945.0)*x4);
        double den = 1.0 + (4.0/9.0)*x2 + (1.0/63.0)*x4;
        
        data[i] = num / den;
    }
}

inline void avx512_outer_product(const double* u, size_t m, const double* v, size_t n, double* res)
{
    size_t vec_len_n = n & ~7ULL; // Process columns in chunks of 8

    for (size_t i = 0; i < m; ++i) {
        // Broadcast the scalar u[i] to all 8 elements of the SIMD register.
        // We do this once per row.
        __m512d v_u = _mm512_set1_pd(u[i]);
        
        // Calculate the starting pointer for the current row in the result matrix
        double* row_start = res + i * n;
        
        size_t j = 0;
        // Inner loop: vectorized over v (columns)
        for (; j < vec_len_n; j += 8) {
            // Load 8 elements from v
            __m512d v_v = _mm512_loadu_pd(v + j);
            
            // Multiply u[i] * v[j...j+7]
            __m512d v_res = _mm512_mul_pd(v_u, v_v);
            
            // Store the result directly into the result matrix
            _mm512_storeu_pd(row_start + j, v_res);
        }

        // Scalar cleanup for remaining elements in the row (if n is not a multiple of 8)
        for (; j < n; ++j) {
            row_start[j] = u[i] * v[j];
        }
    }
}

/**
 * Calculates the derivative of tanh (sech^2) given an array of tanh values.
 * Formula: sech^2(x) = 1 - tanh^2(x)
 * Uses Fused Negative Multiply-Add for 1-cycle execution per vector.
 * * @param tanh_vals Pointer to the input array containing pre-computed tanh values.
 * @param res Pointer to the output array where results will be stored.
 * @param len Number of elements.
 */
inline void avx512_sech2_from_tanh(const double* tanh_vals, double* res, size_t len) {
    const __m512d v_one = _mm512_set1_pd(1.0);
    size_t vec_len = len & ~7ULL; // Process in chunks of 8
    size_t i = 0;

    for (; i < vec_len; i += 8) {
        // Load 8 tanh values
        __m512d v_t = _mm512_loadu_pd(tanh_vals + i);

        // Calculate 1.0 - (tanh * tanh)
        // _mm512_fnmadd_pd(a, b, c) performs: -(a * b) + c
        // This is faster than separate multiply and subtract instructions.
        __m512d v_res = _mm512_fnmadd_pd(v_t, v_t, v_one);

        // Store result
        _mm512_storeu_pd(res + i, v_res);
    }

    // Scalar cleanup
    for (; i < len; ++i) {
        double t = tanh_vals[i];
        res[i] = 1.0 - (t * t);
    }
}


