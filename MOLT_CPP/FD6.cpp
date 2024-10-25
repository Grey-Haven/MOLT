#include "FD6.h"
#include <iostream>
#include <iomanip>

FD6::FD6(int Nx, int Ny, double Lx, double Ly) : Derivative(Nx, Ny, Lx, Ly) { }

/**
 * Name: computeFirstDerivative_FFT
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Computes the first derivative in either the x or y direction of a 2D mesh of complex numbers using the FFTW.
 *              Assumes a periodic domain.
 * Inputs: inputField, derivativeField, isDerivativeInX (boolean indicating which direction the derivative is in)
 * Output: technically none, but derivativeField is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: fftw, to_std_complex
 */
void FD6::computeFirstDerivative(std::complex<double>* inputField, 
                                 std::complex<double>* derivativeField,
                                 bool isDerivativeInX) {

    double h = isDerivativeInX ? dx : dy;

    const double w1 = 3.0/ 4.0;
    const double w2 = 3.0/20.0;
    const double w3 = 1.0/60.0;

    for (int i = 0; i < Nx; i++) {

        int i_m3 = isDerivativeInX ? (i-3+Nx) % Nx : i;
        int i_m2 = isDerivativeInX ? (i-2+Nx) % Nx : i;
        int i_m1 = isDerivativeInX ? (i-1+Nx) % Nx : i;
        int i_p1 = isDerivativeInX ? (i+1   ) % Nx : i;
        int i_p2 = isDerivativeInX ? (i+2   ) % Nx : i;
        int i_p3 = isDerivativeInX ? (i+3   ) % Nx : i;

        for (int j = 0; j < Ny; j++) {

            int j_m3 = isDerivativeInX ? j : (j-3+Ny) % Ny;
            int j_m2 = isDerivativeInX ? j : (j-2+Ny) % Ny;
            int j_m1 = isDerivativeInX ? j : (j-1+Ny) % Ny;
            int j_p1 = isDerivativeInX ? j : (j+1   ) % Ny;
            int j_p2 = isDerivativeInX ? j : (j+2   ) % Ny;
            int j_p3 = isDerivativeInX ? j : (j+3   ) % Ny;

            int idx_m3 = computeIndex(i_m3, j_m3);
            int idx_m2 = computeIndex(i_m2, j_m2);
            int idx_m1 = computeIndex(i_m1, j_m1);
            int idx    = computeIndex(i   , j   );
            int idx_p1 = computeIndex(i_p1, j_p1);
            int idx_p2 = computeIndex(i_p2, j_p2);
            int idx_p3 = computeIndex(i_p3, j_p3);

            derivativeField[idx] = (-w3*inputField[idx_m3] + w2*inputField[idx_m2] - w1*inputField[idx_m1] + w1*inputField[idx_p1] - w2*inputField[idx_p2] + w3*inputField[idx_p3]) / h;
        }
    }
}

/**
 * Name: computeSecondDerivative_FD6
 * Author: Stephen White
 * Date Created: 9/20/24
 * Date Last Modified: 9/20/24 (Stephen White)
 * Description: Computes the second derivative in either the x or y direction of a 2D mesh of complex numbers using the FFTW.
 *              Assumes a periodic domain.
 * Inputs: inputField, derivativeField, isDerivativeInX  (boolean indicating which direction the derivative is in)
 * Output: technically none, but derivativeField is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: fftw, to_std_complex
 */
void FD6::computeSecondDerivative(std::complex<double>* inputField, 
                                  std::complex<double>* derivativeField,
                                  bool isDerivativeInX) {

    double h = isDerivativeInX ? dx : dy;
    double h2 = h*h;

    const double w0 = 49.0/18.0;
    const double w1 = 3.0/ 2.0;
    const double w2 = 3.0/20.0;
    const double w3 = 1.0/90.0;

    for (int i = 0; i < Nx; i++) {

        int i_m3 = isDerivativeInX ? (i-3+Nx) % Nx : i;
        int i_m2 = isDerivativeInX ? (i-2+Nx) % Nx : i;
        int i_m1 = isDerivativeInX ? (i-1+Nx) % Nx : i;
        int i_p1 = isDerivativeInX ? (i+1   ) % Nx : i;
        int i_p2 = isDerivativeInX ? (i+2   ) % Nx : i;
        int i_p3 = isDerivativeInX ? (i+3   ) % Nx : i;

        for (int j = 0; j < Ny; j++) {

            int j_m3 = isDerivativeInX ? j : (j-3+Ny) % Ny;
            int j_m2 = isDerivativeInX ? j : (j-2+Ny) % Ny;
            int j_m1 = isDerivativeInX ? j : (j-1+Ny) % Ny;
            int j_p1 = isDerivativeInX ? j : (j+1   ) % Ny;
            int j_p2 = isDerivativeInX ? j : (j+2   ) % Ny;
            int j_p3 = isDerivativeInX ? j : (j+3   ) % Ny;

            int idx_m3 = computeIndex(i_m3, j_m3);
            int idx_m2 = computeIndex(i_m2, j_m2);
            int idx_m1 = computeIndex(i_m1, j_m1);
            int idx    = computeIndex(i   , j   );
            int idx_p1 = computeIndex(i_p1, j_p1);
            int idx_p2 = computeIndex(i_p2, j_p2);
            int idx_p3 = computeIndex(i_p3, j_p3);

            derivativeField[idx] = (w3*inputField[idx_m3] - w2*inputField[idx_m2] + w1*inputField[idx_m1] - w0*inputField[idx] + w1*inputField[idx_p1] - w2*inputField[idx_p2] + w3*inputField[idx_p3]) / h2;
        }
    }
}

/**
 * Name: solveHelmholtzEquation
 * Author: Stephen White
 * Date Created: 9/20/24
 * Date Last Modified: 9/20/24 (Stephen White)
 * Description: Solves the modified Helmholtz equation (I - (1/alpha^2)Delta) u = RHS using the FFT.
 * Inputs: RHS, LHS, alpha
 * Output: technically none, but LHS is where the result is stored
 * Dependencies: to_std_complex, fftw
 */
void FD6::solveHelmholtzEquation(std::complex<double>* RHS,
                                 std::complex<double>* LHS, double alpha) {

    // We have the system Au = b
    // Here b is the RHS, A is the FD6 matrix
    // The 1D FD6 Second Derivative matrix has the form 
    // (0,0,...,0,1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90, 0, ..., 0)
    // We are solving the Helmholtz equation, which is (I - 1/alpha^2 FD6)u = RHS
    // So A = I-FD6

    std::complex<double> RHS_prev[Nx*Ny];

    std::complex<double> alpha2 = alpha*alpha;
    std::complex<double> alpha2_inv = 1.0/alpha2;

    std::complex<double> dx2 = dx*dx;
    std::complex<double> dy2 = dy*dy;

    std::complex<double> a_ij      = 1.0 - alpha2_inv*(-(49.0/18.0) / dx2 + -(49.0/18.0) / dy2);

    std::complex<double> a_ij_i_m3 = 0.0 - alpha2_inv*(1.0/90.0) / dx2;
    std::complex<double> a_ij_i_m2 = 0.0 - alpha2_inv*(-3.0/20.0) / dx2;
    std::complex<double> a_ij_i_m1 = 0.0 - alpha2_inv*(3.0/2.0) / dx2;

    std::complex<double> a_ij_i_p1 = 0.0 - alpha2_inv*(3.0/2.0) / dx2;
    std::complex<double> a_ij_i_p2 = 0.0 - alpha2_inv*(-3.0/20.0) / dx2;
    std::complex<double> a_ij_i_p3 = 0.0 - alpha2_inv*(1.0/90.0) / dx2;

    std::complex<double> a_ij_j_m3 = 0.0 - alpha2_inv*(1.0/90.0) / dy2;
    std::complex<double> a_ij_j_m2 = 0.0 - alpha2_inv*(-3.0/20.0) / dy2;
    std::complex<double> a_ij_j_m1 = 0.0 - alpha2_inv*(3.0/2.0) / dy2;

    std::complex<double> a_ij_j_p1 = 0.0 - alpha2_inv*(3.0/2.0) / dy2;
    std::complex<double> a_ij_j_p2 = 0.0 - alpha2_inv*(-3.0/20.0) / dy2;
    std::complex<double> a_ij_j_p3 = 0.0 - alpha2_inv*(1.0/90.0) / dy2;

    double TOL = 1e-15;
    int MAX = 10000;
    bool converged = false;

    for (int k = 0; k < MAX; k++) {

        double max_diff = 0;

        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                int ij_idx = computeIndex(i, j);
                RHS_prev[ij_idx] = LHS[ij_idx];
            }
        }

        for (int i = 0; i < Nx; i++) {

            int i_idx_m3 = (i-3+Nx) % Nx;
            int i_idx_m2 = (i-2+Nx) % Nx;
            int i_idx_m1 = (i-1+Nx) % Nx;
            int i_idx    = (i+0   ) % Nx;
            int i_idx_p1 = (i+1   ) % Nx;
            int i_idx_p2 = (i+2   ) % Nx;
            int i_idx_p3 = (i+3   ) % Nx;

            for (int j = 0; j < Ny; j++) {

                int j_idx_m3 = (j-3+Ny) % Ny;
                int j_idx_m2 = (j-2+Ny) % Ny;
                int j_idx_m1 = (j-1+Ny) % Ny;
                int j_idx    = (j+0   ) % Ny;
                int j_idx_p1 = (j+1   ) % Ny;
                int j_idx_p2 = (j+2   ) % Ny;
                int j_idx_p3 = (j+3   ) % Ny;

                int ij_idx      = computeIndex(i_idx   , j_idx);

                int ij_idx_i_m3 = computeIndex(i_idx_m3, j_idx);
                int ij_idx_i_m2 = computeIndex(i_idx_m2, j_idx);
                int ij_idx_i_m1 = computeIndex(i_idx_m1, j_idx);

                int ij_idx_i_p1 = computeIndex(i_idx_p1, j_idx);
                int ij_idx_i_p2 = computeIndex(i_idx_p2, j_idx);
                int ij_idx_i_p3 = computeIndex(i_idx_p3, j_idx);

                int ij_idx_j_m3 = computeIndex(i_idx, j_idx_m3);
                int ij_idx_j_m2 = computeIndex(i_idx, j_idx_m2);
                int ij_idx_j_m1 = computeIndex(i_idx, j_idx_m1);

                int ij_idx_j_p1 = computeIndex(i_idx, j_idx_p1);
                int ij_idx_j_p2 = computeIndex(i_idx, j_idx_p2);
                int ij_idx_j_p3 = computeIndex(i_idx, j_idx_p3);

                std::complex<double> b_ij = RHS[ij_idx];

                std::complex<double> x_ij_i_m3 = LHS[ij_idx_i_m3];
                std::complex<double> x_ij_i_m2 = LHS[ij_idx_i_m2];
                std::complex<double> x_ij_i_m1 = LHS[ij_idx_i_m1];

                std::complex<double> x_ij_i_p1 = LHS[ij_idx_i_p1];
                std::complex<double> x_ij_i_p2 = LHS[ij_idx_i_p2];
                std::complex<double> x_ij_i_p3 = LHS[ij_idx_i_p3];

                std::complex<double> x_ij_j_m3 = LHS[ij_idx_j_m3];
                std::complex<double> x_ij_j_m2 = LHS[ij_idx_j_m2];
                std::complex<double> x_ij_j_m1 = LHS[ij_idx_j_m1];

                std::complex<double> x_ij_j_p1 = LHS[ij_idx_j_p1];
                std::complex<double> x_ij_j_p2 = LHS[ij_idx_j_p2];
                std::complex<double> x_ij_j_p3 = LHS[ij_idx_j_p3];
        
                LHS[ij_idx] = 1.0/a_ij * (b_ij
                                        - a_ij_i_m3*x_ij_i_m3 - a_ij_i_m2*x_ij_i_m2 - a_ij_i_m1*x_ij_i_m1
                                        - a_ij_j_m3*x_ij_j_m3 - a_ij_j_m2*x_ij_j_m2 - a_ij_j_m1*x_ij_j_m1
                                        - a_ij_i_p1*x_ij_i_p1 - a_ij_i_p2*x_ij_i_p2 - a_ij_i_p3*x_ij_i_p3
                                        - a_ij_j_p1*x_ij_j_p1 - a_ij_j_p2*x_ij_j_p2 - a_ij_j_p3*x_ij_j_p3);

                max_diff = (std::abs(RHS_prev[ij_idx] - LHS[ij_idx]) > max_diff) ? std::abs(RHS_prev[ij_idx] - LHS[ij_idx]) : max_diff;
            }
        }

        if (max_diff < TOL) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        std::cout << "Error: Solving the Helmholtz Equation using Gauss-Seidel did not converge." << std::endl;
        throw -1;
    }
}

Derivative::DerivativeMethod FD6::getMethod() { return Derivative::DerivativeMethod::FD6; }
