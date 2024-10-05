#include "FFT.h"
#include <iostream>
#include <iomanip>

FFT::FFT(int Nx, int Ny, double Lx, double Ly) : Derivative(Nx, Ny, Lx, Ly) {

    std::cout << "Entering FFT Constructor" << std::endl;

    this->forwardIn   = new std::complex<double>[Nx * Ny];
    this->forwardOut  = new std::complex<double>[Nx * Ny];
    this->backwardIn  = new std::complex<double>[Nx * Ny];
    this->backwardOut = new std::complex<double>[Nx * Ny];

    std::cout << "Created Arrays" << std::endl;

    // Create FFTW plans for the forward and inverse FFT
    this->forward_plan = fftw_plan_dft_2d(Nx, Ny,
                                          reinterpret_cast<fftw_complex*>(forwardIn), 
                                          reinterpret_cast<fftw_complex*>(forwardOut), 
                                          FFTW_FORWARD, FFTW_ESTIMATE);
    std::cout << "Created Forward Plans" << std::endl;
    this->inverse_plan = fftw_plan_dft_2d(Nx, Ny,
                                          reinterpret_cast<fftw_complex*>(backwardIn), 
                                          reinterpret_cast<fftw_complex*>(backwardOut), 
                                          FFTW_BACKWARD, FFTW_ESTIMATE);
    std::cout << "Created Plans" << std::endl;

    this->kx_deriv_1 = compute_wave_numbers(Nx, Lx, true);
    this->ky_deriv_1 = compute_wave_numbers(Ny, Ly, true);

    this->kx_deriv_2 = compute_wave_numbers(Nx, Lx, false);
    this->ky_deriv_2 = compute_wave_numbers(Ny, Ly, false);

    std::cout << "Exiting FFT Constructor" << std::endl;
}

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
void FFT::computeFirstDerivative(std::complex<double>* inputField, 
                                     std::complex<double>* derivativeField,
                                     bool isDerivativeInX) {

    // Execute the forward FFT
    // fft_execute_dft(plan, in, out)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            forwardIn[j * Nx + i] = inputField[j * Nx + i];
        }
    }
    fftw_execute(forward_plan);

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = isDerivativeInX ? kx_deriv_1 : ky_deriv_1;

    // Apply the derivative operator in the frequency domain
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int idx = computeIndex(i, j); // j * Nx + i;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(forwardOut)[idx]);
            if (isDerivativeInX) {
                freq_component *= std::complex<double>(0, k[i]); // Multiply by sqrt(-1) * kx
            } else {
                freq_component *= std::complex<double>(0, k[j]); // Multiply by sqrt(-1) * ky
            }
            reinterpret_cast<fftw_complex*>(backwardIn)[idx][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(backwardIn)[idx][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    // fftw_execute_dft(inverse_plan, reinterpret_cast<fftw_complex*>(backwardIn), reinterpret_cast<fftw_complex*>(backwardOut));
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            derivativeField[j * Nx + i] = backwardOut[j * Nx + i] / double(Nx * Ny);
        }
    }
}

/**
 * Name: computeSecondDerivative_FFT
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Computes the second derivative in either the x or y direction of a 2D mesh of complex numbers using the FFTW.
 *              Assumes a periodic domain.
 * Inputs: inputField, derivativeField, isDerivativeInX  (boolean indicating which direction the derivative is in)
 * Output: technically none, but derivativeField is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: fftw, to_std_complex
 */
void FFT::computeSecondDerivative(std::complex<double>* inputField, 
                                  std::complex<double>* derivativeField,
                                  bool isDerivativeInX) {

    // Execute the forward FFT
    fftw_execute_dft(forward_plan, reinterpret_cast<fftw_complex*>(inputField), reinterpret_cast<fftw_complex*>(forwardOut));

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = isDerivativeInX ? kx_deriv_2 : ky_deriv_2;

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int idx = computeIndex(i, j); // i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(forwardOut)[idx]);
            double k_val = isDerivativeInX ? k[i] : k[j];
            freq_component *= -k_val * k_val; // Multiply by -k^2
            reinterpret_cast<fftw_complex*>(backwardIn)[idx][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(backwardIn)[idx][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx*Ny; i++) {
        derivativeField[i] = backwardOut[i] / double(Nx * Ny);
    }
}


/**
 * Name: solveHelmholtzEquation
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Solves the modified Helmholtz equation (I - (1/alpha^2)Delta) u = RHS using the FFT.
 * Inputs: RHS, LHS, alpha
 * Output: technically none, but LHS is where the result is stored
 * Dependencies: to_std_complex, fftw
 */
void FFT::solveHelmholtzEquation(std::complex<double>* RHS,
                                 std::complex<double>* LHS, double alpha) {

    // Execute the forward FFT
    fftw_execute_dft(forward_plan, reinterpret_cast<fftw_complex*>(RHS), reinterpret_cast<fftw_complex*>(forwardOut));

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(forwardOut)[index]);
            double k_val =  kx_deriv_2[i]*kx_deriv_2[i] + ky_deriv_2[j]*ky_deriv_2[j];
            freq_component /= (1 + 1/(alpha*alpha) * k_val); // Invert the helmholtz operator (I - (d^2/dx^2 + d^2/dy^2)) ==Fourier==> (I + (kx^2 + ky^2)))
            reinterpret_cast<fftw_complex*>(backwardIn)[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(backwardIn)[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx*Ny; i++) {
        LHS[i] = backwardOut[i] / double(Nx * Ny);
    }
}

Derivative::DerivativeMethod FFT::getMethod() { return Derivative::DerivativeMethod::FFT; }
