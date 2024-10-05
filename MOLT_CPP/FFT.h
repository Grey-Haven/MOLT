#ifndef FFT_H
#define FFT_H

#include <vector>
#include <fftw3.h>
#include "Derivative.h"

class FFT : public Derivative {
    public:
        FFT(int Nx, int Ny, double Lx, double Ly);

        ~FFT() override = default;

        void computeFirstDerivative(std::complex<double>* inputField,
                                        std::complex<double>* derivativeField,
                                        bool isDerivativeInX) override;
        
        void computeSecondDerivative(std::complex<double>* inputField, 
                                         std::complex<double>* derivativeField,
                                         bool isDerivativeInX) override;

        void solveHelmholtzEquation(std::complex<double>* RHS,
                                    std::complex<double>* LHS,
                                    double alpha) override;

        Derivative::DerivativeMethod getMethod() override; // { return DerivativeMethod::FFT; };

    private:
        fftw_plan forward_plan, inverse_plan;
        std::vector<double> kx_deriv_1, ky_deriv_1;
        std::vector<double> kx_deriv_2, ky_deriv_2;
        std::complex<double>* forwardIn;
        std::complex<double>* forwardOut;
        std::complex<double>* backwardIn;
        std::complex<double>* backwardOut;

        std::complex<double> to_std_complex(const fftw_complex& fc) {
            return std::complex<double>(fc[0], fc[1]);
        }

        // Helper function to compute the wave numbers in one dimension
        std::vector<double> compute_wave_numbers(int N, double L, bool first_derivative = true) {
            std::vector<double> k(N);
            double dk = 2 * M_PI / L;
            for (int i = 0; i < N / 2; ++i) {
                k[i] = i * dk;
            }
            k[N / 2] = first_derivative ? 0 : (N/2)*dk;
            for (int i = N / 2 + 1; i < N; ++i) {
                k[i] = (i - N) * dk;
            }
            return k;
        }
};

#endif