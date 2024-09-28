#include <vector>
#include <complex.h>
#include <fftw3.h>

class FFT {
    public:
        FFT(int Nx, int Ny, double Lx, double Ly) {
            this->Nx = Nx;
            this->Ny = Ny;

            this->forwardIn   = new std::complex<double>[Nx * Ny];
            this->forwardOut  = new std::complex<double>[Nx * Ny];
            this->backwardIn  = new std::complex<double>[Nx * Ny];
            this->backwardOut = new std::complex<double>[Nx * Ny];

            // Create FFTW plans for the forward and inverse FFT
            this->forward_plan = fftw_plan_dft_2d(Nx, Ny,
                                                reinterpret_cast<fftw_complex*>(forwardIn), 
                                                reinterpret_cast<fftw_complex*>(forwardOut), 
                                                FFTW_FORWARD, FFTW_ESTIMATE);
            this->inverse_plan = fftw_plan_dft_2d(Nx, Ny,
                                                reinterpret_cast<fftw_complex*>(backwardIn), 
                                                reinterpret_cast<fftw_complex*>(backwardOut), 
                                                FFTW_BACKWARD, FFTW_ESTIMATE);

            this->kx_deriv_1 = compute_wave_numbers(Nx, Lx, true);
            this->ky_deriv_1 = compute_wave_numbers(Ny, Ly, true);

            this->kx_deriv_2 = compute_wave_numbers(Nx, Lx, false);
            this->ky_deriv_2 = compute_wave_numbers(Ny, Ly, false);
        }

        std::complex<double> to_std_complex(const fftw_complex& fc) {
            return std::complex<double>(fc[0], fc[1]);
        }

        int computeIndex(int i, int j) {
            return j*Nx + i;
        }

        void computeFirstDerivative_FFT(std::complex<double>* inputField,
                                        std::complex<double>* derivativeField,
                                        bool isDerivativeInX);
        
        void computeSecondDerivative_FFT(std::complex<double>* inputField, 
                                         std::complex<double>* derivativeField,
                                         bool isDerivativeInX);

        void solveHelmholtzEquation(std::complex<double>* RHS,
                                    std::complex<double>* LHS,
                                    double alpha);

    private:                                     
        int Nx;
        int Ny;
        fftw_plan forward_plan, inverse_plan;
        std::vector<double> kx_deriv_1, ky_deriv_1;
        std::vector<double> kx_deriv_2, ky_deriv_2;
        std::complex<double>* forwardIn;
        std::complex<double>* forwardOut;
        std::complex<double>* backwardIn;
        std::complex<double>* backwardOut;
        

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