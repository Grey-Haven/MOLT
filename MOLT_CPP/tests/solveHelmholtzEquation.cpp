#include <complex>
#include <fftw3.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

void solveHelmholtzEquation(std::vector<std::vector<std::complex<double>>>& RHS,
                            std::vector<std::vector<std::complex<double>>>& LHS,
                            int Nx, int Ny, double Lx, double Ly, double alpha);


std::complex<double> to_std_complex(const fftw_complex& fc) {
    return std::complex<double>(fc[0], fc[1]);
}

// Helper function to compute the wave numbers in one dimension
std::vector<double> compute_wave_numbers(int N, double L) {
    std::vector<double> k(N);
    double dk = 2 * M_PI / L;
    for (int i = 0; i <= N / 2; ++i) {
        k[i] = i * dk;
    }
    for (int i = N / 2 + 1; i < N; ++i) {
        k[i] = (i - N) * dk;
    }
    return k;
}

int main(int argc, char *argv[])
{
    int Nx = 16;
    int Ny = 16;
    double Lx = 2 * M_PI; // Length of the domain in the x-direction
    double Ly = 2 * M_PI; // Length of the domain in the y-direction

    double w_x = 1;
    double w_y = 1;

    double alpha = sqrt(2) / (36.7328);

    // Define the input field
    std::vector<std::vector<std::complex<double>>> RHS(Nx, std::vector<std::complex<double>>(Ny));
    std::vector<std::vector<std::complex<double>>> u_solved(Nx, std::vector<std::complex<double>>(Ny));
    std::vector<std::vector<std::complex<double>>> u_analytic(Nx, std::vector<std::complex<double>>(Ny));
    // Define the function to differentiate (e.g., sin(x) + cos(y))
    // Note this is making the domain [0,Lx-dx] X [0,Ly-dy] (which is proper)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            double x = i * Lx / Nx;
            double y = j * Ly / Ny;
            u_analytic[i][j] = std::sin(w_x*x)*std::sin(w_y*y);
            RHS[i][j] = (1 + 1/(alpha*alpha) * (w_x*w_x + w_y*w_y))*u_analytic[i][j];
            std::cout << x << ", " << y << "; ";
        }
        std::cout << std::endl;
    }

    // Compute the derivative in the x-direction
    solveHelmholtzEquation(RHS, u_solved, Nx, Ny, Lx, Ly, alpha);

    double inf_norm = 0;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            inf_norm = std::max(fabs(u_analytic[i][j].real() - u_solved[i][j].real()), inf_norm);
        }
    }
    
    std::cout << "Inf Norm: " << inf_norm << std::endl;

    return 0;
}

void solveHelmholtzEquation(std::vector<std::vector<std::complex<double>>>& RHS,
                            std::vector<std::vector<std::complex<double>>>& LHS, int Nx, int Ny, double Lx, double Ly, double alpha) {
    // Flatten the 2D input field into a 1D array for FFTW
    std::vector<std::complex<double>> in(Nx * Ny);
    std::vector<std::complex<double>> out(Nx * Ny);
    std::vector<std::complex<double>> derivative(Nx * Ny);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            in[i * Ny + j] = RHS[i][j];
        }
    }

    // Create FFTW plans for the forward and inverse FFT
    fftw_plan forward_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(in.data()), 
        reinterpret_cast<fftw_complex*>(out.data()), 
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(out.data()), 
        reinterpret_cast<fftw_complex*>(derivative.data()), 
        FFTW_BACKWARD, FFTW_ESTIMATE);
        
    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the wave numbers in the appropriate direction
    std::vector<double> kx = compute_wave_numbers(Nx, Lx); 
    std::vector<double> ky = compute_wave_numbers(Ny, Ly);

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(out.data())[index]);
            double k_val =  kx[i]*kx[i] + ky[j]*ky[j];
            freq_component /= (1 + 1/(alpha*alpha) * k_val); // Invert the helmholtz operator (I - DELTA) ==Fourier==> (I + (kx^2 + ky^2)))
            reinterpret_cast<fftw_complex*>(out.data())[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(out.data())[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            derivative[i * Ny + j] /= (Nx * Ny);
            LHS[i][j] = derivative[i * Ny + j];
        }
    }

    // Clean up FFTW plans
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
}