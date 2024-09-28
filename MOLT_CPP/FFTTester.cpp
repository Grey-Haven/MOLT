#include <gtest/gtest.h>
#include "FFT.h"
#include <complex>

// Test fixture class for FFT
class FFTTest : public ::testing::Test {
protected:
    static const int Nx = 16;
    static const int Ny = 16;
    static constexpr double a_x = -8.0;
    static constexpr double b_x =  8.0;
    static constexpr double a_y = -8.0;
    static constexpr double b_y =  8.0;

    FFTTest() : fft(Nx,Ny,b_x - a_x,b_y - a_y) {} // Constructor initializes FFT object
    FFT fft;
};

// Test for computeFirstDerivative_FFT method
TEST_F(FFTTest, FFT_Index) {
    std::complex<double> inputField[Nx*Ny];

    double Lx = b_x - a_x;
    double Ly = b_y - a_y;

    double dx = Lx / double(Nx);
    double dy = Ly / double(Ny);
    
    // Initialize inputField with some values
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = std::sin(2*M_PI*x/Lx)*std::cos(2*M_PI*y/Ly);
            int idx = fft.computeIndex(i, j);
            inputField[idx] = std::complex<double>(f, 0.0);
        }
    }

    // // Check the results
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = std::sin(2*M_PI*x/Lx)*std::cos(2*M_PI*y/Ly);
            int idx = j*Nx + i;
            EXPECT_EQ(inputField[idx], f);
        }
    }
}

// Test for computeFirstDerivative_FFT method
TEST_F(FFTTest, FFT_FirstDerivative_x) {
    std::complex<double> inputField[Nx*Ny];
    std::complex<double> derivativeField[Nx*Ny];

    double Lx = b_x - a_x;
    double Ly = b_y - a_y;

    double dx = Lx / double(Nx);
    double dy = Ly / double(Ny);

    double w_x = 4*M_PI/Lx;
    double w_y = 2*M_PI/Ly;
    
    // Initialize inputField with some values
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = std::sin(w_x * x)*std::cos(w_y * y);
            int idx = fft.computeIndex(i, j);
            inputField[idx] = std::complex<double>(f, 0.0);
        }
    }

    fft.computeFirstDerivative_FFT(inputField, derivativeField, true);

    // Check the results
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = w_x*std::cos(w_x * x)*std::cos(w_y * y);
            int idx = fft.computeIndex(i, j);
            EXPECT_LT(std::abs(derivativeField[idx].real() - f), 1e-5);
        }
    }
}


// Test for computeFirstDerivative_FFT method
TEST_F(FFTTest, FFT_FirstDerivative_y) {
    std::complex<double> inputField[Nx*Ny];
    std::complex<double> derivativeField[Nx*Ny];

    double Lx = b_x - a_x;
    double Ly = b_y - a_y;

    double dx = Lx / double(Nx);
    double dy = Ly / double(Ny);

    double w_x = 4*M_PI/Lx;
    double w_y = 2*M_PI/Ly;
    
    // Initialize inputField with some values
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = std::sin(w_x * x)*std::cos(w_y * y);
            int idx = fft.computeIndex(i, j);
            inputField[idx] = std::complex<double>(f, 0.0);
        }
    }

    fft.computeFirstDerivative_FFT(inputField, derivativeField, false);

    // Check the results
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = -w_y*std::sin(w_x * x)*std::sin(w_y * y);
            int idx = fft.computeIndex(i, j);
            EXPECT_LT(std::abs(derivativeField[idx].real() - f), 1e-5);
        }
    }
}

// Test for computeFirstDerivative_FFT method
TEST_F(FFTTest, FFT_SecondDerivative_x) {
    std::complex<double> inputField[Nx*Ny];
    std::complex<double> derivativeField[Nx*Ny];

    double Lx = b_x - a_x;
    double Ly = b_y - a_y;

    double dx = Lx / double(Nx);
    double dy = Ly / double(Ny);

    double w_x = 4*M_PI/Lx;
    double w_y = 2*M_PI/Ly;
    
    // Initialize inputField with some values
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = std::sin(w_x * x)*std::cos(w_y * y);
            int idx = fft.computeIndex(i, j);
            inputField[idx] = std::complex<double>(f, 0.0);
        }
    }

    fft.computeSecondDerivative_FFT(inputField, derivativeField, true);

    // Check the results
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = -w_x*w_x*std::sin(w_x * x)*std::cos(w_y * y);
            int idx = fft.computeIndex(i, j);
            EXPECT_LT(std::abs(derivativeField[idx].real() - f), 1e-5);
        }
    }
}

// Test for computeFirstDerivative_FFT method
TEST_F(FFTTest, FFT_SecondDerivative_y) {
    std::complex<double> inputField[Nx*Ny];
    std::complex<double> derivativeField[Nx*Ny];

    double Lx = b_x - a_x;
    double Ly = b_y - a_y;

    double dx = Lx / double(Nx);
    double dy = Ly / double(Ny);

    double w_x = 4*M_PI/Lx;
    double w_y = 2*M_PI/Ly;
    
    // Initialize inputField with some values
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = std::sin(w_x * x)*std::cos(w_y * y);
            int idx = fft.computeIndex(i, j);
            inputField[idx] = std::complex<double>(f, 0.0);
        }
    }

    fft.computeSecondDerivative_FFT(inputField, derivativeField, false);

    // Check the results
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double f = -w_y*w_y*std::sin(w_x * x)*std::cos(w_y * y);
            int idx = fft.computeIndex(i, j);
            EXPECT_LT(std::abs(derivativeField[idx].real() - f), 1e-5);
        }
    }
}

// Test for computeFirstDerivative_FFT method
TEST_F(FFTTest, FFT_Helmoltz) {
    std::complex<double> u_analytic[Nx*Ny];
    std::complex<double> u_approx[Nx*Ny];
    std::complex<double> S[Nx*Ny];

    double Lx = b_x - a_x;
    double Ly = b_y - a_y;

    double dx = Lx / double(Nx);
    double dy = Ly / double(Ny);

    double w_x = 4*M_PI/Lx;
    double w_y = 2*M_PI/Ly;

    double w_x2 = w_x*w_x;
    double w_y2 = w_y*w_y;

    double alpha = std::sqrt(2);
    double alpha2 = alpha*alpha;

    /*
     * MMS:
     *
     * Let u(x,y) = sin(w_x*x)*cos(w_y*y)
     * => (I - Delta/alpha^2)u = S(x,y)
     * => u - Delta(u)/alpha^2 = S
     * => u - (-w_x^2 - w_y^2)/alpha^2u = S
     * => S = (1 + (w_x^2 + w_y^2)/alpha^2)u
     */
    
    // Initialize inputField with some values
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            double u = std::sin(w_x * x)*std::cos(w_y * y);
            double s = (1 + (w_x2 + w_y2)/alpha2)*u;
            int idx = fft.computeIndex(i, j);
            S[idx] = std::complex<double>(s, 0.0);
            u_analytic[idx] = u;
        }
    }

    fft.solveHelmholtzEquation(S, u_approx, alpha);

    // Check the results
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            double x = a_x + i*dx;
            double y = a_y + j*dy;
            int idx = fft.computeIndex(i, j);
            EXPECT_LT(std::abs(u_analytic[idx].real() - u_approx[idx].real()), 1e-5);
        }
    }
}
