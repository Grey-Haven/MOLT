#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <complex.h>

class Derivative {
    public:
        enum DerivativeMethod { FFT, FD6, MOLT };

        Derivative(int Nx, int Ny, double Lx, double Ly) {
            this->Nx = Nx;
            this->Ny = Ny;
            this->Lx = Lx;
            this->Ly = Ly;
            this->dx = Lx / double(Nx);
            this->dy = Ly / double(Ny);
        }

        virtual ~Derivative() = default;

        virtual void computeFirstDerivative(std::complex<double>* inputField,
                                            std::complex<double>* derivativeField,
                                            bool isDerivativeInX) = 0;
        
        virtual void computeSecondDerivative(std::complex<double>* inputField, 
                                             std::complex<double>* derivativeField,
                                             bool isDerivativeInX) = 0;

        virtual void solveHelmholtzEquation(std::complex<double>* RHS,
                                            std::complex<double>* LHS,
                                            double alpha) = 0;

        virtual DerivativeMethod getMethod() = 0;

        int computeIndex(int i, int j) {
            return j*Nx + i;
        }

    protected:
        int Nx;
        int Ny;
        double Lx;
        double Ly;
        double dx;
        double dy;
};

#endif