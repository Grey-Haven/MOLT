#ifndef FD6_H
#define FD6_H

#include "Derivative.h"

class FD6 : public Derivative {
    public:
        FD6(int Nx, int Ny, double Lx, double Ly);

        ~FD6() override = default;

        void computeFirstDerivative(std::complex<double>* inputField,
                                    std::complex<double>* derivativeField,
                                    bool isDerivativeInX) override;
        
        void computeSecondDerivative(std::complex<double>* inputField, 
                                     std::complex<double>* derivativeField,
                                     bool isDerivativeInX) override;

        void solveHelmholtzEquation(std::complex<double>* RHS,
                                    std::complex<double>* LHS,
                                    double alpha) override;

        Derivative::DerivativeMethod getMethod() override;
};

#endif
