#ifndef CUBIC_INTERPOLATE_H
#define CUBIC_INTERPOLATE_H

#include "Interpolate.h"

class CubicInterpolate : public Interpolate {
    public:
        CubicInterpolate(int Nx, int Ny, double* x, double* y);

        ~CubicInterpolate() override = default;

        void gatherFields(std::complex<double>** fields,
                          std::vector<double> px, std::vector<double> py,
                          int N_fields, int N_particles,
                          std::vector<std::vector<std::complex<double>>>& vals) override;
        
        void scatterParticles(std::complex<double>** fields,
                              std::vector<double> px, std::vector<double> py,
                              int N_fields, int N_particles,
                              std::vector<std::vector<double>> weights) override;

        Interpolate::InterpolateMethod getMethod() override;

        double center(double fx) {
            return 2.0/3.0 - fx*fx + std::pow(std::abs(fx), 3) / 2.0;
        }

        double off(double fx) {
            return 1.0/6.0 * std::pow((2 - std::abs(fx)), 3);
        }
};

#endif