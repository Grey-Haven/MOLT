#ifndef QUADRATIC_INTERPOLATE_H
#define QUADRATIC_INTERPOLATE_H

#include "Interpolate.h"

class QuadraticInterpolate : public Interpolate {
    public:
        QuadraticInterpolate(int Nx, int Ny, double* x, double* y);

        ~QuadraticInterpolate() override = default;

        void gatherFields(std::complex<double>** fields,
                          std::vector<double> px, std::vector<double> py,
                          int N_fields, int N_particles,
                          std::vector<std::vector<std::complex<double>>>& vals) override;
        
        void scatterParticles(std::complex<double>** fields,
                              std::vector<double> px, std::vector<double> py,
                              int N_fields, int N_particles,
                              std::vector<std::vector<double>> weights) override;

        Interpolate::InterpolateMethod getMethod() override;
};

#endif