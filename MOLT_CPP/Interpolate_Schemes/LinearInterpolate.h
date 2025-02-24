#ifndef LINEAR_INTERPOLATE_H
#define LINEAR_INTERPOLATE_H

#include "Interpolate.h"

class LinearInterpolate : public Interpolate {
    public:
        LinearInterpolate(int Nx, int Ny, double* x, double* y);

        ~LinearInterpolate() override = default;

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