#ifndef QUARTIC_INTERPOLATE_H
#define QUARTIC_INTERPOLATE_H

#include "Interpolate.h"

class QuarticInterpolate : public Interpolate {
    public:
        QuarticInterpolate(int Nx, int Ny, double* x, double* y);

        ~QuarticInterpolate() override = default;

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
            return 115.0/192.0 - 5.0/8.0*std::pow(fx, 2) + 1.0/4.0*std::pow(fx, 4);
        }

        double m1(double fx) {
            return 1.0/96.0*(55.0 + 20.0*std::abs(fx) - 120.0*std::pow(fx, 2) + 80.0*std::pow(std::abs(fx), 3) - 16.0*std::pow(fx, 4));
        }

        double m2(double fx) {
            return 1.0/384.0*std::pow((5.0 - 2.0*std::abs(fx)), 4);
        }
};

#endif