#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include <complex.h>
#include <vector>

class Interpolate {
    public:
        enum InterpolateMethod { Linear, Quadratic };

        Interpolate(int Nx, int Ny, double* x, double* y) {
            this->Nx = Nx;
            this->Ny = Ny;
            this->x = x;
            this->y = y;
            this->dx = x[1]-x[0];
            this->dy = y[1]-y[0];
        }

        virtual ~Interpolate() = default;

        virtual void gatherFields(std::complex<double>** fields,
                                  std::vector<double> px, std::vector<double> py,
                                  int N_fields, int N_particles,
                                  std::vector<std::vector<std::complex<double>>>& vals) = 0;
        
        virtual void scatterParticles(std::complex<double>** fields,
                                      std::vector<double> px, std::vector<double> py,
                                      int N_fields, int N_particles,
                                      std::vector<std::vector<double>> weights) = 0;

        virtual InterpolateMethod getMethod() = 0;

        int computeIndex(int i, int j) {
            return j*Nx + i;
        }

    protected:
        int Nx;
        int Ny;
        double* x;
        double* y;
        double dx;
        double dy;
};

#endif