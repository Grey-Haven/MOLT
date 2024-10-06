#include "LinearInterpolate.h"
#include "Interpolate.h"
#include <iostream>
#include <iomanip>

LinearInterpolate::LinearInterpolate(int Nx, int Ny, double* x, double* y) : Interpolate(Nx, Ny, x, y) { }

/**
 * Name: gatherFields
 * Author: Stephen White
 * Date Created: 10/5/24
 * Date Last Modified: 10/5/24 (Stephen White)
 * Description: 
 * Inputs: fields, px, py, vals
 * Output: technically none, but vals is where the results are stored
 * Dependencies: none
 */
void LinearInterpolate::gatherFields(std::complex<double>** fields,
                                     std::vector<double> px, std::vector<double> py,
                                     int N_fields, int N_particles,
                                     std::vector<std::vector<std::complex<double>>>& vals) {
    // #pragma omp parallel for
    for (int p = 0; p < N_particles; p++) {

        double x_p = px[p];
        double y_p = py[p];

        // We convert from cartesian to logical space
        int lc_x = floor((x_p - x[0])/dx);
        int lc_y = floor((y_p - y[0])/dy);

        const int lc_x_p1 = (lc_x+1) % Nx;
        const int lc_y_p1 = (lc_y+1) % Ny;

        const int ld = computeIndex(lc_x   , lc_y   ); // (left, down)  lc_x,   lc_y
        const int lu = computeIndex(lc_x   , lc_y_p1); // (left, up)    lc_x,   lc_y+1
        const int rd = computeIndex(lc_x_p1, lc_y   ); // (rite, down)  lc_x+1, lc_y
        const int ru = computeIndex(lc_x_p1, lc_y_p1); // (rite, up)    lc_x+1, lc_y+1

        double xNode = this->x[lc_x];
        double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        double fx = (x_p - xNode)/dx;
        double fy = (y_p - yNode)/dy;

        for (int f = 0; f < N_fields; f++) {
            vals[f][p] += (1-fx)*(1-fy)*fields[f][ld];
            vals[f][p] += (1-fx)*(  fy)*fields[f][lu];
            vals[f][p] += (  fx)*(1-fy)*fields[f][rd];
            vals[f][p] += (  fx)*(  fy)*fields[f][ru];
        }
    }
    
}

/**
 * Name: scatterParticles
 * Author: Stephen White
 * Date Created: 10/5/24
 * Date Last Modified: 10/5/24 (Stephen White)
 * Description: 
 * Inputs: fields, px, py, weights
 * Output: technically none, but fields is where the results are stored
 * Dependencies: none
 */
void LinearInterpolate::scatterParticles(std::complex<double>** fields,
                                         std::vector<double> px, std::vector<double> py,
                                         int N_fields, int N_particles,
                                         std::vector<std::vector<double>> weights) {

    for (int p = 0; p < N_particles; p++) {

        double x_p = px[p];
        double y_p = py[p];

        // We convert from cartesian to logical space
        int lc_x = floor((x_p - x[0])/dx);
        int lc_y = floor((y_p - y[0])/dy);

        const int lc_x_p1 = (lc_x+1) % Nx;
        const int lc_y_p1 = (lc_y+1) % Ny;

        const int ld = computeIndex(lc_x   , lc_y   ); // (left, down)  lc_x,   lc_y
        const int lu = computeIndex(lc_x   , lc_y_p1); // (left, up)    lc_x,   lc_y+1
        const int rd = computeIndex(lc_x_p1, lc_y   ); // (rite, down)  lc_x+1, lc_y
        const int ru = computeIndex(lc_x_p1, lc_y_p1); // (rite, up)    lc_x+1, lc_y+1

        double xNode = this->x[lc_x];
        double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        double fx = (x_p - xNode)/dx;
        double fy = (y_p - yNode)/dy;

        for (int f = 0; f < N_fields; f++) {
            double w = weights[f][p];
            fields[f][ld] += (1-fx)*(1-fy)*w;
            fields[f][lu] += (1-fx)*(  fy)*w;
            fields[f][rd] += (  fx)*(1-fy)*w;
            fields[f][ru] += (  fx)*(  fy)*w;
        }
    }
}

Interpolate::InterpolateMethod LinearInterpolate::getMethod() { return Interpolate::Linear; }
