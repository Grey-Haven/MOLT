#include "QuadraticInterpolate.h"
#include "Interpolate.h"
#include <iostream>
#include <iomanip>

QuadraticInterpolate::QuadraticInterpolate(int Nx, int Ny, double* x, double* y) : Interpolate(Nx, Ny, x, y) { }

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
void QuadraticInterpolate::gatherFields(std::complex<double>** fields,
                                     std::vector<double> px, std::vector<double> py,
                                     int N_fields, int N_particles,
                                     std::vector<std::vector<std::complex<double>>>& vals) {
    #pragma omp parallel for
    for (int p = 0; p < N_particles; p++) {

        double x_p = px[p];
        double y_p = py[p];

        // We convert from cartesian to logical space
        int lc_x = int(round((x_p - x[0])/dx)) % Nx;
        int lc_y = int(round((y_p - y[0])/dy)) % Ny;

        const int lc_x_m1 = (lc_x-1 + Nx) % Nx;
        const int lc_x_p1 = (lc_x+1 + Nx) % Nx;

        const int lc_y_m1 = (lc_y-1 + Ny) % Ny;
        const int lc_y_p1 = (lc_y+1 + Ny) % Ny;

        /*
         * F7 F8 F9
         * F4 F5 F6
         * F1 F2 F3
        */

        const int idx1 = computeIndex(lc_x_m1, lc_y_m1);
        const int idx2 = computeIndex(lc_x   , lc_y_m1);
        const int idx3 = computeIndex(lc_x_p1, lc_y_m1);
        const int idx4 = computeIndex(lc_x_m1, lc_y   );
        const int idx5 = computeIndex(lc_x   , lc_y   );
        const int idx6 = computeIndex(lc_x_p1, lc_y   );
        const int idx7 = computeIndex(lc_x_m1, lc_y_p1);
        const int idx8 = computeIndex(lc_x   , lc_y_p1);
        const int idx9 = computeIndex(lc_x_p1, lc_y_p1);

        double xNode = this->x[lc_x];
        double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        double fx = (x_p - xNode)/dx;
        double fy = (y_p - yNode)/dy;

        double wxs_m1 = .5*std::pow((.5 - fx), 2);
        double wxs    = .75 - fx*fx;  
        double wxs_p1 = .5*std::pow((.5 + fx), 2);

        double wys_m1 = .5*std::pow((.5 - fy), 2);
        double wys    = .75 - fy*fy;  
        double wys_p1 = .5*std::pow((.5 + fy), 2);

        for (int f = 0; f < N_fields; f++) {
            vals[f][p] += wxs_m1*wys_m1*fields[f][idx1];
            vals[f][p] += wxs   *wys_m1*fields[f][idx2];
            vals[f][p] += wxs_p1*wys_m1*fields[f][idx3];
            vals[f][p] += wxs_m1*wys   *fields[f][idx4];
            vals[f][p] += wxs   *wys   *fields[f][idx5];
            vals[f][p] += wxs_p1*wys   *fields[f][idx6];
            vals[f][p] += wxs_m1*wys_p1*fields[f][idx7];
            vals[f][p] += wxs   *wys_p1*fields[f][idx8];
            vals[f][p] += wxs_p1*wys_p1*fields[f][idx9];
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
void QuadraticInterpolate::scatterParticles(std::complex<double>** fields,
                                            std::vector<double> px, std::vector<double> py,
                                            int N_fields, int N_particles,
                                            std::vector<std::vector<double>> weights) {

    for (int p = 0; p < N_particles; p++) {

        double x_p = px[p];
        double y_p = py[p];

        // We convert from cartesian to logical space
        int lc_x = int(round((x_p - x[0])/dx)) % Nx;
        int lc_y = int(round((y_p - y[0])/dy)) % Ny;

        const int lc_x_m1 = (lc_x-1 + Nx) % Nx;
        const int lc_x_p1 = (lc_x+1 + Nx) % Nx;

        const int lc_y_m1 = (lc_y-1 + Ny) % Ny;
        const int lc_y_p1 = (lc_y+1 + Ny) % Ny;

        /*
         * F7 F8 F9
         * F4 F5 F6
         * F1 F2 F3
        */

        const int idx1 = computeIndex(lc_x_m1, lc_y_m1);
        const int idx2 = computeIndex(lc_x   , lc_y_m1);
        const int idx3 = computeIndex(lc_x_p1, lc_y_m1);
        const int idx4 = computeIndex(lc_x_m1, lc_y   );
        const int idx5 = computeIndex(lc_x   , lc_y   );
        const int idx6 = computeIndex(lc_x_p1, lc_y   );
        const int idx7 = computeIndex(lc_x_m1, lc_y_p1);
        const int idx8 = computeIndex(lc_x   , lc_y_p1);
        const int idx9 = computeIndex(lc_x_p1, lc_y_p1);

        double xNode = this->x[lc_x];
        double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        double fx = (x_p - xNode)/dx;
        double fy = (y_p - yNode)/dy;

        double wxs_m1 = .5*std::pow((.5 - fx), 2);
        double wxs    = .75 - fx*fx;  
        double wxs_p1 = .5*std::pow((.5 + fx), 2);

        double wys_m1 = .5*std::pow((.5 - fy), 2);
        double wys    = .75 - fy*fy;  
        double wys_p1 = .5*std::pow((.5 + fy), 2);

        for (int f = 0; f < N_fields; f++) {
            double w = weights[f][p];
            fields[f][idx1] +=  wxs_m1*wys_m1*w;
            fields[f][idx2] +=  wxs   *wys_m1*w;
            fields[f][idx3] +=  wxs_p1*wys_m1*w;
            fields[f][idx4] +=  wxs_m1*wys   *w;
            fields[f][idx5] +=  wxs   *wys   *w;
            fields[f][idx6] +=  wxs_p1*wys   *w;
            fields[f][idx7] +=  wxs_m1*wys_p1*w;
            fields[f][idx8] +=  wxs   *wys_p1*w;
            fields[f][idx9] +=  wxs_p1*wys_p1*w;
        }
    }
}

Interpolate::InterpolateMethod QuadraticInterpolate::getMethod() { return Interpolate::Quadratic; }