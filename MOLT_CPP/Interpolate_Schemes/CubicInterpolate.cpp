#include "CubicInterpolate.h"
#include "Interpolate.h"
#include <iostream>
#include <iomanip>

CubicInterpolate::CubicInterpolate(int Nx, int Ny, double* x, double* y) : Interpolate(Nx, Ny, x, y) { }

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
void CubicInterpolate::gatherFields(std::complex<double>** fields,
                                    std::vector<double> px, std::vector<double> py,
                                    int N_fields, int N_particles,
                                    std::vector<std::vector<std::complex<double>>>& vals) {

    #pragma omp parallel for
    for (int p = 0; p < N_particles; p++) {

        double x_p = px[p];
        double y_p = py[p];

        // We convert from cartesian to logical space
        int lc_x_left = int(floor((x_p - x[0])/dx));
        int lc_y_left = int(floor((y_p - y[0])/dy));

        double xNode_left = this->x[lc_x_left];
        double yNode_left = this->y[lc_y_left];

        double xNode_left_m1 = xNode_left - dx;
        double xNode_rite    = xNode_left + dx;
        double xNode_rite_p1 = xNode_left + 2.0*dx;

        double yNode_left_m1 = yNode_left - dy;
        double yNode_rite    = yNode_left + dy;
        double yNode_rite_p1 = yNode_left + 2.0*dy;

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        double fx_left_m1 = (x_p - xNode_left_m1)/dx;
        double fx_left    = (x_p - xNode_left   )/dx;
        double fx_rite    = (x_p - xNode_rite   )/dx;
        double fx_rite_p1 = (x_p - xNode_rite_p1)/dx;

        double fy_left_m1 = (y_p - yNode_left_m1)/dy;
        double fy_left    = (y_p - yNode_left   )/dy;
        double fy_rite    = (y_p - yNode_rite   )/dy;
        double fy_rite_p1 = (y_p - yNode_rite_p1)/dy;

        double wx_left_m1 = off(fx_left_m1);
        double wx_left    = center(fx_left);
        double wx_rite    = center(fx_rite);
        double wx_rite_p1 = off(fx_rite_p1);

        double wy_left_m1 = off(fy_left_m1);
        double wy_left    = center(fy_left);
        double wy_rite    = center(fy_rite);
        double wy_rite_p1 = off(fy_rite_p1);

        const int lc_x_left_m1 = (lc_x_left - 1 + Nx) % Nx;
                  lc_x_left    = (lc_x_left     + Nx) % Nx;
        const int lc_x_rite    = (lc_x_left + 1 + Nx) % Nx;
        const int lc_x_rite_p1 = (lc_x_left + 2 + Nx) % Nx;

        const int lc_y_left_m1 = (lc_y_left - 1 + Ny) % Ny;
                  lc_y_left    = (lc_y_left     + Ny) % Ny;
        const int lc_y_rite    = (lc_y_left + 1 + Ny) % Ny;
        const int lc_y_rite_p1 = (lc_y_left + 2 + Ny) % Ny;

        /*
         * F13  F14  F15  F16
         * F09  F10  F11  F12
         * F05  F06  F07  F08
         * F01  F02  F03  F04
         */

        const int idx01 = computeIndex(lc_x_left_m1, lc_y_left_m1);
        const int idx02 = computeIndex(lc_x_left   , lc_y_left_m1);
        const int idx03 = computeIndex(lc_x_rite   , lc_y_left_m1);
        const int idx04 = computeIndex(lc_x_rite_p1, lc_y_left_m1);
        
        const int idx05 = computeIndex(lc_x_left_m1, lc_y_left);
        const int idx06 = computeIndex(lc_x_left   , lc_y_left);
        const int idx07 = computeIndex(lc_x_rite   , lc_y_left);
        const int idx08 = computeIndex(lc_x_rite_p1, lc_y_left);
        
        const int idx09 = computeIndex(lc_x_left_m1, lc_y_rite);
        const int idx10 = computeIndex(lc_x_left   , lc_y_rite);
        const int idx11 = computeIndex(lc_x_rite   , lc_y_rite);
        const int idx12 = computeIndex(lc_x_rite_p1, lc_y_rite);

        const int idx13 = computeIndex(lc_x_left_m1, lc_y_rite_p1);
        const int idx14 = computeIndex(lc_x_left   , lc_y_rite_p1);
        const int idx15 = computeIndex(lc_x_rite   , lc_y_rite_p1);
        const int idx16 = computeIndex(lc_x_rite_p1, lc_y_rite_p1);

        for (int f = 0; f < N_fields; f++) {
            vals[f][p] += wx_left_m1 * wy_left_m1*fields[f][idx01];
            vals[f][p] += wx_left    * wy_left_m1*fields[f][idx02];
            vals[f][p] += wx_rite    * wy_left_m1*fields[f][idx03];
            vals[f][p] += wx_rite_p1 * wy_left_m1*fields[f][idx04];

            vals[f][p] += wx_left_m1 * wy_left   *fields[f][idx05];
            vals[f][p] += wx_left    * wy_left   *fields[f][idx06];
            vals[f][p] += wx_rite    * wy_left   *fields[f][idx07];
            vals[f][p] += wx_rite_p1 * wy_left   *fields[f][idx08];

            vals[f][p] += wx_left_m1 * wy_rite   *fields[f][idx09];
            vals[f][p] += wx_left    * wy_rite   *fields[f][idx10];
            vals[f][p] += wx_rite    * wy_rite   *fields[f][idx11];
            vals[f][p] += wx_rite_p1 * wy_rite   *fields[f][idx12];

            vals[f][p] += wx_left_m1 * wy_rite_p1*fields[f][idx13];
            vals[f][p] += wx_left    * wy_rite_p1*fields[f][idx14];
            vals[f][p] += wx_rite    * wy_rite_p1*fields[f][idx15];
            vals[f][p] += wx_rite_p1 * wy_rite_p1*fields[f][idx16];
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
void CubicInterpolate::scatterParticles(std::complex<double>** fields,
                                        std::vector<double> px, std::vector<double> py,
                                        int N_fields, int N_particles,
                                        std::vector<std::vector<double>> weights) {

    double S [Nx*Ny] = {0};
    for (int f = 0; f < N_fields; f++) {
        // Presumably there are fewer fields than there are particles.
        // This ordering allows us to parallelize at the relatively low
        // cost of extra computation of indices etc.
        // std::cout << "N_particles: " << N_particles << std::endl;
        #pragma omp parallel for reduction(+:S[:Nx*Ny])
        for (int p = 0; p < N_particles; p++) {

            double x_p = px[p];
            double y_p = py[p];

            // We convert from cartesian to logical space
            int lc_x_left = int(floor((x_p - x[0])/dx));
            int lc_y_left = int(floor((y_p - y[0])/dy));

            double xNode_left = this->x[lc_x_left];
            double yNode_left = this->y[lc_y_left];

            double xNode_left_m1 = xNode_left - dx;
            double xNode_rite    = xNode_left + dx;
            double xNode_rite_p1 = xNode_left + 2.0*dx;

            double yNode_left_m1 = yNode_left - dy;
            double yNode_rite    = yNode_left + dy;
            double yNode_rite_p1 = yNode_left + 2.0*dy;

            // We compute the fractional distance of a particle from
            // the nearest node.
            // eg x=[0,.1,.2,.3], particleX = [.225]
            // The particle's fractional is 1/4
            double fx_left_m1 = (x_p - xNode_left_m1)/dx;
            double fx_left    = (x_p - xNode_left   )/dx;
            double fx_rite    = (x_p - xNode_rite   )/dx;
            double fx_rite_p1 = (x_p - xNode_rite_p1)/dx;

            double fy_left_m1 = (y_p - yNode_left_m1)/dy;
            double fy_left    = (y_p - yNode_left   )/dy;
            double fy_rite    = (y_p - yNode_rite   )/dy;
            double fy_rite_p1 = (y_p - yNode_rite_p1)/dy;

            double wx_left_m1 = off(fx_left_m1);
            double wx_left    = center(fx_left);
            double wx_rite    = center(fx_rite);
            double wx_rite_p1 = off(fx_rite_p1);

            double wy_left_m1 = off(fy_left_m1);
            double wy_left    = center(fy_left);
            double wy_rite    = center(fy_rite);
            double wy_rite_p1 = off(fy_rite_p1);

            const int lc_x_left_m1 = (lc_x_left - 1 + Nx) % Nx;
                      lc_x_left    = (lc_x_left     + Nx) % Nx;
            const int lc_x_rite    = (lc_x_left + 1 + Nx) % Nx;
            const int lc_x_rite_p1 = (lc_x_left + 2 + Nx) % Nx;

            const int lc_y_left_m1 = (lc_y_left - 1 + Ny) % Ny;
                      lc_y_left    = (lc_y_left     + Ny) % Ny;
            const int lc_y_rite    = (lc_y_left + 1 + Ny) % Ny;
            const int lc_y_rite_p1 = (lc_y_left + 2 + Ny) % Ny;

            /*
            * F13  F14  F15  F16
            * F09  F10  F11  F12
            * F05  F06  F07  F08
            * F01  F02  F03  F04
            */

            const int idx01 = computeIndex(lc_x_left_m1, lc_y_left_m1);
            const int idx02 = computeIndex(lc_x_left   , lc_y_left_m1);
            const int idx03 = computeIndex(lc_x_rite   , lc_y_left_m1);
            const int idx04 = computeIndex(lc_x_rite_p1, lc_y_left_m1);
            
            const int idx05 = computeIndex(lc_x_left_m1, lc_y_left);
            const int idx06 = computeIndex(lc_x_left   , lc_y_left);
            const int idx07 = computeIndex(lc_x_rite   , lc_y_left);
            const int idx08 = computeIndex(lc_x_rite_p1, lc_y_left);
            
            const int idx09 = computeIndex(lc_x_left_m1, lc_y_rite);
            const int idx10 = computeIndex(lc_x_left   , lc_y_rite);
            const int idx11 = computeIndex(lc_x_rite   , lc_y_rite);
            const int idx12 = computeIndex(lc_x_rite_p1, lc_y_rite);

            const int idx13 = computeIndex(lc_x_left_m1, lc_y_rite_p1);
            const int idx14 = computeIndex(lc_x_left   , lc_y_rite_p1);
            const int idx15 = computeIndex(lc_x_rite   , lc_y_rite_p1);
            const int idx16 = computeIndex(lc_x_rite_p1, lc_y_rite_p1);

            double w = weights[f][p];

            S[idx01] +=  wx_left_m1 * wy_left_m1 * w;
            S[idx02] +=  wx_left    * wy_left_m1 * w;
            S[idx03] +=  wx_rite    * wy_left_m1 * w;
            S[idx04] +=  wx_rite_p1 * wy_left_m1 * w;
			
			S[idx05] +=  wx_left_m1 * wy_left    * w;
            S[idx06] +=  wx_left    * wy_left    * w;
            S[idx07] +=  wx_rite    * wy_left    * w;
            S[idx08] +=  wx_rite_p1 * wy_left    * w;
			
			S[idx09] +=  wx_left_m1 * wy_rite    * w;
            S[idx10] +=  wx_left    * wy_rite    * w;
            S[idx11] +=  wx_rite    * wy_rite    * w;
            S[idx12] +=  wx_rite_p1 * wy_rite    * w;
			
			S[idx13] +=  wx_left_m1 * wy_rite_p1 * w;
            S[idx14] +=  wx_left    * wy_rite_p1 * w;
            S[idx15] +=  wx_rite    * wy_rite_p1 * w;
            S[idx16] +=  wx_rite_p1 * wy_rite_p1 * w;
        }
        for (int i = 0; i < Nx*Ny; i++) {
            fields[f][i] = S[i];
            S[i] = 0;
        }
    }
}

Interpolate::InterpolateMethod CubicInterpolate::getMethod() { return Interpolate::Cubic; }
