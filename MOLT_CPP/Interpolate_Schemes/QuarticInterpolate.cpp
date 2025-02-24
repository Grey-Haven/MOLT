#include "QuarticInterpolate.h"
#include "Interpolate.h"
#include <iostream>
#include <iomanip>

QuarticInterpolate::QuarticInterpolate(int Nx, int Ny, double* x, double* y) : Interpolate(Nx, Ny, x, y) { }

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
void QuarticInterpolate::gatherFields(std::complex<double>** fields,
                                    std::vector<double> px, std::vector<double> py,
                                    int N_fields, int N_particles,
                                    std::vector<std::vector<std::complex<double>>>& vals) {

    #pragma omp parallel for
    for (int p = 0; p < N_particles; p++) {

        double x_p = px[p];
        double y_p = py[p];

        // We convert from cartesian to logical space
        int lc_x_mid = int(round((x_p - x[0])/dx));
        int lc_y_mid = int(round((y_p - y[0])/dy));

        double xNode_mid = this->x[lc_x_mid];
        double yNode_mid = this->y[lc_y_mid];

        double xNode_left_m2 = xNode_mid - 2.0*dx;
        double xNode_left_m1 = xNode_mid - dx;
        double xNode_rite_p1 = xNode_mid + dx;
        double xNode_rite_p2 = xNode_mid + 2.0*dx;

        double yNode_left_m2 = yNode_mid - 2.0*dy;
        double yNode_left_m1 = yNode_mid - dy;
        double yNode_rite_p1 = yNode_mid + dy;
        double yNode_rite_p2 = yNode_mid + 2.0*dy;

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        double fx_left_m2 = (x_p - xNode_left_m2)/dx;
        double fx_left_m1 = (x_p - xNode_left_m1)/dx;
        double fx_mid     = (x_p - xNode_mid    )/dx;
        double fx_rite_p1 = (x_p - xNode_rite_p1)/dx;
        double fx_rite_p2 = (x_p - xNode_rite_p2)/dx;

        double fy_left_m2 = (y_p - yNode_left_m2)/dy;
        double fy_left_m1 = (y_p - yNode_left_m1)/dy;
        double fy_mid     = (y_p - yNode_mid    )/dy;
        double fy_rite_p1 = (y_p - yNode_rite_p1)/dy;
        double fy_rite_p2 = (y_p - yNode_rite_p2)/dy;

        double wx_left_m2 = m2(fx_left_m2);
        double wx_left_m1 = m1(fx_left_m1);
        double wx_mid     = center(fx_mid);
        double wx_rite_p1 = m1(fx_rite_p1);
        double wx_rite_p2 = m2(fx_rite_p2);

        double wy_left_m2 = m2(fy_left_m2);
        double wy_left_m1 = m1(fy_left_m1);
        double wy_mid     = center(fy_mid);
        double wy_rite_p1 = m1(fy_rite_p1);
        double wy_rite_p2 = m2(fy_rite_p2);

        const int lc_x_left_m2 = (lc_x_mid - 2 + Nx) % Nx;
        const int lc_x_left_m1 = (lc_x_mid - 1 + Nx) % Nx;
                  lc_x_mid     = (lc_x_mid     + Nx) % Nx;
        const int lc_x_rite_p1 = (lc_x_mid + 1 + Nx) % Nx;
        const int lc_x_rite_p2 = (lc_x_mid + 2 + Nx) % Nx;

        const int lc_y_left_m2 = (lc_y_mid - 2 + Ny) % Ny;
        const int lc_y_left_m1 = (lc_y_mid - 1 + Ny) % Ny;
                    lc_y_mid   = (lc_y_mid     + Ny) % Ny;
        const int lc_y_rite_p1 = (lc_y_mid + 1 + Ny) % Ny;
        const int lc_y_rite_p2 = (lc_y_mid + 2 + Ny) % Ny;

        /*
        * F21  F22  F23  F24  F25
        * F16  F17  F18  F19  F20
        * F11  F12  F13  F14  F15
        * F06  F07  F08  F09  F10
        * F01  F02  F03  F04  F05
        */

        const int idx01 = computeIndex(lc_x_left_m2, lc_y_left_m2);
        const int idx02 = computeIndex(lc_x_left_m1, lc_y_left_m2);
        const int idx03 = computeIndex(lc_x_mid    , lc_y_left_m2);
        const int idx04 = computeIndex(lc_x_rite_p1, lc_y_left_m2);
        const int idx05 = computeIndex(lc_x_rite_p2, lc_y_left_m2);
        
        const int idx06 = computeIndex(lc_x_left_m2, lc_y_left_m1);
        const int idx07 = computeIndex(lc_x_left_m1, lc_y_left_m1);
        const int idx08 = computeIndex(lc_x_mid    , lc_y_left_m1);
        const int idx09 = computeIndex(lc_x_rite_p1, lc_y_left_m1);
        const int idx10 = computeIndex(lc_x_rite_p2, lc_y_left_m1);
        
        const int idx11 = computeIndex(lc_x_left_m2, lc_y_mid);
        const int idx12 = computeIndex(lc_x_left_m1, lc_y_mid);
        const int idx13 = computeIndex(lc_x_mid    , lc_y_mid);
        const int idx14 = computeIndex(lc_x_rite_p1, lc_y_mid);
        const int idx15 = computeIndex(lc_x_rite_p2, lc_y_mid);
        
        const int idx16 = computeIndex(lc_x_left_m2, lc_y_rite_p1);
        const int idx17 = computeIndex(lc_x_left_m1, lc_y_rite_p1);
        const int idx18 = computeIndex(lc_x_mid    , lc_y_rite_p1);
        const int idx19 = computeIndex(lc_x_rite_p1, lc_y_rite_p1);
        const int idx20 = computeIndex(lc_x_rite_p2, lc_y_rite_p1);
        
        const int idx21 = computeIndex(lc_x_left_m2, lc_y_rite_p2);
        const int idx22 = computeIndex(lc_x_left_m1, lc_y_rite_p2);
        const int idx23 = computeIndex(lc_x_mid    , lc_y_rite_p2);
        const int idx24 = computeIndex(lc_x_rite_p1, lc_y_rite_p2);
        const int idx25 = computeIndex(lc_x_rite_p2, lc_y_rite_p2);

        for (int f = 0; f < N_fields; f++) {
            vals[f][p] += wx_left_m2 * wy_left_m2 * fields[f][idx01];
            vals[f][p] += wx_left_m1 * wy_left_m2 * fields[f][idx02];
            vals[f][p] += wx_mid     * wy_left_m2 * fields[f][idx03];
            vals[f][p] += wx_rite_p1 * wy_left_m2 * fields[f][idx04];
            vals[f][p] += wx_rite_p2 * wy_left_m2 * fields[f][idx05];

            vals[f][p] += wx_left_m2 * wy_left_m1 * fields[f][idx06];
            vals[f][p] += wx_left_m1 * wy_left_m1 * fields[f][idx07];
            vals[f][p] += wx_mid     * wy_left_m1 * fields[f][idx08];
            vals[f][p] += wx_rite_p1 * wy_left_m1 * fields[f][idx09];
            vals[f][p] += wx_rite_p2 * wy_left_m1 * fields[f][idx10];

            vals[f][p] += wx_left_m2 * wy_mid * fields[f][idx11];
            vals[f][p] += wx_left_m1 * wy_mid * fields[f][idx12];
            vals[f][p] += wx_mid     * wy_mid * fields[f][idx13];
            vals[f][p] += wx_rite_p1 * wy_mid * fields[f][idx14];
            vals[f][p] += wx_rite_p2 * wy_mid * fields[f][idx15];

            vals[f][p] += wx_left_m2 * wy_rite_p1 * fields[f][idx16];
            vals[f][p] += wx_left_m1 * wy_rite_p1 * fields[f][idx17];
            vals[f][p] += wx_mid     * wy_rite_p1 * fields[f][idx18];
            vals[f][p] += wx_rite_p1 * wy_rite_p1 * fields[f][idx19];
            vals[f][p] += wx_rite_p2 * wy_rite_p1 * fields[f][idx20];

            vals[f][p] += wx_left_m2 * wy_rite_p2 * fields[f][idx21];
            vals[f][p] += wx_left_m1 * wy_rite_p2 * fields[f][idx22];
            vals[f][p] += wx_mid     * wy_rite_p2 * fields[f][idx23];
            vals[f][p] += wx_rite_p1 * wy_rite_p2 * fields[f][idx24];
            vals[f][p] += wx_rite_p2 * wy_rite_p2 * fields[f][idx25];
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
void QuarticInterpolate::scatterParticles(std::complex<double>** fields,
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
            int lc_x_mid = int(round((x_p - x[0])/dx));
            int lc_y_mid = int(round((y_p - y[0])/dy));

            double xNode_mid = this->x[lc_x_mid];
            double yNode_mid = this->y[lc_y_mid];

            double xNode_left_m2 = xNode_mid - 2.0*dx;
            double xNode_left_m1 = xNode_mid - dx;
            double xNode_rite_p1 = xNode_mid + dx;
            double xNode_rite_p2 = xNode_mid + 2.0*dx;

            double yNode_left_m2 = yNode_mid - 2.0*dy;
            double yNode_left_m1 = yNode_mid - dy;
            double yNode_rite_p1 = yNode_mid + dy;
            double yNode_rite_p2 = yNode_mid + 2.0*dy;

            // We compute the fractional distance of a particle from
            // the nearest node.
            // eg x=[0,.1,.2,.3], particleX = [.225]
            // The particle's fractional is 1/4
            double fx_left_m2 = (x_p - xNode_left_m2)/dx;
            double fx_left_m1 = (x_p - xNode_left_m1)/dx;
            double fx_mid     = (x_p - xNode_mid    )/dx;
            double fx_rite_p1 = (x_p - xNode_rite_p1)/dx;
            double fx_rite_p2 = (x_p - xNode_rite_p2)/dx;

            double fy_left_m2 = (y_p - yNode_left_m2)/dy;
            double fy_left_m1 = (y_p - yNode_left_m1)/dy;
            double fy_mid     = (y_p - yNode_mid    )/dy;
            double fy_rite_p1 = (y_p - yNode_rite_p1)/dy;
            double fy_rite_p2 = (y_p - yNode_rite_p2)/dy;

            double wx_left_m2 = m2(fx_left_m2);
            double wx_left_m1 = m1(fx_left_m1);
			double wx_mid     = center(fx_mid);
            double wx_rite_p1 = m1(fx_rite_p1);
            double wx_rite_p2 = m2(fx_rite_p2);

            double wy_left_m2 = m2(fy_left_m2);
            double wy_left_m1 = m1(fy_left_m1);
			double wy_mid     = center(fy_mid);
            double wy_rite_p1 = m1(fy_rite_p1);
            double wy_rite_p2 = m2(fy_rite_p2);

            const int lc_x_left_m2 = (lc_x_mid - 2 + Nx) % Nx;
            const int lc_x_left_m1 = (lc_x_mid - 1 + Nx) % Nx;
                      lc_x_mid     = (lc_x_mid     + Nx) % Nx;
            const int lc_x_rite_p1 = (lc_x_mid + 1 + Nx) % Nx;
            const int lc_x_rite_p2 = (lc_x_mid + 2 + Nx) % Nx;

            const int lc_y_left_m2 = (lc_y_mid - 2 + Ny) % Ny;
            const int lc_y_left_m1 = (lc_y_mid - 1 + Ny) % Ny;
                      lc_y_mid     = (lc_y_mid     + Ny) % Ny;
            const int lc_y_rite_p1 = (lc_y_mid + 1 + Ny) % Ny;
            const int lc_y_rite_p2 = (lc_y_mid + 2 + Ny) % Ny;

            /*
            * F21  F22  F23  F24  F25
            * F16  F17  F18  F19  F20
            * F11  F12  F13  F14  F15
            * F06  F07  F08  F09  F10
            * F01  F02  F03  F04  F05
            */

            const int idx01 = computeIndex(lc_x_left_m2, lc_y_left_m2);
            const int idx02 = computeIndex(lc_x_left_m1, lc_y_left_m2);
            const int idx03 = computeIndex(lc_x_mid    , lc_y_left_m2);
            const int idx04 = computeIndex(lc_x_rite_p1, lc_y_left_m2);
            const int idx05 = computeIndex(lc_x_rite_p2, lc_y_left_m2);
            
            const int idx06 = computeIndex(lc_x_left_m2, lc_y_left_m1);
            const int idx07 = computeIndex(lc_x_left_m1, lc_y_left_m1);
            const int idx08 = computeIndex(lc_x_mid    , lc_y_left_m1);
            const int idx09 = computeIndex(lc_x_rite_p1, lc_y_left_m1);
            const int idx10 = computeIndex(lc_x_rite_p2, lc_y_left_m1);
            
            const int idx11 = computeIndex(lc_x_left_m2, lc_y_mid);
            const int idx12 = computeIndex(lc_x_left_m1, lc_y_mid);
            const int idx13 = computeIndex(lc_x_mid    , lc_y_mid);
            const int idx14 = computeIndex(lc_x_rite_p1, lc_y_mid);
            const int idx15 = computeIndex(lc_x_rite_p2, lc_y_mid);
            
            const int idx16 = computeIndex(lc_x_left_m2, lc_y_rite_p1);
            const int idx17 = computeIndex(lc_x_left_m1, lc_y_rite_p1);
            const int idx18 = computeIndex(lc_x_mid    , lc_y_rite_p1);
            const int idx19 = computeIndex(lc_x_rite_p1, lc_y_rite_p1);
            const int idx20 = computeIndex(lc_x_rite_p2, lc_y_rite_p1);
            
            const int idx21 = computeIndex(lc_x_left_m2, lc_y_rite_p2);
            const int idx22 = computeIndex(lc_x_left_m1, lc_y_rite_p2);
            const int idx23 = computeIndex(lc_x_mid    , lc_y_rite_p2);
            const int idx24 = computeIndex(lc_x_rite_p1, lc_y_rite_p2);
            const int idx25 = computeIndex(lc_x_rite_p2, lc_y_rite_p2);

            double w = weights[f][p];

            S[idx01] +=  wx_left_m2 * wy_left_m2 * w;
            S[idx02] +=  wx_left_m1 * wy_left_m2 * w;
            S[idx03] +=  wx_mid     * wy_left_m2 * w;
            S[idx04] +=  wx_rite_p1 * wy_left_m2 * w;
            S[idx05] +=  wx_rite_p2 * wy_left_m2 * w;

            S[idx06] +=  wx_left_m2 * wy_left_m1 * w;
            S[idx07] +=  wx_left_m1 * wy_left_m1 * w;
            S[idx08] +=  wx_mid     * wy_left_m1 * w;
            S[idx09] +=  wx_rite_p1 * wy_left_m1 * w;
            S[idx10] +=  wx_rite_p2 * wy_left_m1 * w;

            S[idx11] +=  wx_left_m2 * wy_mid * w;
            S[idx12] +=  wx_left_m1 * wy_mid * w;
            S[idx13] +=  wx_mid     * wy_mid * w;
            S[idx14] +=  wx_rite_p1 * wy_mid * w;
            S[idx15] +=  wx_rite_p2 * wy_mid * w;

            S[idx16] +=  wx_left_m2 * wy_rite_p1 * w;
            S[idx17] +=  wx_left_m1 * wy_rite_p1 * w;
            S[idx18] +=  wx_mid     * wy_rite_p1 * w;
            S[idx19] +=  wx_rite_p1 * wy_rite_p1 * w;
            S[idx20] +=  wx_rite_p2 * wy_rite_p1 * w;

            S[idx21] +=  wx_left_m2 * wy_rite_p2 * w;
            S[idx22] +=  wx_left_m1 * wy_rite_p2 * w;
            S[idx23] +=  wx_mid     * wy_rite_p2 * w;
            S[idx24] +=  wx_rite_p1 * wy_rite_p2 * w;
            S[idx25] +=  wx_rite_p2 * wy_rite_p2 * w;
        }
        for (int i = 0; i < Nx*Ny; i++) {
            fields[f][i] = S[i];
            S[i] = 0;
        }
    }
}

Interpolate::InterpolateMethod QuarticInterpolate::getMethod() { return Interpolate::Quartic; }
