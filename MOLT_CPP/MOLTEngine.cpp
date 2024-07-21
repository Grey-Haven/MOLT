#include <cmath>
#include <iostream>
#include <fstream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <complex.h>
#include <fftw3.h>
#include <vector>
#include <sys/time.h>

#include <stdio.h>
#include <omp.h>
#include "MOLTEngine.h"

/**
 * Name: step
 * Author: Stephen White
 * Date Created: 9/28/22
 * Date Last Modified: 9/28/22 (Stephen White)
 * Description: Runs a single timestep iteration of a plasma system under the Lorenz gauge. 
 *              The phi, A1, and A2 waves are updated using, for now, Rothe's method under a BDF1 time discretization, 
 *              the particle locations updated using Newton's law with BDF1 computing the time derivative, and the IAEM for the particle velocity update.
 *              For debugging purposes there are timers on each component of the algorithm, eliminating these will result in some time saved.
 * Inputs: NA
 * Output: NA
 * Dependencies: scatterFields, shuffleSteps, updateParticleLocations, updateParticleVelocities, updateWaves
 */
void MOLTEngine::step() {
    // std::cout << "Updating Particle Locations" << std::endl;
    struct timeval begin1, end1, begin2, end2, begin3, end3, begin4, end4, begin5, end5, begin6, end6, begin7, end7;
    gettimeofday( &begin1, NULL );
    updateParticleLocations();
    gettimeofday( &end1, NULL );
    // std::cout << "Scattering Fields" << std::endl;
    gettimeofday( &begin2, NULL );
    scatterFields();
    gettimeofday( &end2, NULL );
    // std::cout << "Updating Waves" << std::endl;
    gettimeofday( &begin3, NULL );
    updateWaves();
    gettimeofday( &end3, NULL );
    // std::cout << "Updating Particle Velocities" << std::endl;
    gettimeofday( &begin4, NULL );
    updateParticleVelocities();
    gettimeofday( &end4, NULL );
    gettimeofday( &begin5, NULL );
    computePhysicalDiagnostics();
    gettimeofday( &end5, NULL );
    if (n % 500 == 0) {
        gettimeofday( &begin7, NULL );
        print();
        gettimeofday( &end7, NULL );
        timeComponent7 += 1.0 * ( end7.tv_sec - begin7.tv_sec ) + 1.0e-6 * ( end7.tv_usec - begin7.tv_usec );
    }
    // std::cout << "Shuffling Steps" << std::endl;
    gettimeofday( &begin6, NULL );
    shuffleSteps();
    gettimeofday( &end6, NULL );
    // std::cout << "Rinse, Repeat" << std::endl;
    timeComponent1 += 1.0 * ( end1.tv_sec - begin1.tv_sec ) + 1.0e-6 * ( end1.tv_usec - begin1.tv_usec );
    timeComponent2 += 1.0 * ( end2.tv_sec - begin2.tv_sec ) + 1.0e-6 * ( end2.tv_usec - begin2.tv_usec );
    timeComponent3 += 1.0 * ( end3.tv_sec - begin3.tv_sec ) + 1.0e-6 * ( end3.tv_usec - begin3.tv_usec );
    timeComponent4 += 1.0 * ( end4.tv_sec - begin4.tv_sec ) + 1.0e-6 * ( end4.tv_usec - begin4.tv_usec );
    timeComponent5 += 1.0 * ( end5.tv_sec - begin5.tv_sec ) + 1.0e-6 * ( end5.tv_usec - begin5.tv_usec );
    timeComponent6 += 1.0 * ( end6.tv_sec - begin6.tv_sec ) + 1.0e-6 * ( end6.tv_usec - begin6.tv_usec );
    n++;
    t += dt;
}

void MOLTEngine::printTimeDiagnostics() {
    std::cout << "updateParticleLocations(): " << timeComponent1 << std::endl;
    std::cout << "scatterFields(): " << timeComponent2 << std::endl;
    std::cout << "updateWaves(): " << timeComponent3 << std::endl;
    std::cout << "updateParticleVelocities(): " << timeComponent4 << std::endl;
    std::cout << "computePhysicalDiagnostics() + : " << timeComponent5 << std::endl;
    std::cout << "shuffleSteps(): " << timeComponent6 << std::endl;
    std::cout << "print(): " << timeComponent7 << std::endl;
}

double MOLTEngine::getTime() {
    return t;
}

int MOLTEngine::getStep() {
    return n;
}

double MOLTEngine::getGaugeL2() {
    return gaugeL2;
}

double MOLTEngine::getGaussL2_divE() {
    return gaussL2_divE;
}

double MOLTEngine::getGaussL2_divA() {
    return gaussL2_divA;
}

double MOLTEngine::getGaussL2_wave() {
    return gaussL2_wave;
}

double MOLTEngine::getTotalCharge() {
    return rhoTotal;
}

double MOLTEngine::getTotalEnergy() {
    return totalEnergy;
}

double MOLTEngine::getTotalMass() {
    return totalMass;
}

double MOLTEngine::getTemperature() {
    return temperature;
}

void MOLTEngine::computePhysicalDiagnostics() {
    computeTotalEnergy();
    computeTotalMass();
    computeGaugeL2();
    computeGaussL2();
    computeTemperature();
}

void MOLTEngine::computeTemperature() {
    double sum_vx = 0;
    double sum_vy = 0;
    for (int p = 0; p < numElectrons; p++) {
        sum_vx += (*vx_elec[lastStepIndex])[p];
        sum_vy += (*vy_elec[lastStepIndex])[p];
    }
    double mean_vx = sum_vx / numElectrons;
    double mean_vy = sum_vy / numElectrons;

    double var_vx = 0;
    double var_vy = 0;
    for (int p = 0; p < numElectrons; p++) {
        var_vx += std::pow((*vx_elec[lastStepIndex])[p] - mean_vx, 2);
        var_vy += std::pow((*vy_elec[lastStepIndex])[p] - mean_vy, 2);
    }
    double std_dev_vx = std::sqrt(var_vx / numElectrons);
    double std_dev_vy = std::sqrt(var_vy / numElectrons);

    this->temperature = (std_dev_vx + std_dev_vy) / 2.0;
}

// Try splitting into kinetic and potential energy
void MOLTEngine::computeTotalEnergy() {

    this->totalEnergy = 0;
    this->eleTotalEnergy = 0;
    this->ionTotalEnergy = 0;

    double S = 0;

    // Beware race condition
    #pragma omp parallel for reduction(+:S)
    for (int i = 0; i < numElectrons; i++) {
        double phi_p = 0;
        double A1_p = 0;
        double A2_p = 0;

        double P1_p = (*Px_elec[lastStepIndex])[i];
        double P2_p = (*Py_elec[lastStepIndex])[i];

        const double p_x = (*x_elec[lastStepIndex])[i];
        const double p_y = (*y_elec[lastStepIndex])[i];
        // ------------------------------
        // Gather Fields
        // We convert from cartesian to logical space
        const double x0 = this->x[0];
        const double y0 = this->y[0];
        const int lc_x = floor((p_x - x0)/dx);
        const int lc_y = floor((p_y - y0)/dy);

        const int lc_x_p1 = (lc_x+1) % Nx;
        const int lc_y_p1 = (lc_y+1) % Ny;

        const int ld = lc_x * Ny + lc_y;          // (left, down)  lc_x,   lc_y
        const int lu = lc_x * Ny + lc_y_p1;       // (left, up)    lc_x,   lc_y+1
        const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down)  lc_x+1, lc_y
        const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)    lc_x+1, lc_y+1

        const double xNode = this->x[lc_x];
        const double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        const double fx = (p_x - xNode)/dx;
        const double fy = (p_y - yNode)/dy;

        const double w_ld = (1-fx)*(1-fy);
        const double w_lu = (1-fx)*(fy);
        const double w_rd = (fx)*(1-fy);
        const double w_ru = (fx)*(fy);

        if (this->method == MOLTEngine::CDF1) {
            phi_p += w_ld*( phi[lastStepIndex][ld].real() + phi[lastStepIndex-1][ld].real() ) / 2.0;
            phi_p += w_lu*( phi[lastStepIndex][lu].real() + phi[lastStepIndex-1][lu].real() ) / 2.0;
            phi_p += w_rd*( phi[lastStepIndex][rd].real() + phi[lastStepIndex-1][rd].real() ) / 2.0;
            phi_p += w_ru*( phi[lastStepIndex][ru].real() + phi[lastStepIndex-1][ru].real() ) / 2.0;
        } else {
            phi_p += w_ld*phi[lastStepIndex][ld].real();
            phi_p += w_lu*phi[lastStepIndex][lu].real();
            phi_p += w_rd*phi[lastStepIndex][rd].real();
            phi_p += w_ru*phi[lastStepIndex][ru].real();
        }

        A1_p += w_ld*A1[lastStepIndex][ld].real();
        A1_p += w_lu*A1[lastStepIndex][lu].real();
        A1_p += w_rd*A1[lastStepIndex][rd].real();
        A1_p += w_ru*A1[lastStepIndex][ru].real();

        A2_p += w_ld*A2[lastStepIndex][ld].real();
        A2_p += w_lu*A2[lastStepIndex][lu].real();
        A2_p += w_rd*A2[lastStepIndex][rd].real();
        A2_p += w_ru*A2[lastStepIndex][ru].real();

        double electron_energy  = std::pow(P1_p - q_ele*A1_p, 2);
               electron_energy += std::pow(P2_p - q_ele*A2_p, 2);
               electron_energy *= 1.0/(2.0*m_ele);
               electron_energy += q_ele*phi_p;
        S += electron_energy;
    }
    this->eleTotalEnergy = S;

    S = 0;

    #pragma omp parallel for reduction(+:S)
    for (int i = 0; i < numIons; i++) {

        double phi_p = 0;
        double A1_p = 0;
        double A2_p = 0;

        // Ions are stationary
        const double P1_p = 0;
        const double P2_p = 0;

        const double p_x = x_ion[i];
        const double p_y = y_ion[i];
        // ------------------------------
        // Gather Fields
        // We convert from cartesian to logical space
        const double x0 = this->x[0];
        const double y0 = this->y[0];
        const int lc_x = floor((p_x - x0)/dx);
        const int lc_y = floor((p_y - y0)/dy);

        const int lc_x_p1 = (lc_x+1) % Nx;
        const int lc_y_p1 = (lc_y+1) % Ny;

        const int ld = lc_x * Ny + lc_y;          // (left, down)  lc_x,   lc_y
        const int lu = lc_x * Ny + lc_y_p1;       // (left, up)    lc_x,   lc_y+1
        const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down)  lc_x+1, lc_y
        const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)    lc_x+1, lc_y+1

        const double xNode = this->x[lc_x];
        const double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        const double fx = (p_x - xNode)/dx;
        const double fy = (p_y - yNode)/dy;

        const double w_ld = (1-fx)*(1-fy);
        const double w_lu = (1-fx)*(fy);
        const double w_rd = (fx)*(1-fy);
        const double w_ru = (fx)*(fy);

        if (this->method == MOLTEngine::CDF1) {
            phi_p += w_ld*( phi[lastStepIndex][ld].real() + phi[lastStepIndex-1][ld].real() ) / 2;
            phi_p += w_lu*( phi[lastStepIndex][lu].real() + phi[lastStepIndex-1][lu].real() ) / 2;
            phi_p += w_rd*( phi[lastStepIndex][rd].real() + phi[lastStepIndex-1][rd].real() ) / 2;
            phi_p += w_ru*( phi[lastStepIndex][ru].real() + phi[lastStepIndex-1][ru].real() ) / 2;
        } else {
            phi_p += w_ld*phi[lastStepIndex][ld].real();
            phi_p += w_lu*phi[lastStepIndex][lu].real();
            phi_p += w_rd*phi[lastStepIndex][rd].real();
            phi_p += w_ru*phi[lastStepIndex][ru].real();
        }

        A1_p += w_ld*A1[lastStepIndex][ld].real();
        A1_p += w_lu*A1[lastStepIndex][lu].real();
        A1_p += w_rd*A1[lastStepIndex][rd].real();
        A1_p += w_ru*A1[lastStepIndex][ru].real();

        A2_p += w_ld*A2[lastStepIndex][ld].real();
        A2_p += w_lu*A2[lastStepIndex][lu].real();
        A2_p += w_rd*A2[lastStepIndex][rd].real();
        A2_p += w_ru*A2[lastStepIndex][ru].real();

        double ion_energy  = std::pow(P1_p - q_ion*A1_p, 2);
               ion_energy += std::pow(P2_p - q_ion*A2_p, 2);
               ion_energy *= 1.0/(2.0*m_ion);
               ion_energy += q_ion*phi_p;
        S += ion_energy;
    }
    this->ionTotalEnergy = S;

    this->totalEnergy = ionTotalEnergy + eleTotalEnergy;

}

void MOLTEngine::computeTotalMass() {
    // Total charge is the charge density in each cell times the cell volume, which
    // followed by a sum over the cells
    std::vector<double> ele_charge(Nx*Ny);
    for (int i = 0; i < Nx*Ny; i++) {
        ele_charge[i] = dx*dy*(rho[lastStepIndex][i].real() - rho_ions[i].real());
    }
    
    // Total mass is the total charge divided by charge per particle times the mass of each particle    
    this->eleTotalMass = 0;
    for (int i = 0; i < Nx*Ny; i++) {
        this->eleTotalMass += m_ele*(ele_charge[i]/q_ele);
    }

    std::vector<double> ion_charge(Nx*Ny);
    for (int i = 0; i < Nx*Ny; i++) {
        ion_charge[i] = dx*dy*(rho[lastStepIndex][i].real() - rho_eles[i].real());
    }
    
    // Total mass is the total charge divided by charge per particle times the mass of each particle    
    this->ionTotalMass = 0;
    for (int i = 0; i < Nx*Ny; i++) {
        this->ionTotalMass += m_ion*(ion_charge[i]/q_ion);
    }

    this->totalMass = ionTotalMass + eleTotalMass;

}

/**
 * Name: computeGaugeL2
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Computes the L2 error of the residual of the Lorenz gauge (eps = 1/k^2 ddt_phi + div(A) = 0)
 * Inputs: none (relies on phi, ddx_A1, ddy_A2)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::computeGaugeL2() {
    double ddt_phi;
    double div_A;
    double l2 = 0;
    
    for (int i = 0; i < Nx*Ny; i++) {
        if (this->method == MOLTEngine::DIRK2 || this->method == MOLTEngine::DIRK3) {

            double b1;
            double b2;

            double c1;
            double c2;

            if (this->method == MOLTEngine::DIRK2) {
                // Qin and Zhang's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/4.0;
                c2 = 3.0/4.0;
            } else if (this->method == MOLTEngine::DIRK3) {
                // Crouzeix's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/2.0 + std::sqrt(3.0)/6.0;
                c2 = 1.0/2.0 - std::sqrt(3.0)/6.0;
            }

            ddt_phi = (phi[lastStepIndex][i].real() - phi[lastStepIndex-1][i].real());

            double div_A_prev = ddx_A1[lastStepIndex-1][i].real() + ddy_A2[lastStepIndex-1][i].real();
            double div_A_curr = ddx_A1[lastStepIndex][i].real() + ddy_A2[lastStepIndex][i].real();

            double RHS_1 = (1-c1)*div_A_prev + c1*div_A_curr;
            double RHS_2 = (1-c2)*div_A_prev + c2*div_A_curr;

            div_A = b1*RHS_1 + b2*RHS_2;

            l2 += std::pow(1.0/(kappa*kappa)*ddt_phi + dt*div_A,2);

        } else {

            if (this->method == MOLTEngine::BDF1 ||
                this->method == MOLTEngine::CDF1 ||
                this->method == MOLTEngine::MOLT_BDF1 ||
                this->method == MOLTEngine::MOLT_BDF1_HYBRID_FFT ||
                this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6) {
                this->ddt_phi[1][i] = (phi[lastStepIndex][i].real() - phi[lastStepIndex-1][i].real()) / dt;

                ddt_phi = (phi[lastStepIndex][i].real() - phi[lastStepIndex-1][i].real());
            } else if (this->method == MOLTEngine::BDF2) {
                this->ddt_phi[1][i] = (phi[lastStepIndex][i].real() - (4.0/3.0)*phi[lastStepIndex-1][i].real() + (1.0/3.0)*phi[lastStepIndex-2][i].real()) / (dt*(2.0/3.0));

                ddt_phi = (phi[lastStepIndex][i].real() - (4.0/3.0)*phi[lastStepIndex-1][i].real() + (1.0/3.0)*phi[lastStepIndex-2][i].real()) / ((2.0/3.0));
            } else {
                throw -1;
            }
            div_A = ddx_A1[lastStepIndex][i].real() + ddy_A2[lastStepIndex][i].real();
            l2 += std::pow(1.0/(kappa*kappa)*ddt_phi + dt*div_A,2);
        }
    }
    gaugeL2 = std::sqrt(dx*dy*l2);
}

/**
 * Name: computeGaussL2
 * Author: Stephen White
 * Date Created: 6/14/24
 * Date Last Modified: 7/16/24 (Stephen White)
 * Description: Computes the L2 error of the residual of Gauss's Law for Electricity (div(E) = -grad(phi) - dA/dt = rho/eps_0)
 *              This L2 error comes in three forms, (div(E), -grad(phi) - dA/dt, and 1/c^2*d2/dt2(phi) - laplacian(phi)] - rho/eps_0
 * Inputs: none (relies on phi, ddx_phi, ddy_phi, A1, A2, ddx_A1, ddy_A2)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::computeGaussL2() {

    double ddt_A1;
    double ddt_A2;

    for (int i = 0; i < Nx*Ny; i++) {
        if (this->method == MOLTEngine::BDF1 ||
            this->method == MOLTEngine::CDF1 ||
            this->method == MOLTEngine::MOLT_BDF1 ||
            this->method == MOLTEngine::MOLT_BDF1_HYBRID_FFT ||
            this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6) {
            ddt_A1 = (A1[lastStepIndex][i].real() - A1[lastStepIndex-1][i].real()) / dt;
            ddt_A2 = (A2[lastStepIndex][i].real() - A2[lastStepIndex-1][i].real()) / dt;
        } else if (this->method == MOLTEngine::BDF2) {
            ddt_A1 = (A1[lastStepIndex][i].real() - (4.0/3.0)*A1[lastStepIndex-1][i].real() + (1.0/3.0)*A1[lastStepIndex-2][i].real()) / ((2.0/3.0)*dt);
            ddt_A2 = (A2[lastStepIndex][i].real() - (4.0/3.0)*A2[lastStepIndex-1][i].real() + (1.0/3.0)*A2[lastStepIndex-2][i].real()) / ((2.0/3.0)*dt);
        } else {
            throw -1;
        }
        if (this->method == MOLTEngine::CDF1) {
            E1[i] = -(ddx_phi[lastStepIndex][i].real() + ddx_phi[lastStepIndex][i].real())/2.0 - ddt_A1;
            E1[i] = -(ddy_phi[lastStepIndex][i].real() + ddy_phi[lastStepIndex][i].real())/2.0 - ddt_A2;
        } else {
            E1[i] = -ddx_phi[lastStepIndex][i].real() - ddt_A1;
            E2[i] = -ddy_phi[lastStepIndex][i].real() - ddt_A2;
        }
    }

    if (this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6) {
        compute_ddx_FD6(E1, ddx_E1);
        compute_ddy_FD6(E2, ddy_E2);
    } else {
        compute_ddx_FFT(E1, ddx_E1);
        compute_ddy_FFT(E2, ddy_E2);
    }

    compute_d2dx(phi[lastStepIndex], d2dx_phi_curr);
    compute_d2dy(phi[lastStepIndex], d2dy_phi_curr);

    if (this->method == MOLTEngine::DIRK2) {

        compute_d2dx(phi[lastStepIndex-1], d2dx_phi_prev);
        compute_d2dy(phi[lastStepIndex-1], d2dy_phi_prev);

        for (int i = 0; i < Nx*Ny; i++) {
            double rho_prev = rho[lastStepIndex][i-1].real();
            double rho_curr = rho[lastStepIndex][i  ].real();
            gauss_RHS[i] = dirk_qin_zhang_rhs(rho_prev, rho_curr) / sigma_1;

            double laplacian_phi_prev = d2dx_phi_curr[i-1].real() + d2dy_phi_curr[i-1].real();
            double laplacian_phi_curr = d2dx_phi_curr[i  ].real() + d2dy_phi_curr[i  ].real();
            laplacian_phi[i] = dirk_qin_zhang_rhs(laplacian_phi_prev, laplacian_phi_curr);

            double div_A_prev = ddx_A1[lastStepIndex-1][i].real() + ddy_A2[lastStepIndex-1][i].real();
            double div_A_curr = ddx_A1[lastStepIndex  ][i].real() + ddy_A2[lastStepIndex  ][i].real();
            ddt_divA_curr[i] = (div_A_curr - div_A_prev) / dt;
        }
    } else {

        for (int i = 0; i < Nx*Ny; i++) {
            gauss_RHS[i] = rho[lastStepIndex][i].real() / sigma_1;
        }

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_phi[i] = d2dx_phi_curr[i] + d2dy_phi_curr[i];
        }

        for (int i = 0; i < Nx*Ny; i++) {
            if (this->method == MOLTEngine::BDF1 ||
                this->method == MOLTEngine::MOLT_BDF1 ||
                this->method == MOLTEngine::DIRK2 ||
                this->method == MOLTEngine::MOLT_BDF1_HYBRID_FFT ||
                this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6) {
                d2dt_phi[i] = (phi[lastStepIndex][i] - 2.0*phi[lastStepIndex-1][i] + phi[lastStepIndex-2][i]).real()/(dt*dt);
            } else if (this->method == MOLTEngine::BDF2) {
                d2dt_phi[i] = (phi[lastStepIndex][i] - 8.0/3.0*phi[lastStepIndex][i] + 22.0/9.0*phi[lastStepIndex][i] - 8.0/9.0*phi[lastStepIndex][i] + 1.0/9.0*phi[lastStepIndex][i]).real() / std::pow(2.0/3.0*dt, 2);
            }
        }

        for (int i = 0; i < Nx*Ny; i++) {
            double div_A_curr = ddx_A1[lastStepIndex][i].real() + ddy_A2[lastStepIndex][i].real();
            double div_A_prev = ddx_A1[lastStepIndex-1][i].real() + ddy_A2[lastStepIndex-1][i].real();
            ddt_divA_curr[i] = (div_A_curr - div_A_prev) / dt;
        }
    }

    double l2 = 0;
    // res = div(E) - rho/sigma_1 (sigma_1 nondimensionalized eps_0)
    for (int i = 0; i < Nx*Ny; i++) {
        l2 += std::pow(ddx_E1[i].real() + ddy_E2[i].real() - gauss_RHS[i], 2);
    }
    gaussL2_divE = std::sqrt(dx*dy*l2);

    l2 = 0;
    // res = d^2phi/dt^2 - laplacian_phi - rho/sigma_1 (sigma_1 nondimensionalized eps_0)
    for (int i = 0; i < Nx*Ny; i++) {
        l2 += std::pow(1.0/(kappa*kappa) * d2dt_phi[i].real() - laplacian_phi[i].real() - gauss_RHS[i], 2);
    }
    gaussL2_wave = std::sqrt(dx*dy*l2);

    l2 = 0;
    // res = -d/dt[div(A)] - laplacian_phi - rho/sigma_1 (sigma_1 nondimensionalized eps_0)
    for (int i = 0; i < Nx*Ny; i++) {
        l2 += std::pow(-ddt_divA_curr[i].real() - laplacian_phi[i].real() - gauss_RHS[i], 2);
    }
    gaussL2_divA = std::sqrt(dx*dy*l2);
}

/**
 * Name: print
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: This prints the potentials and particle information to their own files grouped by mesh refinement, labeled by field and timestep
 * Inputs: none (relies on the field and particle arrays)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::print() {
    std::ofstream phiFile, A1File, A2File;
    std::ofstream ddx_phiFile, ddx_A1File, ddx_A2File;
    std::ofstream ddy_phiFile, ddy_A1File, ddy_A2File;
    std::ofstream rhoFile, J1File, J2File;
    std::ofstream ddt_phiFile;
    std::ofstream electronFile;

    std::string nstr = std::to_string(n);
    int numlen = 5;

    std::string electronFileName = snapshotPath + "/particles.csv";
    std::string phiFileName = snapshotPath + "/phi.csv";
    std::string A1FileName = snapshotPath + "/A1.csv";
    std::string A2FileName = snapshotPath + "/A2.csv";
    std::string ddx_phiFileName = snapshotPath + "/ddx_phi.csv";
    std::string ddy_phiFileName = snapshotPath + "/ddy_phi.csv";
    std::string ddx_A1FileName = snapshotPath + "/ddx_A1.csv";
    std::string ddy_A1FileName = snapshotPath + "/ddy_A1.csv";
    std::string ddx_A2FileName = snapshotPath + "/ddx_A2.csv";
    std::string ddy_A2FileName = snapshotPath + "/ddy_A2.csv";
    std::string rhoFileName = snapshotPath + "/rho.csv";
    std::string J1FileName = snapshotPath + "/J1.csv";
    std::string J2FileName = snapshotPath + "/J2.csv";
    std::string ddt_phiFileName = snapshotPath + "/ddt_phi.csv";
    
    std::ostringstream padder;
    padder << std::internal << std::setfill('0') << std::setw(numlen) << n;
    std::string paddedNum = padder.str();

    electronFile.open(electronFileName, std::ios_base::app);

    phiFile.open(phiFileName, std::ios_base::app);
    A1File.open(A1FileName, std::ios_base::app);
    A2File.open(A2FileName, std::ios_base::app);
    
    ddx_phiFile.open(ddx_phiFileName, std::ios_base::app);
    ddy_phiFile.open(ddy_phiFileName, std::ios_base::app);

    ddx_A1File.open(ddx_A1FileName, std::ios_base::app);
    ddy_A1File.open(ddy_A1FileName, std::ios_base::app);

    ddx_A2File.open(ddx_A2FileName, std::ios_base::app);
    ddy_A2File.open(ddy_A2FileName, std::ios_base::app);

    rhoFile.open(rhoFileName, std::ios_base::app);
    J1File.open(J1FileName, std::ios_base::app);
    J2File.open(J2FileName, std::ios_base::app);
    ddt_phiFile.open(ddt_phiFileName, std::ios_base::app);
    
    electronFile << std::setprecision(16);
    phiFile      << std::setprecision(16);
    A1File       << std::setprecision(16);
    A2File       << std::setprecision(16);
    ddx_phiFile  << std::setprecision(16);
    ddx_A1File   << std::setprecision(16);
    ddx_A2File   << std::setprecision(16);
    ddy_phiFile  << std::setprecision(16);
    ddy_A1File   << std::setprecision(16);
    ddy_A2File   << std::setprecision(16);
    rhoFile      << std::setprecision(16);
    J1File       << std::setprecision(16);
    J2File       << std::setprecision(16);
    ddt_phiFile  << std::setprecision(16);
    
    electronFile << std::endl << t << std::endl;
    phiFile      << std::endl << t << std::endl;
    A1File       << std::endl << t << std::endl;
    A2File       << std::endl << t << std::endl;
    ddx_phiFile  << std::endl << t << std::endl;
    ddx_A1File   << std::endl << t << std::endl;
    ddx_A2File   << std::endl << t << std::endl;
    ddy_phiFile  << std::endl << t << std::endl;
    ddy_A1File   << std::endl << t << std::endl;
    ddy_A2File   << std::endl << t << std::endl;
    rhoFile      << std::endl << t << std::endl;
    J1File       << std::endl << t << std::endl;
    J2File       << std::endl << t << std::endl;
    ddt_phiFile  << std::endl << t << std::endl;

    for (int i = 0; i < Nx*Ny - 1; i++) {
        phiFile     << std::to_string(phi[lastStepIndex][i].real()) << ",";
        A1File      << std::to_string(A1[lastStepIndex][i].real()) << ",";
        A2File      << std::to_string(A2[lastStepIndex][i].real()) << ",";
        ddx_phiFile << std::to_string(ddx_phi[lastStepIndex][i].real()) << ",";
        ddx_A1File  << std::to_string(ddx_A1[lastStepIndex][i].real()) << ",";
        ddx_A2File  << std::to_string(ddx_A2[lastStepIndex][i].real()) << ",";
        ddy_phiFile << std::to_string(ddy_phi[lastStepIndex][i].real()) << ",";
        ddy_A1File  << std::to_string(ddy_A1[lastStepIndex][i].real()) << ",";
        ddy_A2File  << std::to_string(ddy_A2[lastStepIndex][i].real()) << ",";
        rhoFile     << std::to_string(rho[lastStepIndex][i].real()) << ",";
        J1File      << std::to_string(J1[lastStepIndex][i].real()) << ",";
        J2File      << std::to_string(J2[lastStepIndex][i].real()) << ",";
        ddt_phiFile << std::to_string(ddt_phi[1][i].real()) << ",";
    }
    phiFile     << std::to_string(phi[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    A1File      << std::to_string(A1[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    A2File      << std::to_string(A2[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    ddx_phiFile << std::to_string(ddx_phi[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    ddx_A1File  << std::to_string(ddx_A1[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    ddx_A2File  << std::to_string(ddx_A2[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    ddy_phiFile << std::to_string(ddy_phi[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    ddy_A1File  << std::to_string(ddy_A1[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    ddy_A2File  << std::to_string(ddy_A2[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    rhoFile     << std::to_string(rho[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    J1File      << std::to_string(J1[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    J2File      << std::to_string(J2[lastStepIndex][Nx*Ny-1].real()) << std::endl;
    ddt_phiFile << std::to_string(ddt_phi[1][Nx*Ny-1].real()) << std::endl;

    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << std::to_string((*x_elec[lastStepIndex])[p]) << ",";
    }
    electronFile << std::to_string((*x_elec[lastStepIndex])[numElectrons-1]) << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << std::to_string((*y_elec[lastStepIndex])[p]) << ",";
    }
    electronFile << std::to_string((*y_elec[lastStepIndex])[numElectrons-1]) << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << std::to_string((*vx_elec[lastStepIndex])[p]) << ",";
    }
    electronFile << std::to_string((*vx_elec[lastStepIndex])[numElectrons-1]) << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << std::to_string((*vy_elec[lastStepIndex])[p]) << ",";
    }
    electronFile << std::to_string((*vy_elec[lastStepIndex])[numElectrons-1]) << std::endl;

    electronFile << std::endl;
    phiFile      << std::endl;
    A1File       << std::endl;
    A2File       << std::endl;
    ddx_phiFile  << std::endl;
    ddx_A1File   << std::endl;
    ddx_A2File   << std::endl;
    ddy_phiFile  << std::endl;
    ddy_A1File   << std::endl;
    ddy_A2File   << std::endl;
    rhoFile      << std::endl;
    J1File       << std::endl;
    J2File       << std::endl;
    ddt_phiFile  << std::endl;

    electronFile.close();
    phiFile.close();
    A1File.close();
    A2File.close();
    ddx_phiFile.close();
    ddx_A1File.close();
    ddx_A2File.close();
    ddy_phiFile.close();
    ddy_A1File.close();
    ddy_A2File.close();
    rhoFile.close();
    J1File.close();
    J2File.close();
    ddt_phiFile.close();
}

/**
 * Name: updateParticleLocations
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Updates the particle locations using Newton's law
 * Inputs: none (relies on x, y, x_elec, y_elec, vx_elec, vy_elec)
 * Output: none
 * Dependencies: OpenMP
 */
void MOLTEngine::updateParticleLocations() {
    double Lx = x[Nx-1] - x[0] + dx;
    double Ly = y[Ny-1] - y[0] + dy;

    #pragma omp parallel for
    for (int i = 0; i < numElectrons; i++) {
        double vx_star = 2.0*(*vx_elec[lastStepIndex-1])[i] - (*vx_elec[lastStepIndex-2])[i];
        double vy_star = 2.0*(*vy_elec[lastStepIndex-1])[i] - (*vy_elec[lastStepIndex-2])[i];

        (*x_elec[lastStepIndex])[i] = (*x_elec[lastStepIndex-1])[i] + dt*vx_star;
        (*y_elec[lastStepIndex])[i] = (*y_elec[lastStepIndex-1])[i] + dt*vy_star;

        (*x_elec[lastStepIndex])[i] = (*x_elec[lastStepIndex])[i] - Lx*floor(((*x_elec[lastStepIndex])[i] - this->x[0]) / Lx);
        (*y_elec[lastStepIndex])[i] = (*y_elec[lastStepIndex])[i] - Ly*floor(((*y_elec[lastStepIndex])[i] - this->y[0]) / Ly);
    }
}

/**
 * Name: DIRK2_advance_per
 * Author: Stephen White
 * Date Created: 6/26/24 (ish)
 * Date Last Modified: 6/26/24 (Stephen White)
 * Description: Updates the wave and its time derivative. It takes the second order
 *              wave equation and decomposes it into two first order equations and solves
 *              them using a Runge-Kutta method with Qin and Zhang's update scheme.
 * Inputs: u (value), v (ddt_value), u_next (value output), v_next (ddt_value output), src_prev, src_curr
 * Output: none (stores the results in u_next and v_next)
 * Dependencies: none
 */
void MOLTEngine::DIRK2_advance_per(std::complex<double>* u, std::complex<double>* v,
                                   std::complex<double>* u_next, std::complex<double>* v_next,
                                   std::complex<double>* src_prev, std::complex<double>* src_curr) {

        double a11 = 1.0/4.0;
        // double a12 = 0.0;
        double a21 = 1.0/2.0;
        double a22 = 1.0/4.0;

        double b1 = 1.0/2.0;
        double b2 = 1.0/2.0;

        double c1 = 1.0/4.0;
        double c2 = 3.0/4.0;

        double alpha_1 = 1.0/(dt*a11*kappa);
        double alpha_2 = 1.0/(dt*a22*kappa);

        for (int i = 0; i < Nx*Ny; i++) {
            S_1[i] = (1-c1)*src_prev[i] + c1*src_curr[i];
            S_2[i] = (1-c2)*src_prev[i] + c2*src_curr[i];
        }

        compute_d2dx(u, d2dx_u);
        compute_d2dy(u, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u[i] = d2dx_u[i] + d2dy_u[i];
        }

        for (int i = 0; i < Nx*Ny; i++) {
            RHS1[i] = v[i] + dt*a11*kappa*kappa*(laplacian_u[i] + S_1[i]);
        }

        solveHelmholtzEquation(RHS1, u1, alpha_1);

        compute_d2dx(u1, d2dx_u);
        compute_d2dy(u1, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u1[i] = d2dx_u[i] + d2dy_u[i];
        }
        
        for (int i = 0; i < Nx*Ny; i++) {
            v1[i] = kappa*kappa*(laplacian_u[i] + S_1[i] + dt*a11*laplacian_u1[i]);
        }

        for (int i = 0; i < Nx*Ny; i++) {
            RHS2[i] = v[i] + dt*a21*v1[i] + dt*a22*kappa*kappa*(laplacian_u[i] + dt*a21*laplacian_u1[i] + S_2[i]);
        }

        solveHelmholtzEquation(RHS2, u2, alpha_2);

        compute_d2dx(u2, d2dx_u);
        compute_d2dy(u2, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u2[i] = d2dx_u[i] + d2dy_u[i];
        }

        for (int i = 0; i < Nx*Ny; i++) {
            v2[i] = kappa*kappa*(laplacian_u[i] + dt*a21*laplacian_u1[i] + dt*a22*laplacian_u2[i] + S_2[i]);
        }

        for (int i = 0; i < Nx*Ny; i++) {
            u_next[i] = u[i] + dt*(b1*u1[i] + b2*u2[i]);
            v_next[i] = v[i] + dt*(b1*v1[i] + b2*v2[i]);
        }
}

/**
 * Name: DIRK3_advance_per
 * Author: Stephen White
 * Date Created: 6/26/24 (ish)
 * Date Last Modified: 6/26/24 (Stephen White)
 * Description: Updates the wave and its time derivative. It takes the second order
 *              wave equation and decomposes it into two first order equations and solves
 *              them using a Runge-Kutta method with Crouzeix's update scheme.
 * Inputs: u (value), v (ddt_value), u_next (value output), v_next (ddt_value output), src_prev, src_curr
 * Output: none (stores the results in u_next and v_next)
 * Dependencies: none
 */
void MOLTEngine::DIRK3_advance_per(std::complex<double>* u, std::complex<double>* v,
                                   std::complex<double>* u_next, std::complex<double>* v_next,
                                   std::complex<double>* src_prev, std::complex<double>* src_curr) {

        double a11 = 1.0/2.0 + std::sqrt(3.0)/6.0;
        // double a12 = 0.0;
        double a21 = -std::sqrt(3.0)/6.0;
        double a22 = 1.0/2.0 + std::sqrt(3.0)/6.0;

        double b1 = 1.0/2.0;
        double b2 = 1.0/2.0;

        double c1 = 1.0/2.0 + std::sqrt(3.0)/6.0;
        double c2 = 1.0/2.0 - std::sqrt(3.0)/6.0;

        double alpha_1 = 1.0/(dt*a11*kappa);
        double alpha_2 = 1.0/(dt*a22*kappa);

        for (int i = 0; i < Nx*Ny; i++) {
            S_1[i] = (1-c1)*src_prev[i] + c1*src_curr[i];
            S_2[i] = (1-c2)*src_prev[i] + c2*src_curr[i];
        }

        compute_d2dx(u, d2dx_u);
        compute_d2dy(u, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u[i] = d2dx_u[i] + d2dy_u[i];
        }

        for (int i = 0; i < Nx*Ny; i++) {
            RHS1[i] = v[i] + dt*a11*kappa*kappa*(laplacian_u[i] + S_1[i]);
        }

        solveHelmholtzEquation(RHS1, u1, alpha_1);

        compute_d2dx(u1, d2dx_u);
        compute_d2dy(u1, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u1[i] = d2dx_u[i] + d2dy_u[i];
        }
        
        for (int i = 0; i < Nx*Ny; i++) {
            v1[i] = kappa*kappa*(laplacian_u[i] + S_1[i] + dt*a11*laplacian_u1[i]);
        }

        for (int i = 0; i < Nx*Ny; i++) {
            RHS2[i] = v[i] + dt*a21*v1[i] + dt*a22*kappa*kappa*(laplacian_u[i] + dt*a21*laplacian_u1[i] + S_2[i]);
        }

        solveHelmholtzEquation(RHS2, u2, alpha_2);

        compute_d2dx(u2, d2dx_u);
        compute_d2dy(u2, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u2[i] = d2dx_u[i] + d2dy_u[i];
        }

        for (int i = 0; i < Nx*Ny; i++) {
            v2[i] = kappa*kappa*(laplacian_u[i] + dt*a21*laplacian_u1[i] + dt*a22*laplacian_u2[i] + S_2[i]);
        }

        for (int i = 0; i < Nx*Ny; i++) {
            u_next[i] = u[i] + dt*(b1*u1[i] + b2*u2[i]);
            v_next[i] = v[i] + dt*(b1*v1[i] + b2*v2[i]);
        }

}

/**
 * Name: updateWaves
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: This updates the phi, A1, and A2 waves by Rothe's method. Discretizing in time using, for now, the BDF1 method, and bringing the previous
 *              timesteps to the RHS along with the source function results in the modified Helmholtz equation, which we solve using the FFT. It then computes
 *              the corresponding derivatives.
 * Inputs: none (relies on rho, J1, J2, phi, A1, A2)
 * Output: none
 * Dependencies: solveHelmholtzEquation, compute_ddx, compute_ddy
 */
void MOLTEngine::updateWaves() {
    double alpha = beta/(kappa*dt);
    double alpha2 = alpha*alpha;

    if (this->method == MOLTEngine::MOLT_BDF1 || this->method == MOLTEngine::MOLT_BDF1_HYBRID_FFT || this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6) {
        std::complex<double>* phi_RHS = new std::complex<double>[Nx*Ny];
        std::complex<double>*  A1_RHS = new std::complex<double>[Nx*Ny];
        std::complex<double>*  A2_RHS = new std::complex<double>[Nx*Ny];

        for (int i = 0; i < Nx*Ny; i++) {
            phi_RHS[i] = 1.0/sigma_1 * rho[lastStepIndex][i];
            A1_RHS[i] = sigma_2 * J1[lastStepIndex][i];
            A2_RHS[i] = sigma_2 * J2[lastStepIndex][i];
        }

        MOLT_BDF1_combined_per_advance(phi, phi_RHS, phi[lastStepIndex], ddx_phi[lastStepIndex], ddy_phi[lastStepIndex]);
        MOLT_BDF1_combined_per_advance(A1,  A1_RHS,  A1[lastStepIndex],  ddx_A1[lastStepIndex],  ddy_A1[lastStepIndex] );
        MOLT_BDF1_combined_per_advance(A2,  A2_RHS,  A2[lastStepIndex],  ddx_A2[lastStepIndex],  ddy_A2[lastStepIndex] );

        delete[] phi_RHS;
        delete[] A1_RHS;
        delete[] A2_RHS;
    } else {
        if (this->method == MOLTEngine::DIRK2 || this->method == MOLTEngine::DIRK3) {

            for (int i = 0; i < Nx*Ny; i++) {
                phi_src[i] = 1.0/sigma_1 * rho[lastStepIndex][i];
                A1_src[i] = sigma_2 * J1[lastStepIndex][i];
                A2_src[i] = sigma_2 * J2[lastStepIndex][i];

                phi_src_prev[i] = 1.0/sigma_1 * rho[lastStepIndex-1][i];
                A1_src_prev[i] = sigma_2 * J1[lastStepIndex-1][i];
                A2_src_prev[i] = sigma_2 * J2[lastStepIndex-1][i];
            }

            for (int i = 0; i < Nx*Ny; i++) {
                ddt_phi_curr[i] = (phi[lastStepIndex-1][i] - phi[lastStepIndex-2][i]) / dt;
                ddt_A1_curr[i]  = ( A1[lastStepIndex-1][i] - A1[lastStepIndex-2][i] ) / dt;
                ddt_A2_curr[i]  = ( A2[lastStepIndex-1][i] - A2[lastStepIndex-2][i] ) / dt;
            }

            if (this->method == MOLTEngine::DIRK2) {
                DIRK2_advance_per(phi[lastStepIndex-1], ddt_phi_curr, phi[lastStepIndex], ddt_phi[1], phi_src_prev, phi_src);
                DIRK2_advance_per(A1[lastStepIndex-1], ddt_A1_curr, A1[lastStepIndex], ddt_A1[1], A1_src_prev, A1_src);
                DIRK2_advance_per(A2[lastStepIndex-1], ddt_A2_curr, A2[lastStepIndex], ddt_A2[1], A2_src_prev, A2_src);
            } else if (this->method == MOLTEngine::DIRK3) {
                DIRK3_advance_per(phi[lastStepIndex-1], ddt_phi_curr, phi[lastStepIndex], ddt_phi[1], phi_src_prev, phi_src);
                DIRK3_advance_per(A1[lastStepIndex-1], ddt_A1_curr, A1[lastStepIndex], ddt_A1[1], A1_src_prev, A1_src);
                DIRK3_advance_per(A2[lastStepIndex-1], ddt_A2_curr, A2[lastStepIndex], ddt_A2[1], A2_src_prev, A2_src);
            }

        } else {
            if (this->method == MOLTEngine::BDF1) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi_src[i] = 2.0*phi[lastStepIndex-1][i] - phi[lastStepIndex-2][i] + 1.0/alpha2 * 1.0/sigma_1 * rho[lastStepIndex][i];
                    A1_src[i] = 2.0*A1[lastStepIndex-1][i] - A1[lastStepIndex-2][i] + 1.0/alpha2 * sigma_2 * J1[lastStepIndex][i];
                    A2_src[i] = 2.0*A2[lastStepIndex-1][i] - A2[lastStepIndex-2][i] + 1.0/alpha2 * sigma_2 * J2[lastStepIndex][i];
                }
            } else if (this->method == MOLTEngine::BDF2) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi_src[i] = 8.0/3.0*phi[lastStepIndex-1][i] - 22.0/9.0*phi[lastStepIndex-2][i] + 8.0/9.0*phi[lastStepIndex-3][i] - 1.0/9.0*phi[lastStepIndex-4][i] + 1.0/alpha2 * 1.0/sigma_1 * rho[lastStepIndex][i];
                    A1_src[i] = 8.0/3.0*A1[lastStepIndex-1][i] - 22.0/9.0*A1[lastStepIndex-2][i] + 8.0/9.0*A1[lastStepIndex-3][i] - 1.0/9.0*A1[lastStepIndex-4][i] + 1.0/alpha2 * sigma_2 * J1[lastStepIndex][i];
                    A2_src[i] = 8.0/3.0*A2[lastStepIndex-1][i] - 22.0/9.0*A2[lastStepIndex-2][i] + 8.0/9.0*A2[lastStepIndex-3][i] - 1.0/9.0*A2[lastStepIndex-4][i] + 1.0/alpha2 * sigma_2 * J2[lastStepIndex][i];
                }
            } else if (this->method == MOLTEngine::CDF1) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi_src[i] = 1.0/alpha2*(rho[lastStepIndex][i] + rho[lastStepIndex-2][i])/sigma_1 + 2.0*phi[lastStepIndex-1][i];
                    A1_src[i] = sigma_2/alpha2*(J1[lastStepIndex][i] + J1[lastStepIndex-2][i]) + 2.0*A1[lastStepIndex-1][i];
                    A2_src[i] = sigma_2/alpha2*(J2[lastStepIndex][i] + J2[lastStepIndex-2][i]) + 2.0*A2[lastStepIndex-1][i];
                }
            }

            solveHelmholtzEquation(phi_src, phi[lastStepIndex], alpha);
            solveHelmholtzEquation(A1_src,  A1[lastStepIndex], alpha);
            solveHelmholtzEquation(A2_src,  A2[lastStepIndex], alpha);

            if (this->method == MOLTEngine::CDF1) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi[lastStepIndex][i] -= phi[lastStepIndex-2][i];
                    A1[lastStepIndex][i]  -= A1[lastStepIndex-2][i];
                    A2[lastStepIndex][i]  -= A2[lastStepIndex-2][i];
                }
            }
        }

        compute_ddx_FFT(phi[lastStepIndex], ddx_phi[lastStepIndex]);
        compute_ddy_FFT(phi[lastStepIndex], ddy_phi[lastStepIndex]);
        compute_ddx_FFT(A1[lastStepIndex],  ddx_A1[lastStepIndex]);
        compute_ddy_FFT(A1[lastStepIndex],  ddy_A1[lastStepIndex]);
        compute_ddx_FFT(A2[lastStepIndex],  ddx_A2[lastStepIndex]);
        compute_ddy_FFT(A2[lastStepIndex],  ddy_A2[lastStepIndex]);
    }
}

/**
 * Name: updateParticleVelocities
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Oh boy, this one. This uses the Improved Asymmetric Euler Method to update the particle velocities. Simple, right?
 *              Except it requires the gathering of eight field values for each particle, which is easily the most computationally
 *              intensive process in this simulation. Now, this is embarassingly parallel, which is a small grace, however, if we 
 *              wish to use OpenMP we can't rely on gatherField or gatherFields, it has to be in line, which makes things redundant.
 * Inputs: none
 * Output: none
 * Dependencies: OpenMP
 */
void MOLTEngine::updateParticleVelocities() {

    #pragma omp parallel for
    for (int i = 0; i < numElectrons; i++) {
        double ddx_phi_p = 0;
        double ddy_phi_p = 0;
        double A1_p = 0;
        double ddx_A1_p = 0;
        double ddy_A1_p = 0;
        double A2_p = 0;
        double ddx_A2_p = 0;
        double ddy_A2_p = 0;
        const double p_x = (*x_elec[lastStepIndex])[i];
        const double p_y = (*y_elec[lastStepIndex])[i];
        // ------------------------------
        // Gather Fields
        // We convert from cartesian to logical space
        const double x0 = this->x[0];
        const double y0 = this->y[0];
        const int lc_x = floor((p_x - x0)/dx);
        const int lc_y = floor((p_y - y0)/dy);

        const int lc_x_p1 = (lc_x+1) % Nx;
        const int lc_y_p1 = (lc_y+1) % Ny;

        const int ld = lc_x * Ny + lc_y;          // (left, down)  lc_x,   lc_y
        const int lu = lc_x * Ny + lc_y_p1;       // (left, up)    lc_x,   lc_y+1
        const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down)  lc_x+1, lc_y
        const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)    lc_x+1, lc_y+1

        const double xNode = this->x[lc_x];
        const double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        const double fx = (p_x - xNode)/dx;
        const double fy = (p_y - yNode)/dy;

        const double w_ld = (1-fx)*(1-fy);
        const double w_lu = (1-fx)*(fy);
        const double w_rd = (fx)*(1-fy);
        const double w_ru = (fx)*(fy);

        if (this->method == MOLTEngine::CDF1) {
            ddx_phi_p += w_ld*( ddx_phi[lastStepIndex][ld].real() + ddx_phi[lastStepIndex-1][ld].real() ) / 2;
            ddx_phi_p += w_lu*( ddx_phi[lastStepIndex][lu].real() + ddx_phi[lastStepIndex-1][lu].real() ) / 2;
            ddx_phi_p += w_rd*( ddx_phi[lastStepIndex][rd].real() + ddx_phi[lastStepIndex-1][rd].real() ) / 2;
            ddx_phi_p += w_ru*( ddx_phi[lastStepIndex][ru].real() + ddx_phi[lastStepIndex-1][ru].real() ) / 2;

            ddy_phi_p += w_ld*( ddy_phi[lastStepIndex][ld].real() + ddy_phi[lastStepIndex-1][ld].real() ) / 2;
            ddy_phi_p += w_lu*( ddy_phi[lastStepIndex][lu].real() + ddy_phi[lastStepIndex-1][lu].real() ) / 2;
            ddy_phi_p += w_rd*( ddy_phi[lastStepIndex][rd].real() + ddy_phi[lastStepIndex-1][rd].real() ) / 2;
            ddy_phi_p += w_ru*( ddy_phi[lastStepIndex][ru].real() + ddy_phi[lastStepIndex-1][ru].real() ) / 2;
        } else {
            ddx_phi_p += (1-fx)*(1-fy)*ddx_phi[lastStepIndex][ld].real();
            ddx_phi_p += (1-fx)*(fy)*ddx_phi[lastStepIndex][lu].real();
            ddx_phi_p += (fx)*(1-fy)*ddx_phi[lastStepIndex][rd].real();
            ddx_phi_p += (fx)*(fy)*ddx_phi[lastStepIndex][ru].real();

            ddy_phi_p += (1-fx)*(1-fy)*ddy_phi[lastStepIndex][ld].real();
            ddy_phi_p += (1-fx)*(fy)*ddy_phi[lastStepIndex][lu].real();
            ddy_phi_p += (fx)*(1-fy)*ddy_phi[lastStepIndex][rd].real();
            ddy_phi_p += (fx)*(fy)*ddy_phi[lastStepIndex][ru].real();
        }

        A1_p += w_ld*A1[lastStepIndex][ld].real();
        A1_p += w_lu*A1[lastStepIndex][lu].real();
        A1_p += w_rd*A1[lastStepIndex][rd].real();
        A1_p += w_ru*A1[lastStepIndex][ru].real();

        ddx_A1_p += w_ld*ddx_A1[lastStepIndex][ld].real();
        ddx_A1_p += w_lu*ddx_A1[lastStepIndex][lu].real();
        ddx_A1_p += w_rd*ddx_A1[lastStepIndex][rd].real();
        ddx_A1_p += w_ru*ddx_A1[lastStepIndex][ru].real();

        ddy_A1_p += w_ld*ddy_A1[lastStepIndex][ld].real();
        ddy_A1_p += w_lu*ddy_A1[lastStepIndex][lu].real();
        ddy_A1_p += w_rd*ddy_A1[lastStepIndex][rd].real();
        ddy_A1_p += w_ru*ddy_A1[lastStepIndex][ru].real();

        A2_p += w_ld*A2[lastStepIndex][ld].real();
        A2_p += w_lu*A2[lastStepIndex][lu].real();
        A2_p += w_rd*A2[lastStepIndex][rd].real();
        A2_p += w_ru*A2[lastStepIndex][ru].real();

        ddx_A2_p += w_ld*ddx_A2[lastStepIndex][ld].real();
        ddx_A2_p += w_lu*ddx_A2[lastStepIndex][lu].real();
        ddx_A2_p += w_rd*ddx_A2[lastStepIndex][rd].real();
        ddx_A2_p += w_ru*ddx_A2[lastStepIndex][ru].real();

        ddy_A2_p += w_ld*ddy_A2[lastStepIndex][ld].real();
        ddy_A2_p += w_lu*ddy_A2[lastStepIndex][lu].real();
        ddy_A2_p += w_rd*ddy_A2[lastStepIndex][rd].real();
        ddy_A2_p += w_ru*ddy_A2[lastStepIndex][ru].real();

        double vx_star = 2.0*(*vx_elec[lastStepIndex-1])[i] - (*vx_elec[lastStepIndex-2])[i];
        double vy_star = 2.0*(*vy_elec[lastStepIndex-1])[i] - (*vy_elec[lastStepIndex-2])[i];

        double rhs1 = -q_ele*ddx_phi_p + q_ele*( ddx_A1_p*vx_star + ddx_A2_p*vy_star );
        double rhs2 = -q_ele*ddy_phi_p + q_ele*( ddy_A1_p*vx_star + ddy_A2_p*vy_star );

        // Compute the new momentum
        (*Px_elec[lastStepIndex])[i] = (*Px_elec[lastStepIndex-1])[i] + dt*rhs1;
        (*Py_elec[lastStepIndex])[i] = (*Py_elec[lastStepIndex-1])[i] + dt*rhs2;

        double denom = std::sqrt(std::pow((*Px_elec[lastStepIndex])[i] - q_ele*A1_p, 2) +
                                 std::pow((*Py_elec[lastStepIndex])[i] - q_ele*A2_p, 2) +
                                 std::pow(m_ele*kappa, 2));

        // Compute the new velocity using the updated momentum
        (*vx_elec[lastStepIndex])[i] = (kappa*((*Px_elec[lastStepIndex])[i] - q_ele*A1_p)) / denom;
        (*vy_elec[lastStepIndex])[i] = (kappa*((*Py_elec[lastStepIndex])[i] - q_ele*A2_p)) / denom;
    }
}

/**
 * Name: solveHelmholtzEquation
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Solves the modified Helmholtz equation (I - (1/alpha^2)Delta) u = RHS using the FFT.
 * Inputs: RHS, LHS, alpha
 * Output: technically none, but LHS is where the result is stored
 * Dependencies: to_std_complex, fftw
 */
void MOLTEngine::solveHelmholtzEquation(std::complex<double>* RHS,
                                        std::complex<double>* LHS, double alpha) {

    // Execute the forward FFT
    fftw_execute_dft(forward_plan, reinterpret_cast<fftw_complex*>(RHS), reinterpret_cast<fftw_complex*>(forwardOut));

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(forwardOut)[index]);
            double k_val =  kx_deriv_2[i]*kx_deriv_2[i] + ky_deriv_2[j]*ky_deriv_2[j];
            freq_component /= (1 + 1/(alpha*alpha) * k_val); // Invert the helmholtz operator (I - (d^2/dx^2 + d^2/dy^2)) ==Fourier==> (I + (kx^2 + ky^2)))
            reinterpret_cast<fftw_complex*>(backwardIn)[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(backwardIn)[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx*Ny; i++) {
        LHS[i] = backwardOut[i] / double(Nx * Ny);
    }
}

/**
 * Name: shuffleSteps
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: We use time history throughout this simulation, which means we need a copy of Nh previous timesteps.
 *              At the end of each iteration we shuffle the results down one timestep, making room for the next.
 * Inputs: none (relies on global values x_elec, y_elec, vx_elec, vy_elec, Px_elec, Py_elec, phi, ddx_phi, ddy_phi, A1, ddx_A1, ddy_A1, A2, ddx_A2, ddy_A2)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::shuffleSteps() {
    std::vector<double>* x_elec_dlt_ptr = x_elec[0];
    std::vector<double>* y_elec_dlt_ptr = y_elec[0];
    std::vector<double>* vx_elec_dlt_ptr = vx_elec[0];
    std::vector<double>* vy_elec_dlt_ptr = vy_elec[0];
    std::vector<double>* Px_elec_dlt_ptr = Px_elec[0];
    std::vector<double>* Py_elec_dlt_ptr = Py_elec[0];

    std::complex<double>* phi_dlt_ptr = phi[0];
    std::complex<double>* ddx_phi_dlt_ptr = ddx_phi[0];
    std::complex<double>* ddy_phi_dlt_ptr = ddy_phi[0];
    std::complex<double>* A1_dlt_ptr = A1[0];
    std::complex<double>* ddx_A1_dlt_ptr = ddx_A1[0];
    std::complex<double>* ddy_A1_dlt_ptr = ddy_A1[0];
    std::complex<double>* A2_dlt_ptr = A2[0];
    std::complex<double>* ddx_A2_dlt_ptr = ddx_A2[0];
    std::complex<double>* ddy_A2_dlt_ptr = ddy_A2[0];

    std::complex<double>* rho_dlt_ptr = rho[0];
    std::complex<double>* J1_dlt_ptr = J1[0];
    std::complex<double>* J2_dlt_ptr = J2[0];

    for (int h = 0; h < Nh-1; h++) {

        x_elec[h] = x_elec[h+1];
        y_elec[h] = y_elec[h+1];
        vx_elec[h] = vx_elec[h+1];
        vy_elec[h] = vy_elec[h+1];
        Px_elec[h] = Px_elec[h+1];
        Py_elec[h] = Py_elec[h+1];
        
        phi[h] = phi[h+1];
        ddx_phi[h] = ddx_phi[h+1];
        ddy_phi[h] = ddy_phi[h+1];
        A1[h] = A1[h+1];
        ddx_A1[h] = ddx_A1[h+1];
        ddy_A1[h] = ddy_A1[h+1];
        A2[h] = A2[h+1];
        ddx_A2[h] = ddx_A2[h+1];
        ddy_A2[h] = ddy_A2[h+1];

        rho[h] = rho[h+1];
        J1[h] = J1[h+1];
        J2[h] = J2[h+1];
    }

    x_elec[lastStepIndex] = x_elec_dlt_ptr;
    y_elec[lastStepIndex] = y_elec_dlt_ptr;
    vx_elec[lastStepIndex] = vx_elec_dlt_ptr;
    vy_elec[lastStepIndex] = vy_elec_dlt_ptr;
    Px_elec[lastStepIndex] = Px_elec_dlt_ptr;
    Py_elec[lastStepIndex] = Py_elec_dlt_ptr;

    phi[lastStepIndex] = phi_dlt_ptr;
    ddx_phi[lastStepIndex] = ddx_phi_dlt_ptr;
    ddy_phi[lastStepIndex] = ddy_phi_dlt_ptr;
    A1[lastStepIndex] = A1_dlt_ptr;
    ddx_A1[lastStepIndex] = ddx_A1_dlt_ptr;
    ddy_A1[lastStepIndex] = ddy_A1_dlt_ptr;
    A2[lastStepIndex] = A2_dlt_ptr;
    ddx_A2[lastStepIndex] = ddx_A2_dlt_ptr;
    ddy_A2[lastStepIndex] = ddy_A2_dlt_ptr;

    rho[lastStepIndex] = rho_dlt_ptr;
    J1[lastStepIndex] = J1_dlt_ptr;
    J2[lastStepIndex] = J2_dlt_ptr;

    std::complex<double>* ddt_phi_dlt_ptr = ddt_phi[0];
    std::complex<double>* ddt_A1_dlt_ptr = ddt_A1[0];
    std::complex<double>* ddt_A2_dlt_ptr = ddt_A2[0];

    ddt_phi[0] = ddt_phi[1];
    ddt_A1[0] = ddt_A1[1];
    ddt_A2[0] = ddt_A2[1];

    ddt_phi[1] = ddt_phi_dlt_ptr;
    ddt_A1[1] = ddt_A1_dlt_ptr;
    ddt_A2[1] = ddt_A2_dlt_ptr;
}

/**
 * Name: gatherFields
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Eliminates redundancies in the gatherField method, instead of multiple calls to gatherField
 *              we pass in a vector of fields and compute the fractional weight for each particle, using this
 *              for each field.
 * Inputs: p_x, p_y, fields, fields_out
 * Output: technically none, but fields_out is where the results are stored.
 * Dependencies: none
 */
void MOLTEngine::gatherFields(double p_x, double p_y, std::vector<std::vector<std::complex<double>>>& fields, std::vector<double>& fields_out) {
    // We convert from cartesian to logical space
    const double x0 = this->x[0];
    const double y0 = this->y[0];
    const int lc_x = floor((p_x - x0)/dx);
    const int lc_y = floor((p_y - y0)/dy);

    const int lc_x_p1 = (lc_x+1) % Nx;
    const int lc_y_p1 = (lc_y+1) % Ny;

    const int ld = lc_x * Ny + lc_y;          // (left,down)  lc_x,   lc_y
    const int lu = lc_x * Ny + lc_y_p1;       // (left, up)   lc_x,   lc_y+1
    const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down) lc_x+1, lc_y
    const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)   lc_x+1, lc_y+1

    const double xNode = this->x[lc_x];
    const double yNode = this->y[lc_y];

    // We compute the fractional distance of a particle from
    // the nearest node.
    // eg x=[0,.1,.2,.3], particleX = [.225]
    // The particle's fractional is 1/4
    const double fx = (p_x - xNode)/dx;
    const double fy = (p_y - yNode)/dy;

    const int N = fields.size();

    double field_00, field_01, field_10, field_11;

    for (int i = 0; i < N; i++) {
        // Now we acquire the field values at the surrounding nodes
        field_00 = fields[i][ld].real();
        field_01 = fields[i][lu].real();
        field_10 = fields[i][rd].real();
        field_11 = fields[i][ru].real();

        // Returning the combined total of all the fields in proportion
        // with the fractional distance
        fields_out[i] = (1-fx)*(1-fy)*field_00 + (1-fx)*(fy)*field_01 + (fx)*(1-fy)*field_10 + (fx)*(fy)*field_11;
    }
}

/**
 * Name: scatterFields
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Scatters the particles across the current meshes (J1, J2), then computes rho by taking the divergence of J
 *              and using the continuity equation (rho_t + div(J) = 0).
 * Inputs: none (relies on global values rho, J1, J2, and particle location, velocities, and charge.)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::scatterFields() {
    for (int i = 0; i < Nx*Ny; i++) {
        J1[lastStepIndex][i] = 0.0;
        J2[lastStepIndex][i] = 0.0;
    }

    for (int i = 0; i < numElectrons; i++) {
        double vx_star = 2.0*(*vx_elec[lastStepIndex-1])[i] - (*vx_elec[lastStepIndex-2])[i];
        double vy_star = 2.0*(*vy_elec[lastStepIndex-1])[i] - (*vy_elec[lastStepIndex-2])[i];

        double x_value = q_ele*vx_star*w_ele;
        double y_value = q_ele*vy_star*w_ele;

        double x_p = this->method == MOLTEngine::CDF1 ? ( (*x_elec[lastStepIndex])[i] + (*x_elec[lastStepIndex-1])[i] ) / 2 : (*x_elec[lastStepIndex])[i];
        double y_p = this->method == MOLTEngine::CDF1 ? ( (*y_elec[lastStepIndex])[i] + (*y_elec[lastStepIndex-1])[i] ) / 2 : (*y_elec[lastStepIndex])[i];

        // scatterField(x_p, y_p, x_value, J1[lastStepIndex]);
        // scatterField(x_p, y_p, y_value, J2[lastStepIndex]);

        // We convert from cartesian to logical space
        int lc_x = floor((x_p - x[0])/dx);
        int lc_y = floor((y_p - y[0])/dy);

        const int lc_x_p1 = (lc_x+1) % Nx;
        const int lc_y_p1 = (lc_y+1) % Ny;

        const int ld = lc_x * Ny + lc_y;          // (left,down)  lc_x,   lc_y
        const int lu = lc_x * Ny + lc_y_p1;       // (left, up)   lc_x,   lc_y+1
        const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down) lc_x+1, lc_y
        const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)   lc_x+1, lc_y+1

        double xNode = this->x[lc_x];
        double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        double fx = (x_p - xNode)/dx;
        double fy = (y_p - yNode)/dy;

        // Now we acquire the particle value and add it to the corresponding field
        J1[lastStepIndex][ld] += (1-fx)*(1-fy)*x_value;
        J1[lastStepIndex][lu] += (1-fx)*(fy)*x_value;
        J1[lastStepIndex][rd] += (fx)*(1-fy)*x_value;
        J1[lastStepIndex][ru] += (fx)*(fy)*x_value;

        J2[lastStepIndex][ld] += (1-fx)*(1-fy)*y_value;
        J2[lastStepIndex][lu] += (1-fx)*(fy)*y_value;
        J2[lastStepIndex][rd] += (fx)*(1-fy)*y_value;
        J2[lastStepIndex][ru] += (fx)*(fy)*y_value;
    }
    double volume = 1.0/(dx*dy);
    for (int i = 0; i < Nx*Ny; i++) {
        J1[lastStepIndex][i] *= volume;
        J2[lastStepIndex][i] *= volume;
    }

    // fftw_execute_dft(forward_plan, reinterpret_cast<fftw_complex*>(J1[lastStepIndex]), reinterpret_cast<fftw_complex*>(forwardOut));
    // fftw_execute_dft(inverse_plan, reinterpret_cast<fftw_complex*>(forwardOut), reinterpret_cast<fftw_complex*>(J1[lastStepIndex]));

    // fftw_execute_dft(forward_plan, reinterpret_cast<fftw_complex*>(J2[lastStepIndex]), reinterpret_cast<fftw_complex*>(forwardOut));
    // fftw_execute_dft(inverse_plan, reinterpret_cast<fftw_complex*>(forwardOut), reinterpret_cast<fftw_complex*>(J2[lastStepIndex]));

    // Compute div J
    if (this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6) {
        compute_ddx_FD6(J1[lastStepIndex], ddx_J1);
        compute_ddy_FD6(J2[lastStepIndex], ddy_J2);
    } else {
        compute_ddx_FFT(J1[lastStepIndex], ddx_J1);
        compute_ddy_FFT(J2[lastStepIndex], ddy_J2);
    }

    // double Gamma = 0;

    // for (int i = 0; i < Nx*Ny; i++) {
    //     Gamma += ddx_J1[i].real() + ddy_J2[i].real();
    // }
    // Gamma *= -1.0/(Nx*Ny);

    // int idx;

    // for (int i = 0; i < Nx; i++) {
    //     for (int j = 0; j < Ny; j++) {
    //         idx = i*Ny + j;
    //         F1[idx] = .5*Gamma*x[i];
    //         F2[idx] = .5*Gamma*y[j];
    //     }
    // }

    // for (int i = 0; i < Nx*Ny; i++) {
    //     J1[lastStepIndex][i] += F1[i];
    //     J2[lastStepIndex][i] += F2[i];
    // }

    // // Compute div J from the Lagrange Multiplied Current
    // compute_ddx(J1[lastStepIndex], ddx_J1);
    // compute_ddy(J2[lastStepIndex], ddy_J2);

    if (this->rhoUpdate == MOLTEngine::CONSERVING) {
        if (this->method == MOLTEngine::BDF1 ||
            this->method == MOLTEngine::MOLT_BDF1 ||
            this->method == MOLTEngine::MOLT_BDF1_HYBRID_FFT ||
            this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6 ||
            this->method == MOLTEngine::CDF1) {
            for (int i = 0; i < Nx*Ny; i++) {
                rho[lastStepIndex][i] = rho[lastStepIndex-1][i] - dt*(ddx_J1[i] + ddy_J2[i]);
            }
        } else if (this->method == MOLTEngine::BDF2) {
            for (int i = 0; i < Nx*Ny; i++) {
                rho[lastStepIndex][i] = 4.0/3.0*rho[lastStepIndex-1][i] - 1.0/3.0*rho[lastStepIndex-2][i] - ((2.0/3.0)*dt)*(ddx_J1[i] + ddy_J2[i]);
            }
        } else if (this->method == MOLTEngine::DIRK2 || this->method == MOLTEngine::DIRK3) {

            double b1;
            double b2;

            double c1;
            double c2;

            if (this->method == MOLTEngine::DIRK2) {
                // Qin and Zhang's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/4.0;
                c2 = 3.0/4.0;
            } else if (this->method == MOLTEngine::DIRK3) {
                // Crouzeix's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/2.0 + std::sqrt(3.0)/6.0;
                c2 = 1.0/2.0 - std::sqrt(3.0)/6.0;
            }

            compute_ddx_FFT(J1[lastStepIndex-1], ddx_J1_prev);
            compute_ddy_FFT(J2[lastStepIndex-1], ddy_J2_prev);

            for (int i = 0; i < Nx*Ny; i++) {

                std::complex<double> rhs_prev = ddx_J1_prev[i] + ddy_J2_prev[i];
                std::complex<double> rhs_curr = ddx_J1[i] + ddy_J2[i];

                std::complex<double> RHS_1 = (1-c1)*rhs_prev + c1*rhs_curr;
                std::complex<double> RHS_2 = (1-c2)*rhs_prev + c2*rhs_curr;

                rho[lastStepIndex][i] = rho[lastStepIndex-1][i] - dt*(b1*RHS_1 + b2*RHS_2);
            }
        }
        else {
            throw -1;
        }
    // Normalizing rho to enforce continuity
    // double alpha = rhoTotalPrev / rhoTotal;
    // for (int i = 0; i < Nx*Ny; i++) {
    //     rho[lastStepIndex][i] *= alpha;
    // }
    } else if (this->rhoUpdate == MOLTEngine::NAIVE) {
        for (int i = 0; i < Nx*Ny; i++) {
            rho[lastStepIndex][i] = 0.0;
        }
        double S [Nx*Ny] = {0};
        #pragma omp parallel for reduction(+:S[:Nx*Ny])
        for (int i = 0; i < numElectrons; i++) {

            double charge_value = q_ele*w_ele;

            double x_p = this->method == MOLTEngine::CDF1 ? ( (*x_elec[lastStepIndex])[i] + (*x_elec[lastStepIndex-1])[i] ) / 2 : (*x_elec[lastStepIndex])[i];
            double y_p = this->method == MOLTEngine::CDF1 ? ( (*y_elec[lastStepIndex])[i] + (*y_elec[lastStepIndex-1])[i] ) / 2 : (*y_elec[lastStepIndex])[i];

            // scatterField(x_p, y_p, x_value, J1[lastStepIndex]);
            // scatterField(x_p, y_p, y_value, J2[lastStepIndex]);

            // We convert from cartesian to logical space
            int lc_x = floor((x_p - x[0])/dx);
            int lc_y = floor((y_p - y[0])/dy);

            const int lc_x_p1 = (lc_x+1) % Nx;
            const int lc_y_p1 = (lc_y+1) % Ny;

            const int ld = lc_x    * Ny + lc_y;       // (left,down)  lc_x,   lc_y
            const int lu = lc_x    * Ny + lc_y_p1;    // (left, up)   lc_x,   lc_y+1
            const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down) lc_x+1, lc_y
            const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)   lc_x+1, lc_y+1

            double xNode = this->x[lc_x];
            double yNode = this->y[lc_y];

            // We compute the fractional distance of a particle from
            // the nearest node.
            // eg x=[0,.1,.2,.3], particleX = [.225]
            // The particle's fractional is 1/4
            double fx = (x_p - xNode)/dx;
            double fy = (y_p - yNode)/dy;

            // Now we acquire the particle value and add it to the corresponding field
            // rho[lastStepIndex][ld] += (1-fx)*(1-fy)*charge_value;
            // rho[lastStepIndex][lu] += (1-fx)*(  fy)*charge_value;
            // rho[lastStepIndex][rd] += (  fx)*(1-fy)*charge_value;
            // rho[lastStepIndex][ru] += (  fx)*(  fy)*charge_value;
            S[ld] += (1-fx)*(1-fy)*charge_value;
            S[lu] += (1-fx)*(  fy)*charge_value;
            S[rd] += (  fx)*(1-fy)*charge_value;
            S[ru] += (  fx)*(  fy)*charge_value;
        }

        double volume = 1.0/(dx*dy);
        for (int i = 0; i < Nx*Ny; i++) {
            rho_eles[i] = S[i]*volume;
            rho[lastStepIndex][i] = rho_eles[i] + rho_ions[i];
        }
    }
    // double rhoTotalPrev = this->rhoTotal;
    this->rhoTotal = 0.0;
    for (int i = 0; i < Nx*Ny; i++) {
        this->rhoTotal += rho[lastStepIndex][i].real();
    }
}

/**
 * Name: gatherField
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Takes a particle location in cartesian space, converts it to logical space, and using bilinear interpolation gathers
 *              the value of the field in question, returning this value.
 * Inputs: p_x, p_y, field
 * Output: value
 * Dependencies: none
 */
double MOLTEngine::gatherField(double p_x, double p_y, std::complex<double>* field) {
    // We convert from cartesian to logical space
    double x0 = this->x[0];
    double y0 = this->y[0];
    int lc_x = floor((p_x - x0)/dx);
    int lc_y = floor((p_y - y0)/dy);

    const int lc_x_p1 = (lc_x+1) % Nx;
    const int lc_y_p1 = (lc_y+1) % Ny;

    const int ld = lc_x * Ny + lc_y;          // (left,down)  lc_x,   lc_y
    const int lu = lc_x * Ny + lc_y_p1;       // (left, up)   lc_x,   lc_y+1
    const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down) lc_x+1, lc_y
    const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)   lc_x+1, lc_y+1

    double xNode = this->x[lc_x];
    double yNode = this->y[lc_y];

    // We compute the fractional distance of a particle from
    // the nearest node.
    // eg x=[0,.1,.2,.3], particleX = [.225]
    // The particle's fractional is 1/4
    double fx = (p_x - xNode)/dx;
    double fy = (p_y - yNode)/dy;

    // Now we acquire the field values at the surrounding nodes
    const double field_00 = field[ld].real();
    const double field_01 = field[lu].real();
    const double field_10 = field[rd].real();
    const double field_11 = field[ru].real();

    // Returning the combined total of all the fields in proportion
    // with the fractional distance
    return (1-fx)*(1-fy)*field_00 + (1-fx)*(fy)*field_01 + (fx)*(1-fy)*field_10 + (fx)*(fy)*field_11;
}

/**
 * Name: scatterField
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Takes a particle location in cartesian space, converts it to logical space, and using bilinear interpolation partitions
 *              the value of the particle and adds it to the field.
 * Inputs: p_x, p_y, value, field
 * Output: technically none, but field is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: none
 */
void MOLTEngine::scatterField(double p_x, double p_y, double value, std::complex<double>* field) {

    // We convert from cartesian to logical space
    double x0 = this->x[0];
    double y0 = this->y[0];

    int lc_x = floor((p_x - x0)/dx);
    int lc_y = floor((p_y - y0)/dy);

    const int lc_x_p1 = (lc_x+1) % Nx;
    const int lc_y_p1 = (lc_y+1) % Ny;

    const int ld = lc_x    * Ny + lc_y;       // (left,down)  lc_x,   lc_y
    const int lu = lc_x    * Ny + lc_y_p1;    // (left, up)   lc_x,   lc_y+1
    const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down) lc_x+1, lc_y
    const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)   lc_x+1, lc_y+1

    if (lc_x >= Nx || lc_x < 0 || lc_y >= Ny || lc_y < 0) {
        std::cerr << "OUT OF BOUNDS FOR DOMAIN: " << "[" << x[0] << ", " << x[Nx-1] << "] X [" << y[0] << ", " << y[Ny-1] << "]" << std::endl;
        std::cerr << "x_p" << ", " << "y_p" << "    " << "lc_x" << ", " << "lc_y" << std::endl;
        std::cerr << p_x << ", " << p_y << "    " << lc_x << ", " << lc_y << std::endl;
        throw -1;
    }    

    double xNode = this->x[lc_x];
    double yNode = this->y[lc_y];

    // We compute the fractional distance of a particle from
    // the nearest node.
    // eg x=[0,.1,.2,.3], particleX = [.225]
    // The particle's fractional is 1/4
    double fx = (p_x - xNode)/dx;
    double fy = (p_y - yNode)/dy;

    // Now we acquire the particle value and add it to the corresponding field
    field[ld] += (1-fx)*(1-fy)*value;
    field[lu] += (1-fx)*(  fy)*value;
    field[rd] += (  fx)*(1-fy)*value;
    field[ru] += (  fx)*(  fy)*value;
}

void MOLTEngine::MOLT_BDF1_combined_per_advance(std::vector<std::complex<double>*> u, std::complex<double>* RHS,
                                                std::complex<double>* u_out, std::complex<double>* dudx_out, std::complex<double>* dudy_out) {

    MOLT_BDF1_advance_per(u, RHS, u_out);

    if (this->method == MOLTEngine::MOLT_BDF1) {
        MOLT_BDF1_ddx_advance_per(u, RHS, dudx_out);
        MOLT_BDF1_ddy_advance_per(u, RHS, dudy_out);
    } else if (this->method == MOLTEngine::MOLT_BDF1_HYBRID_FFT) {
        compute_ddx_FFT(u[lastStepIndex], dudx_out);
        compute_ddy_FFT(u[lastStepIndex], dudy_out);
    } else if (this->method == MOLTEngine::MOLT_BDF1_HYBRID_FD6) {
        compute_ddx_FD6(u[lastStepIndex], dudx_out);
        compute_ddy_FD6(u[lastStepIndex], dudy_out);
    } else {
        throw -1;
    }
}

void MOLTEngine::MOLT_BDF1_advance_per(std::vector<std::complex<double>*> input_field_hist, std::complex<double>* RHS, std::complex<double>* output) {
    const double alpha = beta/(kappa*dt);
    const double alpha2 = alpha*alpha;
    std::complex<double>* R1 = new std::complex<double>[Nx*Ny];
    std::complex<double>* R2 = new std::complex<double>[Nx*Ny];

    for (int i = 0; i < Nx*Ny; i++) {
        R1[i] = 2.0*input_field_hist[lastStepIndex-1][i] - input_field_hist[lastStepIndex-2][i] + 1.0/alpha2*RHS[i];
    }

    get_L_y_inverse_per(R1, R2);
    get_L_x_inverse_per(R2, output);

    delete[] R1;
    delete[] R2;
}

void MOLTEngine::MOLT_BDF1_ddx_advance_per(std::vector<std::complex<double>*> input_field_hist, std::complex<double>* RHS, std::complex<double>* output) {
    const double alpha = beta/(kappa*dt);
    const double alpha2 = alpha*alpha;
    std::complex<double>* R1 = new std::complex<double>[Nx*Ny];
    std::complex<double>* R2 = new std::complex<double>[Nx*Ny];

    for (int i = 0; i < Nx*Ny; i++) {
        R1[i] = 2.0*input_field_hist[lastStepIndex-1][i] - input_field_hist[lastStepIndex-2][i] + 1.0/alpha2*RHS[i];
    }

    get_L_y_inverse_per(R1, R2);
    get_ddx_L_x_inverse_per(R2, output);

    delete[] R1;
    delete[] R2;
}

void MOLTEngine::MOLT_BDF1_ddy_advance_per(std::vector<std::complex<double>*> input_field_hist, std::complex<double>* RHS, std::complex<double>* output) {
    const double alpha = beta/(kappa*dt);
    const double alpha2 = alpha*alpha;
    std::complex<double>* R1 = new std::complex<double>[Nx*Ny];
    std::complex<double>* R2 = new std::complex<double>[Nx*Ny];

    for (int i = 0; i < Nx*Ny; i++) {
        R1[i] = 2.0*input_field_hist[lastStepIndex-1][i] - input_field_hist[lastStepIndex-2][i] + 1.0/alpha2*RHS[i];
    }

    get_ddy_L_y_inverse_per(R1, R2);
    get_L_x_inverse_per(R2, output);

    delete[] R1;
    delete[] R2;
}

void MOLTEngine::get_L_x_inverse_per(std::complex<double>* u, std::complex<double>* inverseOut) {
    
    const double alpha = beta/(kappa*dt);
    const double mu_x = std::exp(-alpha*( x[Nx-1] - x[0] + dx ) );

    const int J_N = Nx+1;

    std::vector<std::complex<double>> u_ext(Nx+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    std::vector<std::complex<double>> J(J_N);
    // std::vector<std::vector<std::complex<double>>> inverse(Nx*Ny);

    // Go row by row
    for (int j = 0; j < Ny; j++) {

        int idx_Nm1 = (Ny-1)*Ny + j;
        int idx_Nm2 = (Ny-2)*Ny + j;

        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];

        for (int i = 0; i < Nx; i++) {
            int idx = i*Ny + j;
            u_ext[i+2] = u[idx];
        }

        int idx0 = 0*Ny + j;
        int idx1 = 1*Ny + j;
        int idx2 = 2*Ny + j;
        u_ext[Nx+2] = u[idx0];
        u_ext[Nx+3] = u[idx1];
        u_ext[Nx+4] = u[idx2];

        linear5_L(u_ext, alpha, J_R);
        linear5_R(u_ext, alpha, J_L);

        fast_convolution(J_R, J_L, alpha);

        for (int i = 0; i < J_N; i++) {
            J[i] = .5*(J_L[i] + J_R[i]);
        }

        double I_a = J[0].real();
        double I_b = J[J_N-1].real();
        
        double A_x = I_b/(1-mu_x);
        double B_x = I_a/(1-mu_x);

        apply_A_and_B(J, x, dx, Nx, alpha, A_x, B_x);

        for (int i = 0; i < Nx; i++) {
            int idx = i*Ny + j;
            inverseOut[idx] = J[i];
        }
    }
    
}

void MOLTEngine::get_L_y_inverse_per(std::complex<double>* u, std::complex<double>* inverseOut) {
    
    const double alpha = beta/(kappa*dt);
    const double mu_y = std::exp(-alpha*( y[Ny-1] - y[0] + dy ) );

    const int J_N = Ny+1;

    std::vector<std::complex<double>> u_ext(Ny+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    std::vector<std::complex<double>> J(J_N);

    // Go column by column
    for (int i = 0; i < Nx; i++) {
        int idx_Nm1 = i*Ny + (Nx-1);
        int idx_Nm2 = i*Ny + (Nx-2);
        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];
        for (int j = 0; j < Ny; j++) {
            int idx = i*Ny + j;
            u_ext[j+2] = u[idx];
        }
        int idx0 = i*Ny + 0;
        int idx1 = i*Ny + 1;
        int idx2 = i*Ny + 2;
        u_ext[Ny+2] = u[idx0];
        u_ext[Ny+3] = u[idx1];
        u_ext[Ny+4] = u[idx2];

        linear5_L(u_ext, alpha, J_R);
        linear5_R(u_ext, alpha, J_L);

        fast_convolution(J_R, J_L, alpha);

        for (int j = 0; j < J_N; j++) {
            J[j] = .5*(J_L[j] + J_R[j]);
        }

        double I_a = J[0].real();
        double I_b = J[J_N-1].real();
        
        double A_y = I_b/(1-mu_y);
        double B_y = I_a/(1-mu_y);

        apply_A_and_B(J, y, dy, Ny, alpha, A_y, B_y);

        for (int j = 0; j < Ny; j++) {
            int idx = i*Ny + j;
            inverseOut[idx] = J[j];
        }
    }
}

void MOLTEngine::get_ddx_L_x_inverse_per(std::complex<double>* u, std::complex<double>* ddxOut) {

    const double alpha = beta/(kappa*dt);
    const double mu_x = std::exp(-alpha*( x[Nx-1] - x[0] + dx ));

    const int J_N = Nx+1;

    std::vector<std::complex<double>> u_ext(Nx+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    std::vector<std::complex<double>> J(J_N);
    std::vector<std::complex<double>> ddx(J_N);
    // std::vector<std::vector<std::complex<double>>> inverse(Nx*Ny);

    // Go column by column
    for (int j = 0; j < Ny; j++) {
        int idx_Nm1 = (Ny-1)*Ny + j;
        int idx_Nm2 = (Ny-2)*Ny + j;
        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];
        for (int i = 0; i < Nx; i++) {
            int idx = i*Ny + j;
            u_ext[i+2] = u[idx];
        }
        int idx0 = 0*Ny + j;
        int idx1 = 1*Ny + j;
        int idx2 = 2*Ny + j;
        u_ext[Nx+2] = u[idx0];
        u_ext[Nx+3] = u[idx1];
        u_ext[Nx+4] = u[idx2];

        linear5_L(u_ext, alpha, J_R);
        linear5_R(u_ext, alpha, J_L);

        fast_convolution(J_R, J_L, alpha);

        for (int i = 0; i < J_N; i++) {
            J[i] = .5*(J_L[i] + J_R[i]);
        }

        double I_a = J[0].real();
        double I_b = J[J_N-1].real();
        
        double A_x = I_b/(1-mu_x);
        double B_x = I_a/(1-mu_x);

        // apply_A_and_B(J, y, Ny, alpha, A_y, B_y);
        for (int i = 0; i < J_N-1; i++) {
            double mu_i1 = std::exp(-alpha*(x[i]-x[0]));
            double mu_i2 = std::exp(-alpha*(x[Nx-1]-x[i]+dx));
            ddx[i] = .5*alpha*(-J_R[i] + J_L[i]) - alpha*A_x*mu_i1 + alpha*B_x*mu_i2;
        }
        double mu_i1 = std::exp(-alpha*(x[Nx-1]+dx - x[0]));
        double mu_i2 = std::exp(-alpha*(0));
        ddx[J_N-1] = .5*alpha*(-J_R[J_N-1] + J_L[J_N-1]) - alpha*A_x*mu_i1 + alpha*B_x*mu_i2;

        for (int i = 0; i < Nx; i++) {
            int idx = i*Ny + j;
            ddxOut[idx] = ddx[i];
        }
    }
}

void MOLTEngine::get_ddy_L_y_inverse_per(std::complex<double>* u, std::complex<double>* ddyOut) {

    const double alpha = beta/(kappa*dt);
    const double mu_y = std::exp(-alpha*( y[Ny-1] - y[0] + dy ));

    const int J_N = Ny+1;

    std::vector<std::complex<double>> u_ext(Ny+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    std::vector<std::complex<double>> J(J_N);
    std::vector<std::complex<double>> ddy(J_N);

    // Go column by column
    for (int i = 0; i < Nx; i++) {
        int idx_Nm1 = i*Ny + (Nx-1);
        int idx_Nm2 = i*Ny + (Nx-2);
        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];
        for (int j = 0; j < Ny; j++) {
            int idx = i*Ny + j;
            u_ext[j+2] = u[idx];
        }
        int idx0 = i*Ny + 0;
        int idx1 = i*Ny + 1;
        int idx2 = i*Ny + 2;
        u_ext[Ny+2] = u[idx0];
        u_ext[Ny+3] = u[idx1];
        u_ext[Ny+4] = u[idx2];

        linear5_L(u_ext, alpha, J_R);
        linear5_R(u_ext, alpha, J_L);

        fast_convolution(J_R, J_L, alpha);

        for (int j = 0; j < J_N; j++) {
            J[j] = .5*(J_L[j] + J_R[j]);
        }

        double I_a = J[0].real();
        double I_b = J[J_N-1].real();
        
        double A_y = I_b/(1-mu_y);
        double B_y = I_a/(1-mu_y);

        // apply_A_and_B(J, y, Ny, alpha, A_y, B_y);
        for (int j = 0; j < J_N-1; j++) {
            double mu_j1 = std::exp(-alpha*(y[j]-y[0]));
            double mu_j2 = std::exp(-alpha*(y[Ny-1]-y[j]+dy));
            ddy[j] = .5*alpha*(-J_R[j] + J_L[j]) - alpha*A_y*mu_j1 + alpha*B_y*mu_j2;
        }
        double mu_j1 = std::exp(-alpha*(y[Ny-1]+dy - y[0]));
        double mu_j2 = std::exp(-alpha*(0));
        ddy[J_N-1] = .5*alpha*(-J_R[J_N-1] + J_L[J_N-1]) - alpha*A_y*mu_j1 + alpha*B_y*mu_j2;
        for (int j = 0; j < Ny; j++) {
            int idx = i*Ny + j;
            ddyOut[idx] = ddy[j];
        }
    }

}

void MOLTEngine::linear5_L(std::vector<std::complex<double>> v_ext, double gamma, std::vector<std::complex<double>>& J_L) {
    /*
     * Compute the fifth order approximation to the 
     * left convolution integral using a six point global stencil
     * and linear weights.
     */

    // std::vector<std::complex<double>> J_L(N);

    // We need gamma*dx here, so we adjust the value of gamma
    const double gam = gamma*dx;
    const double gam2 = gam*gam;
    const double gam3 = gam*gam*gam;
    const double gam4 = gam*gam*gam*gam;

    const double exp_neg_gam = std::exp(-gam);
    
    // Get the total number of elements in v_ext (N = N_ext - 5)
    const int N_ext = v_ext.size();
    
    /****************************************************************************************************
    * Compute weights for the quadrature using the precomputed expressions for the left approximation
    *
    * Note: Can precompute these at the beginning of the simulation and load them later for speed
    *****************************************************************************************************/
    const double cl_11 = ( 6.0 - 6.0*gam + 2.0*gam2 - ( 6.0 - gam2 )*exp_neg_gam )/(6.0*gam3);
    const double cl_12 = -( 6.0 - 8.0*gam + 3.0*gam2 - ( 6.0 - 2.0*gam - 2.0*gam2 )*exp_neg_gam )/(2.0*gam3);
    const double cl_13 = ( 6.0 - 10.0*gam + 6.0*gam2 - ( 6.0 - 4.0*gam - gam2 + 2.0*gam3 )*exp_neg_gam )/(2.0*gam3);
    const double cl_14 = -( 6.0 - 12.0*gam + 11.0*gam2 - 6.0*gam3 - ( 6.0 - 6.0*gam + 2.0*gam2)*exp_neg_gam )/(6.0*gam3);
    const double cl_21 = ( 6.0 - gam2 - ( 6.0 + 6.0*gam + 2.0*gam2 )*exp_neg_gam )/(6.0*gam3);
    const double cl_22 = -( 6.0 - 2.0*gam - 2.0*gam2 - ( 6.0 + 4.0*gam - gam2 - 2.0*gam3 )*exp_neg_gam )/(2.0*gam3);
    const double cl_23 = ( 6.0 - 4.0*gam - gam2 + 2.0*gam3 - ( 6.0 + 2.0*gam - 2.0*gam2 )*exp_neg_gam )/(2.0*gam3);
    const double cl_24 = -( 6.0 - 6.0*gam + 2.0*gam2 - ( 6.0 - gam2 )*exp_neg_gam )/(6.0*gam3);
    const double cl_31 = ( 6.0 + 6.0*gam + 2.0*gam2 - ( 6.0 + 12*gam + 11.0*gam2 + 6.0*gam3 )*exp_neg_gam )/(6.0*gam3);
    const double cl_32 = -( 6.0 + 4.0*gam - gam2 - 2.0*gam3 - ( 6.0 + 10.0*gam + 6.0*gam2 )*exp_neg_gam )/(2.0*gam3 );
    const double cl_33 = ( 6.0 + 2.0*gam - 2.0*gam2 - ( 6.0 + 8.0*gam + 3.0*gam2 )*exp_neg_gam )/(2.0*gam3 );
    const double cl_34 = -( 6.0 - gam2 - ( 6.0 + 6.0*gam + 2.0*gam2 )*exp_neg_gam )/(6.0*gam3);
        
    /****************************************************************************************************
    * Compute the linear WENO weights
    * Note: Can precompute these at the beginning of the simulation and load them later for speed
    *****************************************************************************************************/
    double d1 = ( 60.0 - 15.0*gam2 + 2.0*gam4 - ( 60.0 + 60.0*gam + 15.0*gam2 - 5.0*gam3 - 3.0*gam4)*exp_neg_gam );
    d1 = d1/(10.0*(gam2)*( 6.0 - 6.0*gam + 2.0*gam2 - ( 6.0 - gam2 )*exp_neg_gam ) );

    double d3 = ( 60.0 - 60.0*gam + 15.0*gam2 + 5.0*gam3 - 3.0*gam4 - ( 60.0 - 15.0*gam2 + 2.0*gam4)*exp_neg_gam ) ;
    d3 = d3/(10.0*(gam2)*( 6.0 - gam2 - ( 6.0 + 6.0*gam + 2.0*gam2 )*exp_neg_gam ) );

    double d2 = 1.0 - d1 - d3;
        
    /****************************************************************************************************
    * Compute the local integrals J_{i}^{L} on x_{i-1} to x_{i}, i = 1,...,N+1
    *****************************************************************************************************/

    J_L[0] = 0.0;
    
    // Loop through the interior points of the extended array
    // Offset is from the left end-point being excluded
    for (int i = 3; i < N_ext-2; i++) { // for i = 4:N_ext-2
        // Polynomial interpolants on the smaller stencils
        std::complex<double> p1 = cl_11*v_ext[i-3] + cl_12*v_ext[i-2] + cl_13*v_ext[i-1] + cl_14*v_ext[i  ];
        std::complex<double> p2 = cl_21*v_ext[i-2] + cl_22*v_ext[i-1] + cl_23*v_ext[i  ] + cl_24*v_ext[i+1];
        std::complex<double> p3 = cl_31*v_ext[i-1] + cl_32*v_ext[i  ] + cl_33*v_ext[i+1] + cl_34*v_ext[i+2];
        
        // Compute the integral using the nonlinear weights and the local polynomials
        J_L[i-2] = d1*p1 + d2*p2 + d3*p3;
    }
}

void MOLTEngine::linear5_R(std::vector<std::complex<double>> v_ext, double gamma, std::vector<std::complex<double>>& J_R) {
    /*
     * Compute the fifth order approximation to the 
     * right convolution integral using a six point global stencil
     * and linear weights.
     */

    // J_R = zeros(1,N);

    const double gam = gamma*dx;
    const double gam2 = gam*gam;
    const double gam3 = gam*gam*gam;
    const double gam4 = gam*gam*gam*gam;

    const double exp_neg_gam = std::exp(-gam);
    
    // Get the total number of elements in v_ext (N = N_ext - 5)
    const int N_ext = v_ext.size();
    
    /****************************************************************************************************
     * Compute weights for the quadrature using the precomputed expressions for the left approximation
     *
     * Note: Can precompute these at the beginning of the simulation and load them later for speed
     *****************************************************************************************************/
    const double cr_34 = ( 6.0 - 6.0*gam + 2.0*gam2 - ( 6.0 - gam2 )*exp_neg_gam )/(6.0*gam3);
    const double cr_33 = -( 6.0 - 8.0*gam + 3.0*gam2 - ( 6.0 - 2.0*gam - 2.0*gam2 )*exp_neg_gam )/(2.0*gam3);
    const double cr_32 = ( 6.0 - 10.0*gam + 6.0*gam2 - ( 6.0 - 4.0*gam - gam2 + 2.0*gam3 )*exp_neg_gam )/(2.0*gam3);
    const double cr_31 = -( 6.0 - 12.0*gam + 11.0*gam2 - 6.0*gam3 - ( 6.0 - 6.0*gam + 2.0*gam2)*exp_neg_gam )/(6.0*gam3);
    const double cr_24 = ( 6.0 - gam2 - ( 6.0 + 6.0*gam + 2.0*gam2 )*exp_neg_gam )/(6.0*gam3);
    const double cr_23 = -( 6.0 - 2.0*gam - 2.0*gam2 - ( 6.0 + 4.0*gam - gam2 - 2.0*gam3 )*exp_neg_gam )/(2.0*gam3);
    const double cr_22 = ( 6.0 - 4.0*gam - gam2 + 2.0*gam3 - ( 6.0 + 2.0*gam - 2.0*gam2 )*exp_neg_gam )/(2.0*gam3);
    const double cr_21 = -( 6.0 - 6.0*gam + 2.0*gam2 - ( 6.0 - gam2 )*exp_neg_gam )/(6.0*gam3);
    const double cr_14 = ( 6.0 + 6.0*gam +2.0*gam2 - ( 6.0 + 12.0*gam + 11.0*gam2 + 6.0*gam3 )*exp_neg_gam )/(6.0*gam3);
    const double cr_13 = -( 6.0 + 4.0*gam - gam2 - 2.0*gam3 - ( 6.0 + 10.0*gam + 6.0*gam2 )*exp_neg_gam )/(2.0*gam3 );
    const double cr_12 = ( 6.0 + 2.0*gam - 2.0*gam2 - ( 6.0 + 8.0*gam + 3.0*gam2 )*exp_neg_gam )/(2.0*gam3 );
    const double cr_11 = -( 6.0 - gam2 - ( 6.0 + 6.0*gam + 2.0*gam2 )*exp_neg_gam )/(6.0*gam3);
    
    /****************************************************************************************************
     * Compute the linear WENO weights
     * Note: Can precompute these at the beginning of the simulation and load them later for speed
     *****************************************************************************************************/
    double d3 = ( 60.0 - 15.0*gam2 + 2.0*gam4 - ( 60.0 + 60.0*gam + 15.0*gam2 - 5.0*gam3 - 3.0*gam4)*exp_neg_gam );
    d3 = d3/(10.0*(gam2)*( 6.0 - 6.0*gam + 2.0*gam2 - ( 6.0 - gam2 )*exp_neg_gam ) );
    
    double d1 = ( 60.0 - 60.0*gam + 15.0*gam2 + 5.0*gam3 - 3.0*gam4 - ( 60.0 - 15.0*gam2 + 2.0*gam4)*exp_neg_gam );
    d1 = d1/(10.0*(gam2)*( 6.0 - gam2 - ( 6.0 + 6.0*gam + 2.0*gam2 )*exp_neg_gam ) );
    
    double d2 = 1 - d1 - d3;
        
    /****************************************************************************************************
     * Compute the local integrals J_{i}^{R} on x_{i} to x_{i+1}, i = 0,...,N
     *****************************************************************************************************/
    
    // Loop through the interior points
    // Offset is from the right end-point being excluded
    // for i = 3:N_ext-3
    for (int i = 2; i < N_ext-3; i++) {    
        // Polynomial interpolants on the smaller stencils
        std::complex<double> p1 = cr_11*v_ext[i-2] + cr_12*v_ext[i-1] + cr_13*v_ext[i  ] + cr_14*v_ext[i+1];
        std::complex<double> p2 = cr_21*v_ext[i-1] + cr_22*v_ext[i  ] + cr_23*v_ext[i+1] + cr_24*v_ext[i+2];
        std::complex<double> p3 = cr_31*v_ext[i  ] + cr_32*v_ext[i+1] + cr_33*v_ext[i+2] + cr_34*v_ext[i+3];
        
        // Compute the integral using the nonlinear weights and the local polynomials
        J_R[i-2] = d1*p1 + d2*p2 + d3*p3;
    }
    J_R[J_R.size()-1] = 0.0;
    
}

void MOLTEngine::fast_convolution(std::vector<std::complex<double>> &I_L, std::vector<std::complex<double>> &I_R, double alpha) {

    const int N = I_L.size();
    
    // Precompute the recursion weight
    double weight = std::exp( -alpha*dx );
    
    // Perform the sweeps to the right
    // for i = 2:N
    for (int i = 1; i < N; i++) {
        I_L[i] = weight*I_L[i-1] + I_L[i];
    }
    
    // Perform the sweeps to the left
    // for i = N-1:-1:1
    for (int i = N-1; i >= 0; i--) {
        I_R[i] = weight*I_R[i+1] + I_R[i];
    }
}

void MOLTEngine::apply_A_and_B(std::vector<std::complex<double>> &I_, double* x, double dx, int N, double alpha, double A, double B) {

    int I_len = I_.size();
    int Lx = x[N-1] - x[0] + dx;

    for (int i = 0; i < I_len-1; i++) {
        I_[i] += A*std::exp(-alpha*( x[i  ] - x[0] ));
        I_[i] += B*std::exp(-alpha*( Lx - x[i] ));
    }
    I_[I_len-1] += A*std::exp(-alpha*( Lx - x[0] ));
    I_[I_len-1] += B*std::exp(-alpha*( 0 ));
}

/**
 * Name: computeFirstDerivative_FFT
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Computes the first derivative in either the x or y direction of a 2D mesh of complex numbers using the FFTW.
 *              Assumes a periodic domain.
 * Inputs: inputField, derivativeField, isDerivativeInX (boolean indicating which direction the derivative is in)
 * Output: technically none, but derivativeField is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: fftw, to_std_complex
 */
void MOLTEngine::computeFirstDerivative_FFT(std::complex<double>* inputField, 
                                            std::complex<double>* derivativeField,
                                            bool isDerivativeInX) {

    // Execute the forward FFT
    // fft_execute_dft(plan, in, out)
    fftw_execute_dft(forward_plan, reinterpret_cast<fftw_complex*>(inputField), reinterpret_cast<fftw_complex*>(forwardOut));

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = isDerivativeInX ? kx_deriv_1 : ky_deriv_1;

    // Apply the derivative operator in the frequency domain
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(forwardOut)[index]);
            if (isDerivativeInX) {
                freq_component *= std::complex<double>(0, k[i]); // Multiply by i * kx
            } else {
                freq_component *= std::complex<double>(0, k[j]); // Multiply by i * ky
            }
            reinterpret_cast<fftw_complex*>(backwardIn)[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(backwardIn)[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx*Ny; i++) {
        derivativeField[i] = backwardOut[i] / double(Nx * Ny);
    }
}

void MOLTEngine::computeFirstDerivative_FD6(std::complex<double>* inputField, 
                                            std::complex<double>* derivativeField,
                                            bool isDerivativeInX) {

    double h = isDerivativeInX ? dx : dy;
    double w1 = 3.0/4.0;
    double w2 = 3.0/20.0;
    double w3 = 1.0/60.0;
    // Apply the derivative operator in the frequency domain
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int i_m3 = isDerivativeInX ? (i-3+Nx) : i;
            int i_m2 = isDerivativeInX ? (i-2+Nx) : i;
            int i_m1 = isDerivativeInX ? (i-1+Nx) : i;
            int i_p1 = isDerivativeInX ? (i+1   ) : i;
            int i_p2 = isDerivativeInX ? (i+2   ) : i;
            int i_p3 = isDerivativeInX ? (i+3   ) : i;
            
            int j_m3 = isDerivativeInX ? j : (j-3+Ny) % Ny;
            int j_m2 = isDerivativeInX ? j : (j-2+Ny) % Ny;
            int j_m1 = isDerivativeInX ? j : (j-1+Ny) % Ny;
            int j_p1 = isDerivativeInX ? j : (j+1   ) % Ny;
            int j_p2 = isDerivativeInX ? j : (j+2   ) % Ny;
            int j_p3 = isDerivativeInX ? j : (j+3   ) % Ny;
            
            int idx_m3 = i_m3 * Ny + j_m3;
            int idx_m2 = i_m2 * Ny + j_m2;
            int idx_m1 = i_m1 * Ny + j_m1;
            int idx    = i    * Ny + j;
            int idx_p3 = i_p3 * Ny + j_p3;
            int idx_p2 = i_p2 * Ny + j_p2;
            int idx_p1 = i_p1 * Ny + j_p1;

            derivativeField[idx] = (-w3*inputField[idx_m3] + w2*inputField[idx_m2] - w1*inputField[idx_m1] + w1*inputField[idx_p1] - w2*inputField[idx_p2] + w3*inputField[idx_p3]) / h;
        }
    }
}

/**
 * Name: computeSecondDerivative_FFT
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Computes the second derivative in either the x or y direction of a 2D mesh of complex numbers using the FFTW.
 *              Assumes a periodic domain.
 * Inputs: inputField, derivativeField, isDerivativeInX  (boolean indicating which direction the derivative is in)
 * Output: technically none, but derivativeField is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: fftw, to_std_complex
 */
void MOLTEngine::computeSecondDerivative_FFT(std::complex<double>* inputField, 
                                             std::complex<double>* derivativeField,
                                             bool isDerivativeInX) {

    // Execute the forward FFT
    fftw_execute_dft(forward_plan, reinterpret_cast<fftw_complex*>(inputField), reinterpret_cast<fftw_complex*>(forwardOut));

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = isDerivativeInX ? kx_deriv_2 : ky_deriv_2;

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(forwardOut)[index]);
            double k_val = isDerivativeInX ? k[i] : k[j];
            freq_component *= -k_val * k_val; // Multiply by -k^2
            reinterpret_cast<fftw_complex*>(backwardIn)[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(backwardIn)[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx*Ny; i++) {
        derivativeField[i] = backwardOut[i] / double(Nx * Ny);
    }
}

// void MOLTEngine::sortParticlesByLocation() {

//     std::vector<int> particleCounts = new int[Nx*Ny];

//     // Red-Black Bucket Sort

//     // Count up the number of particles at each location
//     for (int i = 0; i < Np; i++) {

//         double charge_value = q_ele*w_elec;

//         double x_p = this->method == MOLTEngine::CDF1 ? ( (*x_elec[lastStepIndex])[i] + (*x_elec[lastStepIndex-1])[i] ) / 2 : (*x_elec[lastStepIndex])[i];
//         double y_p = this->method == MOLTEngine::CDF1 ? ( (*y_elec[lastStepIndex])[i] + (*y_elec[lastStepIndex-1])[i] ) / 2 : (*y_elec[lastStepIndex])[i];

//         // We convert from cartesian to logical space
//         int lc_x = floor((x_p - x[0])/dx);
//         int lc_y = floor((y_p - y[0])/dy);

//         const int ld = lc_x * Ny + lc_y; // (left,down)  lc_x,   lc_y

//         // Now we acquire the particle value and add it to the corresponding field
//         particleCounts[ld]++;
//     }

// }
