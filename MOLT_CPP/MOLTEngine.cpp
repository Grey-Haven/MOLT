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
    computeGaugeL2();
    gettimeofday( &end5, NULL );
    // std::cout << "Shuffling Steps" << std::endl;
    if (n % 5000 == 0) {
        gettimeofday( &begin7, NULL );
        print();
        gettimeofday( &end7, NULL );
        timeComponent7 += 1.0 * ( end7.tv_sec - begin7.tv_sec ) + 1.0e-6 * ( end7.tv_usec - begin7.tv_usec );
    }
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
    std::cout << "computeGaugeL2(): " << timeComponent5 << std::endl;
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

double MOLTEngine::getTotalCharge() {
    return rhoTotal;
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

            if (this->method == MOLTEngine::BDF1 || this->method == MOLTEngine::CDF1) {
                ddt_phi = (phi[lastStepIndex][i].real() - phi[lastStepIndex-1][i].real());
            } else if (this->method == MOLTEngine::BDF2) {
                ddt_phi = (phi[lastStepIndex][i].real() - (4.0/3.0)*phi[lastStepIndex-1][i].real() + (1.0/3.0)*phi[lastStepIndex-2][i].real()) / ((2.0/3.0));
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
 * Date Last Modified: 6/14/24 (Stephen White)
 * Description: Computes the L2 error of the residual of Gauss's Law for Electricity (div(E) = -grad(phi) - dA/dt = rho/eps_0)
 * Inputs: none (relies on phi, ddx_phi, ddy_phi, A1, A2, ddx_A1, ddy_A2)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::computeGaussL2() {
    // double ddt_phi;
    // double div_A;
    double l2 = 0;
    double ddt_A1;
    double ddt_A2;
    
    for (int i = 0; i < Nx*Ny; i++) {
        if (this->method == MOLTEngine::DIRK2 || this->method == MOLTEngine::DIRK3) {

        } else {
            for (int i = 0; i < Nx*Ny; i++) {
                if (this->method == MOLTEngine::BDF1 || this->method == MOLTEngine::CDF1) {
                    for (int i = 0; i < Nx*Ny; i++) {
                        ddt_A1 = (A1[lastStepIndex][i].real() - A1[lastStepIndex-1][i].real()) / dt;
                        ddt_A2 = (A2[lastStepIndex][i].real() - A2[lastStepIndex-1][i].real()) / dt;
                    }
                } else if (this->method == MOLTEngine::BDF2) {
                    for (int i = 0; i < Nx*Ny; i++) {
                        ddt_A1 = (A1[lastStepIndex][i].real() - (4.0/3.0)*A1[lastStepIndex-1][i].real() + (1.0/3.0)*A1[lastStepIndex-2][i].real()) / ((2.0/3.0)*dt);
                        ddt_A2 = (A2[lastStepIndex][i].real() - (4.0/3.0)*A2[lastStepIndex-1][i].real() + (1.0/3.0)*A2[lastStepIndex-2][i].real()) / ((2.0/3.0)*dt);
                    }
                }
                E1[i] = -ddx_phi[lastStepIndex][i] - ddt_A1;
                E2[i] = -ddy_phi[lastStepIndex][i] - ddt_A2;
            }
        }
    }
    compute_ddx(E1, ddx_E1);
    compute_ddy(E2, ddy_E2);

    // res = div(E) - rho/sigma_1 (sigma_1 nondimensionalized eps_0)
    for (int i = 0; i < Nx*Ny; i++) {
        l2 += std::pow(ddx_E1[i].real() + ddy_E2[i].real() - rho[lastStepIndex][i].real() / sigma_1, 2);
    }

    gaussL2_divE = std::sqrt(dx*dy*l2);
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
    std::ofstream electronFile;
    // std::ofstream rhoFile, J1File, J2File;
    std::string nxn = std::to_string(Nx) + "x" + std::to_string(Ny);
    std::string subpath;
    if (this->method == MOLTEngine::CDF1) {
        subpath = "CDF1_mod_debye";
    } else if (this->method == MOLTEngine::BDF1) {
        subpath = "BDF1_mod_debye";
    } else if (this->method == MOLTEngine::BDF2) {
        subpath = "BDF2_mod_debye";
    } else if (this->method == MOLTEngine::DIRK2) {
        subpath = "DIRK2_mod_debye";
    } else if (this->method == MOLTEngine::DIRK3) {
        subpath = "DIRK3_mod_debye";
    }
    std::string path = "results/" + subpath + "/" + nxn + "/";
    std::string nstr = std::to_string(n);
    int numlen = 5;
    
    std::ostringstream padder;
    padder << std::internal << std::setfill('0') << std::setw(numlen) << n;
    std::string paddedNum = padder.str();
    electronFile.open(path + "elec_" + paddedNum + ".csv");
    phiFile.open(path + "phi_" + paddedNum + ".csv");
    A1File.open(path + "A1_" + paddedNum + ".csv");
    A2File.open(path + "A2_" + paddedNum + ".csv");
    // rhoFile.open(path + "rho_" + paddedNum + ".csv");
    // J1File.open(path + "J1_" + paddedNum + ".csv");
    // J2File.open(path + "J2_" + paddedNum + ".csv");
    int idx = 0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny-1; j++) {
            idx = i*Ny + j;
            phiFile << std::to_string(phi[lastStepIndex][idx].real()) + ",";
            A1File << std::to_string(A1[lastStepIndex][idx].real()) + ",";
            A2File << std::to_string(A2[lastStepIndex][idx].real()) + ",";
            // rhoFile << std::to_string(rho[lastStepIndex][i][j].real()) + ",";
            // J1File << std::to_string(J1[lastStepIndex][i][j].real()) + ",";
            // J2File << std::to_string(J2[lastStepIndex][i][j].real()) + ",";
        }
        idx = i*Ny + Ny-1;
        phiFile << std::to_string(phi[lastStepIndex][idx].real());
        A1File << std::to_string(A1[lastStepIndex][idx].real());
        A2File << std::to_string(A2[lastStepIndex][idx].real());
        // rhoFile << std::to_string(rho[lastStepIndex][i][Ny-1].real());
        // J1File << std::to_string(J1[lastStepIndex][i][Ny-1].real());
        // J2File << std::to_string(J2[lastStepIndex][i][Ny-1].real());
        phiFile << "\n";
        A1File << "\n";
        A2File << "\n";
        // rhoFile << "\n";
        // J1File << "\n";
        // J2File << "\n";
    }
    // for (int i = 0; i < Nx; i++) {
    //     for (int j = 0; j < Ny-1; j++) {
    //         phiFile << std::to_string((*phi[lastStepIndex])[i][j].real()) + ",";
    //         A1File << std::to_string((*A1[lastStepIndex])[i][j].real()) + ",";
    //         A2File << std::to_string((*A2[lastStepIndex])[i][j].real()) + ",";
    //         // rhoFile << std::to_string(rho[lastStepIndex][i][j].real()) + ",";
    //         // J1File << std::to_string(J1[lastStepIndex][i][j].real()) + ",";
    //         // J2File << std::to_string(J2[lastStepIndex][i][j].real()) + ",";
    //     }
    //     phiFile << std::to_string((*phi[lastStepIndex])[i][Ny-1].real());
    //     A1File << std::to_string((*A1[lastStepIndex])[i][Ny-1].real());
    //     A2File << std::to_string((*A2[lastStepIndex])[i][Ny-1].real());
    //     // rhoFile << std::to_string(rho[lastStepIndex][i][Ny-1].real());
    //     // J1File << std::to_string(J1[lastStepIndex][i][Ny-1].real());
    //     // J2File << std::to_string(J2[lastStepIndex][i][Ny-1].real());
    //     phiFile << "\n";
    //     A1File << "\n";
    //     A2File << "\n";
    //     // rhoFile << "\n";
    //     // J1File << "\n";
    //     // J2File << "\n";
    // }
    for (int p = 0; p < Np; p++) {
        electronFile << std::to_string((*x_elec[lastStepIndex])[p]) + "," + std::to_string((*y_elec[lastStepIndex])[p]) + "," + 
                        std::to_string((*vx_elec[lastStepIndex])[p]) + "," + std::to_string((*vy_elec[lastStepIndex])[p]) << "\n";
    }
    electronFile.close();
    phiFile.close();
    A1File.close();
    A2File.close();
    // rhoFile.close();
    // J1File.close();
    // J2File.close();
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
    double Lx = x[Nx-1] - x[0];
    double Ly = y[Ny-1] - y[0];

    double vx_star;
    double vy_star;

    #pragma omp parallel for
    for (int i = 0; i < Np; i++) {
        vx_star = 2.0*(*vx_elec[lastStepIndex-1])[i] - (*vx_elec[lastStepIndex-2])[i];
        vy_star = 2.0*(*vy_elec[lastStepIndex-1])[i] - (*vy_elec[lastStepIndex-2])[i];

        (*x_elec[lastStepIndex])[i] = (*x_elec[lastStepIndex-1])[i] + dt*vx_star;
        (*y_elec[lastStepIndex])[i] = (*y_elec[lastStepIndex-1])[i] + dt*vy_star;

        (*x_elec[lastStepIndex])[i] = (*x_elec[lastStepIndex])[i] - Lx*floor(((*x_elec[lastStepIndex])[i] - this->x[0]) / Lx);
        (*y_elec[lastStepIndex])[i] = (*y_elec[lastStepIndex])[i] - Ly*floor(((*y_elec[lastStepIndex])[i] - this->y[0]) / Ly);
    }
}

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

        std::complex<double>* S_1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* S_2 = new std::complex<double>[Nx*Ny];

        std::complex<double>* d2dx_u = new std::complex<double>[Nx*Ny];
        std::complex<double>* d2dy_u = new std::complex<double>[Nx*Ny];

        std::complex<double>* laplacian_u = new std::complex<double>[Nx*Ny];
        std::complex<double>* laplacian_u1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* laplacian_u2 = new std::complex<double>[Nx*Ny];

        std::complex<double>* RHS1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* RHS2 = new std::complex<double>[Nx*Ny];

        std::complex<double>* u1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* v1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* u2 = new std::complex<double>[Nx*Ny];
        std::complex<double>* v2 = new std::complex<double>[Nx*Ny];

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

        delete[] S_1;
        delete[] S_2;

        delete[] d2dx_u;
        delete[] d2dy_u;

        delete[] laplacian_u;
        delete[] laplacian_u1;
        delete[] laplacian_u2;

        delete[] RHS1;
        delete[] RHS2;

        delete[] u1;
        delete[] v1;
        delete[] u2;
        delete[] v2;
}

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

        std::complex<double>* S_1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* S_2 = new std::complex<double>[Nx*Ny];

        std::complex<double>* d2dx_u = new std::complex<double>[Nx*Ny];
        std::complex<double>* d2dy_u = new std::complex<double>[Nx*Ny];

        std::complex<double>* laplacian_u = new std::complex<double>[Nx*Ny];
        std::complex<double>* laplacian_u1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* laplacian_u2 = new std::complex<double>[Nx*Ny];

        std::complex<double>* RHS1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* RHS2 = new std::complex<double>[Nx*Ny];

        std::complex<double>* u1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* v1 = new std::complex<double>[Nx*Ny];
        std::complex<double>* u2 = new std::complex<double>[Nx*Ny];
        std::complex<double>* v2 = new std::complex<double>[Nx*Ny];

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

        delete[] S_1;
        delete[] S_2;

        delete[] d2dx_u;
        delete[] d2dy_u;

        delete[] laplacian_u;
        delete[] laplacian_u1;
        delete[] laplacian_u2;

        delete[] RHS1;
        delete[] RHS2;

        delete[] u1;
        delete[] v1;
        delete[] u2;
        delete[] v2;

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

    if (this->method == MOLTEngine::DIRK2 || this->method == MOLTEngine::DIRK3) {
        
        std::complex<double>* phi_src_prev = new std::complex<double>[Nx*Ny];
        std::complex<double>* A1_src_prev = new std::complex<double>[Nx*Ny];
        std::complex<double>* A2_src_prev = new std::complex<double>[Nx*Ny];

        for (int i = 0; i < Nx*Ny; i++) {
            phi_src[i] = 1.0/sigma_1 * rho[lastStepIndex][i];
            A1_src[i] = sigma_2 * J1[lastStepIndex][i];
            A2_src[i] = sigma_2 * J2[lastStepIndex][i];

            phi_src_prev[i] = 1.0/sigma_1 * rho[lastStepIndex-1][i];
            A1_src_prev[i] = sigma_2 * J1[lastStepIndex-1][i];
            A2_src_prev[i] = sigma_2 * J2[lastStepIndex-1][i];
        }
        
        std::complex<double>* ddt_phi_curr = new std::complex<double>[Nx*Ny];
        std::complex<double>* ddt_A1_curr = new std::complex<double>[Nx*Ny];
        std::complex<double>* ddt_A2_curr = new std::complex<double>[Nx*Ny];

        for (int i = 0; i < Nx*Ny; i++) {
            ddt_phi_curr[i] = (phi[lastStepIndex-1][i] - phi[lastStepIndex-2][i]) / dt;
            ddt_A1_curr[i]  = ( A1[lastStepIndex-1][i] - A1[lastStepIndex-2][i] ) / dt;
            ddt_A2_curr[i]  = ( A2[lastStepIndex-1][i] - A2[lastStepIndex-2][i] ) / dt;
        }

        // DIRK2_advance_per(phi[lastStepIndex-1], ddt_phi[0], phi[lastStepIndex], ddt_phi[1], phi_src_prev, phi_src);
        // DIRK2_advance_per(A1[lastStepIndex-1], ddt_A1[0], A1[lastStepIndex], ddt_A1[1], A1_src_prev, A1_src);
        // DIRK2_advance_per(A2[lastStepIndex-1], ddt_A2[0], A2[lastStepIndex], ddt_A2[1], A2_src_prev, A2_src);
        if (this->method == MOLTEngine::DIRK2) {
            DIRK2_advance_per(phi[lastStepIndex-1], ddt_phi_curr, phi[lastStepIndex], ddt_phi[1], phi_src_prev, phi_src);
            DIRK2_advance_per(A1[lastStepIndex-1], ddt_A1_curr, A1[lastStepIndex], ddt_A1[1], A1_src_prev, A1_src);
            DIRK2_advance_per(A2[lastStepIndex-1], ddt_A2_curr, A2[lastStepIndex], ddt_A2[1], A2_src_prev, A2_src);
        } else if (this->method == MOLTEngine::DIRK3) {
            DIRK3_advance_per(phi[lastStepIndex-1], ddt_phi_curr, phi[lastStepIndex], ddt_phi[1], phi_src_prev, phi_src);
            DIRK3_advance_per(A1[lastStepIndex-1], ddt_A1_curr, A1[lastStepIndex], ddt_A1[1], A1_src_prev, A1_src);
            DIRK3_advance_per(A2[lastStepIndex-1], ddt_A2_curr, A2[lastStepIndex], ddt_A2[1], A2_src_prev, A2_src);
        }
        delete[] phi_src_prev;
        delete[] A1_src_prev;
        delete[] A2_src_prev;

        delete[] ddt_phi_curr;
        delete[] ddt_A1_curr;
        delete[] ddt_A2_curr;

    } else if (this->method == MOLTEngine::DIRK3) {

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

    compute_ddx(phi[lastStepIndex], ddx_phi[lastStepIndex]);
    compute_ddy(phi[lastStepIndex], ddy_phi[lastStepIndex]);
    compute_ddx(A1[lastStepIndex],  ddx_A1[lastStepIndex]);
    compute_ddy(A1[lastStepIndex],  ddy_A1[lastStepIndex]);
    compute_ddx(A2[lastStepIndex],  ddx_A2[lastStepIndex]);
    compute_ddy(A2[lastStepIndex],  ddy_A2[lastStepIndex]);
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
    for (int i = 0; i < Np; i++) {
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

        double rhs1 = -elec_charge*ddx_phi_p + elec_charge*( ddx_A1_p*vx_star + ddx_A2_p*vy_star );
        double rhs2 = -elec_charge*ddy_phi_p + elec_charge*( ddy_A1_p*vx_star + ddy_A2_p*vy_star );

        // Compute the new momentum
        (*Px_elec[lastStepIndex])[i] = (*Px_elec[lastStepIndex-1])[i] + dt*rhs1;
        (*Py_elec[lastStepIndex])[i] = (*Py_elec[lastStepIndex-1])[i] + dt*rhs2;

        double denom = std::sqrt(std::pow((*Px_elec[lastStepIndex])[i] - elec_charge*A1_p, 2) +
                                 std::pow((*Py_elec[lastStepIndex])[i] - elec_charge*A2_p, 2) +
                                 std::pow(elec_mass*kappa, 2));

        // Compute the new velocity using the updated momentum
        (*vx_elec[lastStepIndex])[i] = (kappa*((*Px_elec[lastStepIndex])[i] - elec_charge*A1_p)) / denom;
        (*vy_elec[lastStepIndex])[i] = (kappa*((*Py_elec[lastStepIndex])[i] - elec_charge*A2_p)) / denom;
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

    #pragma omp parallel for
    for (int i = 0; i < Np; i++) {
        double vx_star = 2.0*(*vx_elec[lastStepIndex-1])[i] - (*vx_elec[lastStepIndex-2])[i];
        double vy_star = 2.0*(*vy_elec[lastStepIndex-1])[i] - (*vy_elec[lastStepIndex-2])[i];

        double x_value = elec_charge*vx_star*w_elec;
        double y_value = elec_charge*vy_star*w_elec;

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
    double volume = 1/(dx*dy);
    for (int i = 0; i < Nx*Ny; i++) {
        J1[lastStepIndex][i] *= volume;
        J2[lastStepIndex][i] *= volume;
    }

    // Compute div J
    compute_ddx(J1[lastStepIndex], ddx_J1);
    compute_ddy(J2[lastStepIndex], ddy_J2);

    if (this->method == MOLTEngine::BDF1 || this->method == MOLTEngine::CDF1) {
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

        std::complex<double>* ddx_J1_prev = new std::complex<double>[Nx*Ny];
        std::complex<double>* ddy_J2_prev = new std::complex<double>[Nx*Ny];

        compute_ddx(J1[lastStepIndex-1], ddx_J1_prev);
        compute_ddy(J2[lastStepIndex-1], ddy_J2_prev);

        for (int i = 0; i < Nx*Ny; i++) {

            std::complex<double> rhs_prev = ddx_J1_prev[i] + ddy_J2_prev[i];
            std::complex<double> rhs_curr = ddx_J1[i] + ddy_J2[i];

            std::complex<double> RHS_1 = (1-c1)*rhs_prev + c1*rhs_curr;
            std::complex<double> RHS_2 = (1-c2)*rhs_prev + c2*rhs_curr;

            rho[lastStepIndex][i] = rho[lastStepIndex-1][i] - dt*(b1*RHS_1 + b2*RHS_2);
        }

        delete[] ddx_J1_prev;
        delete[] ddy_J2_prev;
    }
    this->rhoTotal = 0;
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

    const int ld = lc_x * Ny + lc_y;          // (left,down)  lc_x,   lc_y
    const int lu = lc_x * Ny + lc_y_p1;       // (left, up)   lc_x,   lc_y+1
    const int rd = lc_x_p1 * Ny + lc_y;       // (rite, down) lc_x+1, lc_y
    const int ru = lc_x_p1 * Ny + lc_y_p1;    // (rite, up)   lc_x+1, lc_y+1

    if (lc_x >= Nx || lc_x < 0 || lc_y >= Ny || lc_y < 0) {
        std::cerr << lc_x << " " << lc_y << " OUT OF BOUNDS" << std::endl;
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
    field[lu] += (1-fx)*(fy)*value;
    field[rd] += (fx)*(1-fy)*value;
    field[ru] += (fx)*(fy)*value;
}

/**
 * Name: computeFirstDerivative
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Computes the first derivative in either the x or y direction of a 2D mesh of complex numbers using the FFTW.
 *              Assumes a periodic domain.
 * Inputs: inputField, derivativeField, isDerivativeInX (boolean indicating which direction the derivative is in)
 * Output: technically none, but derivativeField is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: fftw, to_std_complex
 */
void MOLTEngine::computeFirstDerivative(std::complex<double>* inputField, 
                                        std::complex<double>* derivativeField,
                                        bool isDerivativeInX) {

    // Execute the forward FFT
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

/**
 * Name: computeSecondDerivative
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: Computes the second derivative in either the x or y direction of a 2D mesh of complex numbers using the FFTW.
 *              Assumes a periodic domain.
 * Inputs: inputField, derivativeField, isDerivativeInX  (boolean indicating which direction the derivative is in)
 * Output: technically none, but derivativeField is the 2D mesh (vector of vectors) in which the results are stored.
 * Dependencies: fftw, to_std_complex
 */
void MOLTEngine::computeSecondDerivative(std::complex<double>* inputField, 
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
