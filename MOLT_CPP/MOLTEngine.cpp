#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <complex.h>
#include <fftw3.h>
#include <vector>
#include <sys/time.h>

#include <unistd.h>
#include <stdio.h>
#include <omp.h>
#include "MOLTEngine.h"

#include "Derivative.h"
#include "FFT.h"
#include "FD6.h"

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
    struct timeval begin1, end1, begin2, end2, begin3, end3, begin4, end4, begin5, end5, begin6, end6;
    // std::cout << "Updating Particle Locations" << std::endl;
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
    if (correctTheGauge) {
        correctGauge();
    }
    gettimeofday( &begin5, NULL );
    computePhysicalDiagnostics();
    gettimeofday( &end5, NULL );
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
    std::cout << "computePhysicalDiagnostics(): " << timeComponent5 << std::endl;
    std::cout << "shuffleSteps(): " << timeComponent6 << std::endl;
    std::cout << "print(): " << timeComponent7 << std::endl;
}

double MOLTEngine::getTime() {
    return t;
}

int MOLTEngine::getStep() {
    return n;
}

double MOLTEngine::getForce() {
    return eleForce;
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

double MOLTEngine::getElecCharge() {
    return rhoElecTotal;
}

double MOLTEngine::getIonsCharge() {
    return rhoIonsTotal;
}

double MOLTEngine::getKineticEnergy() {
    return kineticEnergy;
}

double MOLTEngine::getPotentialEnergy() {
    return potentialEnergy;
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

double MOLTEngine::getMagneticMagnitude() {
    return magneticMagnitude;
}

double MOLTEngine::getTotalMomentum() {
    return totalMomentum;
}

void MOLTEngine::saveParticleInformation() {

    std::ofstream electronFile;

    std::string electronFileName = snapshotPath + "/particles.csv";
    electronFile.open(electronFileName, std::ios_base::app);

    electronFile << std::setprecision(16);
    electronFile << t << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << (*x_elec[lastStepIndex-1])[p] << ",";
    }
    electronFile << (*x_elec[lastStepIndex-1])[numElectrons-1] << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << (*y_elec[lastStepIndex-1])[p] << ",";
    }
    electronFile << (*y_elec[lastStepIndex-1])[numElectrons-1] << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << (*vx_elec[lastStepIndex-1])[p] << ",";
    }
    electronFile << (*vx_elec[lastStepIndex-1])[numElectrons-1] << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << (*vy_elec[lastStepIndex-1])[p] << ",";
    }
    electronFile << (*vy_elec[lastStepIndex-1])[numElectrons-1] << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << (*Px_elec[lastStepIndex-1])[p] << ",";
    }
    electronFile << (*Px_elec[lastStepIndex-1])[numElectrons-1] << std::endl;
    for (int p = 0; p < numElectrons-1; p++) {
        electronFile << (*Py_elec[lastStepIndex-1])[p] << ",";
    }
    electronFile << (*Py_elec[lastStepIndex-1])[numElectrons-1] << std::endl;

    electronFile << std::endl;
    electronFile.close();
}

void MOLTEngine::computePhysicalDiagnostics() {
    // std::cout << "computeTotalEnergy" << std::endl;
    computeTotalEnergy();
    // std::cout << "computeForce" << std::endl;
    computeForce();
    // std::cout << "computeTotalMass" << std::endl;
    computeTotalMass();
    // std::cout << "computeGaugeL2" << std::endl;
    computeGaugeL2();
    // std::cout << "computeGaussL2" << std::endl;
    computeGaussL2();
    // std::cout << "computeMagneticMagnitude" << std::endl;
    computeMagneticMagnitude();
    // std::cout << "computeTemperature" << std::endl;
    computeTemperature();
    // std::cout << "computeTotalMomentum" << std::endl;
    computeTotalMomentum();
    // std::cout << "Done computing physical diagnostics." << std::endl;
}

void MOLTEngine::computeTotalMomentum() {
    totalMomentum = 0;
    for (int i = 0; i < numElectrons; i++) {
        totalMomentum += std::sqrt( std::pow((*Px_elec[lastStepIndex])[i], 2) + std::pow((*Py_elec[lastStepIndex])[i], 2) );
    }
}

void MOLTEngine::computeMagneticMagnitude() {
    this->magneticMagnitude = 0.0;
    for (int i = 0; i < Nx*Ny; i++) {
        double B3 = ddx_A2[lastStepIndex][i].real() - ddy_A1[lastStepIndex][i].real();
        this->magneticMagnitude += B3*B3;
    }
    this->magneticMagnitude = std::sqrt(dx*dy*this->magneticMagnitude);
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

void MOLTEngine::computeForce() {
    /*
     * nonrelativistic: F = ma 
     * relativistic: F = dP/dt =  -q_i*grad(phi) + (q_ic^2(grad(A))dot(P_i - q_iA))/sqrt(c^2(P_i-q_iA)^2 + (m_ic^2)^2)
     * grad(A) = [[dA1/dx, dA1/dy], [dA2/dx, dA2,dy]]
     * => grad(A) dot u = [dA1/dx*u1 + dA1/dy*u2, dA2/dx*u1 + dA2/dy*u2]
     */
    double F = 0;
    #pragma omp parallel for reduction(+:F)
        for (int i = 0; i < numElectrons; i++) {
            double ddx_phi_p = 0;
            double ddy_phi_p = 0;
            double A1_p = 0;
            double A2_p = 0;
            double ddx_A1_p = 0;
            double ddx_A2_p = 0;
            double ddy_A1_p = 0;
            double ddy_A2_p = 0;

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

            const int ld = computeIndex(lc_x   , lc_y   ); // (left, down)  lc_x,   lc_y
            const int lu = computeIndex(lc_x   , lc_y_p1); // (left, up)    lc_x,   lc_y+1
            const int rd = computeIndex(lc_x_p1, lc_y   ); // (rite, down)  lc_x+1, lc_y
            const int ru = computeIndex(lc_x_p1, lc_y_p1); // (rite, up)    lc_x+1, lc_y+1

            const double xNode = this->x[lc_x];
            const double yNode = this->y[lc_y];

            // We compute the fractional distance of a particle from
            // the nearest node.
            // eg x=[0,.1,.2,.3], particleX = [.225]
            // The particle's fractional is 1/4
            const double fx = (p_x - xNode)/dx;
            const double fy = (p_y - yNode)/dy;

            const double w_ld = (1-fx)*(1-fy);
            const double w_lu = (1-fx)*(  fy);
            const double w_rd = (  fx)*(1-fy);
            const double w_ru = (  fx)*(  fy);

            if (this->updateMethod == MOLTEngine::CDF2) {
                ddx_phi_p += w_ld*( ddx_phi[lastStepIndex][ld].real() + ddx_phi[lastStepIndex-1][ld].real() ) / 2.0;
                ddx_phi_p += w_lu*( ddx_phi[lastStepIndex][lu].real() + ddx_phi[lastStepIndex-1][lu].real() ) / 2.0;
                ddx_phi_p += w_rd*( ddx_phi[lastStepIndex][rd].real() + ddx_phi[lastStepIndex-1][rd].real() ) / 2.0;
                ddx_phi_p += w_ru*( ddx_phi[lastStepIndex][ru].real() + ddx_phi[lastStepIndex-1][ru].real() ) / 2.0;

                ddy_phi_p += w_ld*( ddy_phi[lastStepIndex][ld].real() + ddy_phi[lastStepIndex-1][ld].real() ) / 2.0;
                ddy_phi_p += w_lu*( ddy_phi[lastStepIndex][lu].real() + ddy_phi[lastStepIndex-1][lu].real() ) / 2.0;
                ddy_phi_p += w_rd*( ddy_phi[lastStepIndex][rd].real() + ddy_phi[lastStepIndex-1][rd].real() ) / 2.0;
                ddy_phi_p += w_ru*( ddy_phi[lastStepIndex][ru].real() + ddy_phi[lastStepIndex-1][ru].real() ) / 2.0;
            } else {
                ddx_phi_p += w_ld*ddx_phi[lastStepIndex][ld].real();
                ddx_phi_p += w_lu*ddx_phi[lastStepIndex][lu].real();
                ddx_phi_p += w_rd*ddx_phi[lastStepIndex][rd].real();
                ddx_phi_p += w_ru*ddx_phi[lastStepIndex][ru].real();

                ddy_phi_p += w_ld*ddy_phi[lastStepIndex][ld].real();
                ddy_phi_p += w_lu*ddy_phi[lastStepIndex][lu].real();
                ddy_phi_p += w_rd*ddy_phi[lastStepIndex][rd].real();
                ddy_phi_p += w_ru*ddy_phi[lastStepIndex][ru].real();
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

            double P_minus_qA_1 = (P1_p - q_ele*A1_p);
            double P_minus_qA_2 = (P2_p - q_ele*A2_p);
            double kappa2 = kappa*kappa;

            double gradA_dot_P_minus_qA_1 = (ddx_A1_p*P_minus_qA_1 + ddy_A1_p*P_minus_qA_2);
            double gradA_dot_P_minus_qA_2 = (ddx_A2_p*P_minus_qA_1 + ddy_A2_p*P_minus_qA_2);

            double denom = std::sqrt(kappa2 * (P_minus_qA_1*P_minus_qA_1 + P_minus_qA_2*P_minus_qA_2) + std::pow(m_ele*kappa2, 2));

            double numerator_1 = q_ele*kappa2*gradA_dot_P_minus_qA_1;
            double numerator_2 = q_ele*kappa2*gradA_dot_P_minus_qA_2;

            double F1 = -q_ele*ddx_phi_p + numerator_1 / denom;
            double F2 = -q_ele*ddy_phi_p + numerator_2 / denom;

            F += std::sqrt(F1*F1 + F2*F2);
        }
        this->eleForce = F;

}

void MOLTEngine::computeTotalEnergy() {

    this->totalEnergy = 0;
    this->eleTotalEnergy = 0;
    this->ionTotalEnergy = 0;

    double T = 0;
    double P = 0;

    // Beware race condition
    #pragma omp parallel for reduction(+:T,P)
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

        const int ld = computeIndex(lc_x   , lc_y   ); // (left, down)  lc_x,   lc_y
        const int lu = computeIndex(lc_x   , lc_y_p1); // (left, up)    lc_x,   lc_y+1
        const int rd = computeIndex(lc_x_p1, lc_y   ); // (rite, down)  lc_x+1, lc_y
        const int ru = computeIndex(lc_x_p1, lc_y_p1); // (rite, up)    lc_x+1, lc_y+1

        const double xNode = this->x[lc_x];
        const double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        const double fx = (p_x - xNode)/dx;
        const double fy = (p_y - yNode)/dy;

        const double w_ld = (1-fx)*(1-fy);
        const double w_lu = (1-fx)*(  fy);
        const double w_rd = (  fx)*(1-fy);
        const double w_ru = (  fx)*(  fy);

        if (this->updateMethod == MOLTEngine::CDF2) {
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

        // double electron_energy  = std::pow(P1_p - q_ele*A1_p, 2);
        //        electron_energy += std::pow(P2_p - q_ele*A2_p, 2);
        //        electron_energy *= 1.0/(2.0*m_ele);
        //        electron_energy += q_ele*phi_p;
        // S += electron_energy;
        double kinetic_energy = std::sqrt(std::pow(kappa*P1_p - q_ele*A1_p, 2) + std::pow(kappa*P2_p - q_ele*A2_p, 2) + std::pow((m_ele * kappa*kappa), 2));
        double potential_energy = q_ele*phi_p;
        T += kinetic_energy;
        P += potential_energy;
    }
    this->eleKineticEnergy = T;
    this->elePotentialEnergy = P;
    this->eleTotalEnergy = T + P;

    T = 0;
    P = 0;

    #pragma omp parallel for reduction(+:T,P)
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

        const int ld = computeIndex(lc_x   , lc_y   ); // (left, down)  lc_x,   lc_y
        const int lu = computeIndex(lc_x   , lc_y_p1); // (left, up)    lc_x,   lc_y+1
        const int rd = computeIndex(lc_x_p1, lc_y   ); // (rite, down)  lc_x+1, lc_y
        const int ru = computeIndex(lc_x_p1, lc_y_p1); // (rite, up)    lc_x+1, lc_y+1

        const double xNode = this->x[lc_x];
        const double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        const double fx = (p_x - xNode)/dx;
        const double fy = (p_y - yNode)/dy;

        const double w_ld = (1-fx)*(1-fy);
        const double w_lu = (1-fx)*(  fy);
        const double w_rd = (  fx)*(1-fy);
        const double w_ru = (  fx)*(  fy);

        if (this->updateMethod == MOLTEngine::CDF2) {
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

        // double ion_energy  = std::pow(P1_p - q_ion*A1_p, 2);
        //        ion_energy += std::pow(P2_p - q_ion*A2_p, 2);
        //        ion_energy *= 1.0/(2.0*m_ion);
        //        ion_energy += q_ion*phi_p;
        // S += ion_energy;
        double kinetic_energy = std::sqrt(std::pow(kappa*P1_p - q_ion*A1_p, 2) + std::pow(kappa*P2_p - q_ion*A2_p, 2) + std::pow((m_ion * kappa*kappa), 2));
        double potential_energy = q_ion*phi_p;
        T += kinetic_energy;
        P += potential_energy;
    }
    this->ionKineticEnergy = T;
    this->ionPotentialEnergy = P;
    this->ionTotalEnergy = T + P;

    this->potentialEnergy = ionPotentialEnergy + elePotentialEnergy;
    this->kineticEnergy = ionKineticEnergy + eleKineticEnergy;
    this->totalEnergy = ionTotalEnergy + eleTotalEnergy;

}

void MOLTEngine::computeTotalMass() {
    // Total charge is the charge density in each cell times the cell volume, which
    // followed by a sum over the cells
    std::vector<double> ele_charge(Nx*Ny);
    for (int i = 0; i < Nx*Ny; i++) {
        ele_charge[i] = dx*dy*rho_eles[i].real();
    }
    
    // Total mass is the total charge divided by charge per particle times the mass of each particle    
    this->eleTotalMass = 0;
    for (int i = 0; i < Nx*Ny; i++) {
        this->eleTotalMass += m_ele*(ele_charge[i]/q_ele);
    }

    std::vector<double> ion_charge(Nx*Ny);
    for (int i = 0; i < Nx*Ny; i++) {
        ion_charge[i] = dx*dy*rho_ions[i].real();
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
        if (this->updateMethod == MOLTEngine::DIRK2 || this->updateMethod == MOLTEngine::DIRK3) {

            double b1;
            double b2;

            double c1;
            double c2;

            if (this->updateMethod == MOLTEngine::DIRK2) {
                // Qin and Zhang's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/4.0;
                c2 = 3.0/4.0;
            } else if (this->updateMethod == MOLTEngine::DIRK3) {
                // Crouzeix's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/2.0 + std::sqrt(3.0)/6.0;
                c2 = 1.0/2.0 - std::sqrt(3.0)/6.0;
            }

            ddt_phi = (phi[lastStepIndex][i].real() - phi[lastStepIndex-1][i].real()) / dt;

            double div_A_prev = ddx_A1[lastStepIndex-1][i].real() + ddy_A2[lastStepIndex-1][i].real();
            double div_A_curr = ddx_A1[lastStepIndex][i].real() + ddy_A2[lastStepIndex][i].real();

            double RHS_1 = (1-c1)*div_A_prev + c1*div_A_curr;
            double RHS_2 = (1-c2)*div_A_prev + c2*div_A_curr;

            div_A = b1*RHS_1 + b2*RHS_2;

            l2 += std::pow(1.0/(kappa*kappa)*ddt_phi + div_A,2);

        } else {

            if (this->updateMethod == MOLTEngine::BDF1 ||
                this->updateMethod == MOLTEngine::CDF2) {

                this->ddt_phi[1][i] = (phi[lastStepIndex][i].real() - phi[lastStepIndex-1][i].real()) / dt;

                ddt_phi = (phi[lastStepIndex][i].real() - phi[lastStepIndex-1][i].real()) / dt;
                div_A = ddx_A1[lastStepIndex][i].real() + ddy_A2[lastStepIndex][i].real();

            } else if (this->updateMethod == MOLTEngine::BDF2) {

                this->ddt_phi[1][i] = (phi[lastStepIndex][i].real() - (4.0/3.0)*phi[lastStepIndex-1][i].real() + (1.0/3.0)*phi[lastStepIndex-2][i].real()) / (dt*(2.0/3.0));

                ddt_phi = (phi[lastStepIndex][i].real() - (4.0/3.0)*phi[lastStepIndex-1][i].real() + (1.0/3.0)*phi[lastStepIndex-2][i].real()) / (dt*(2.0/3.0));
                div_A = ddx_A1[lastStepIndex][i].real() + ddy_A2[lastStepIndex][i].real();

            } else {
                throw -1;
            }
            l2 += std::pow(1.0/(kappa*kappa)*ddt_phi + div_A,2);
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
        if (this->updateMethod == MOLTEngine::BDF1 ||
            this->updateMethod == MOLTEngine::CDF2 ||
            this->updateMethod == MOLTEngine::DIRK2) {
            ddt_A1 = (A1[lastStepIndex][i].real() - A1[lastStepIndex-1][i].real()) / dt;
            ddt_A2 = (A2[lastStepIndex][i].real() - A2[lastStepIndex-1][i].real()) / dt;
        } else if (this->updateMethod == MOLTEngine::BDF2) {
            ddt_A1 = (A1[lastStepIndex][i].real() - (4.0/3.0)*A1[lastStepIndex-1][i].real() + (1.0/3.0)*A1[lastStepIndex-2][i].real()) / ((2.0/3.0)*dt);
            ddt_A2 = (A2[lastStepIndex][i].real() - (4.0/3.0)*A2[lastStepIndex-1][i].real() + (1.0/3.0)*A2[lastStepIndex-2][i].real()) / ((2.0/3.0)*dt);
        } else {
            throw -1;
        }
        if (this->updateMethod == MOLTEngine::CDF2) {
            E1[i] = -(ddx_phi[lastStepIndex][i].real() + ddx_phi[lastStepIndex-2][i].real())/2.0 - ddt_A1;
            E2[i] = -(ddy_phi[lastStepIndex][i].real() + ddy_phi[lastStepIndex-2][i].real())/2.0 - ddt_A2;
        } else {
            E1[i] = -ddx_phi[lastStepIndex][i].real() - ddt_A1;
            E2[i] = -ddy_phi[lastStepIndex][i].real() - ddt_A2;
        }
    }

    // Extremely clunky, need to fix.
    if (this->derivative_utility == nullptr) {

        double Lx = x[Nx-1] - x[0] + dx;
        double Ly = y[Nx-1] - y[0] + dy;

        this->derivative_utility = new FD6(Nx, Ny, Lx, Ly);

        compute_ddx_numerical(E1, ddx_E1);
        compute_ddy_numerical(E2, ddy_E2);

        compute_d2dx_numerical(phi[lastStepIndex], d2dx_phi_curr);
        compute_d2dy_numerical(phi[lastStepIndex], d2dy_phi_curr);

        delete this->derivative_utility;
        this->derivative_utility = nullptr;

    } else {

        compute_ddx_numerical(E1, ddx_E1);
        compute_ddy_numerical(E2, ddy_E2);

        compute_d2dx_numerical(phi[lastStepIndex], d2dx_phi_curr);
        compute_d2dy_numerical(phi[lastStepIndex], d2dy_phi_curr);

    }

    std::vector<double> div_E(Nx*Ny);

    for (int i = 0; i < Nx*Ny; i++) {
        div_E[i] = ddx_E1[i].real() + ddy_E2[i].real();
    }

    if (this->updateMethod == MOLTEngine::DIRK2) {

        compute_d2dx_numerical(phi[lastStepIndex-1], d2dx_phi_prev);
        compute_d2dy_numerical(phi[lastStepIndex-1], d2dy_phi_prev);

        for (int i = 0; i < Nx*Ny; i++) {
            double rho_prev = rho[lastStepIndex-1][i].real();
            double rho_curr = rho[lastStepIndex  ][i].real();
            gauss_RHS[i] = dirk_qin_zhang_rhs(rho_prev, rho_curr) / sigma_1;

            double laplacian_phi_prev = d2dx_phi_prev[i].real() + d2dx_phi_prev[i].real();
            double laplacian_phi_curr = d2dx_phi_curr[i].real() + d2dy_phi_curr[i].real();
            laplacian_phi[i] = dirk_qin_zhang_rhs(laplacian_phi_prev, laplacian_phi_curr);

            double div_A_prev = ddx_A1[lastStepIndex-1][i].real() + ddy_A2[lastStepIndex-1][i].real();
            double div_A_curr = ddx_A1[lastStepIndex  ][i].real() + ddy_A2[lastStepIndex  ][i].real();
            ddt_divA_curr[i] = (div_A_curr - div_A_prev) / dt;
        }
    } else {

        if (this->updateMethod == MOLTEngine::CDF2) {
            for (int i = 0; i < Nx*Ny; i++) {
                gauss_RHS[i] = (rho[lastStepIndex][i].real() + rho[lastStepIndex-2][i].real()) / (2.0*sigma_1);
            }
        } else {
            for (int i = 0; i < Nx*Ny; i++) {
                gauss_RHS[i] = rho[lastStepIndex][i].real() / sigma_1;
            }
        }

        if (this->updateMethod == MOLTEngine::CDF2) {

            compute_d2dx_numerical(phi[lastStepIndex-2], d2dx_phi_prev);
            compute_d2dy_numerical(phi[lastStepIndex-2], d2dy_phi_prev);

            for (int i = 0; i < Nx*Ny; i++) {
                laplacian_phi[i] = (d2dx_phi_curr[i] + d2dx_phi_prev[i] + d2dy_phi_curr[i] + d2dy_phi_prev[i]) / 2.0;
            }
        } else {
            for (int i = 0; i < Nx*Ny; i++) {
                laplacian_phi[i] = d2dx_phi_curr[i] + d2dy_phi_curr[i];
            }
        }

        for (int i = 0; i < Nx*Ny; i++) {
            if (this->updateMethod == MOLTEngine::BDF1 ||
                this->updateMethod == MOLTEngine::DIRK2 ||
                this->updateMethod == MOLTEngine::CDF2) {
                d2dt_phi[i] = (phi[lastStepIndex][i] - 2.0*phi[lastStepIndex-1][i] + phi[lastStepIndex-2][i]).real()/(dt*dt);
            } else if (this->updateMethod == MOLTEngine::BDF2) {
                d2dt_phi[i] = (phi[lastStepIndex][i] - 8.0/3.0*phi[lastStepIndex-1][i] + 22.0/9.0*phi[lastStepIndex-2][i] - 8.0/9.0*phi[lastStepIndex-3][i] + 1.0/9.0*phi[lastStepIndex-4][i]).real() / std::pow(2.0/3.0*dt, 2);
            } else {
                throw -1;
            }
        }

        for (int i = 0; i < Nx*Ny; i++) {
            double div_A_curr = ddx_A1[lastStepIndex  ][i].real() + ddy_A2[lastStepIndex  ][i].real();
            double div_A_prev = ddx_A1[lastStepIndex-1][i].real() + ddy_A2[lastStepIndex-1][i].real();
            double div_A_nm1  = ddx_A1[lastStepIndex-2][i].real() + ddy_A2[lastStepIndex-2][i].real();
            if (this->updateMethod == MOLTEngine::BDF1 ||
                this->updateMethod == MOLTEngine::DIRK2 ||
                this->updateMethod == MOLTEngine::CDF2) {
                    ddt_divA_curr[i] = (div_A_curr - div_A_prev) / dt;
                } else if (this->updateMethod == MOLTEngine::BDF2) {
                    ddt_divA_curr[i] = ( div_A_curr - 4.0/3.0*div_A_prev + 1.0/3.0*div_A_nm1 ) / ((2.0/3.0)*dt);
                } else {
                    throw -1;
                }
        }
    }

    double l2 = 0;
    // res = div(E) - rho/sigma_1 (sigma_1 nondimensionalized eps_0)
    for (int i = 0; i < Nx*Ny; i++) {
        l2 += std::pow(div_E[i] - gauss_RHS[i], 2);
    }
    gaussL2_divE = std::sqrt(dx*dy*l2);

    l2 = 0;
    // res = -d/dt[div(A)] - laplacian_phi - rho/sigma_1 (sigma_1 nondimensionalized eps_0)
    for (int i = 0; i < Nx*Ny; i++) {
        l2 += std::pow(-ddt_divA_curr[i].real() - laplacian_phi[i].real() - gauss_RHS[i], 2);
    }
    gaussL2_divA = std::sqrt(dx*dy*l2);

    l2 = 0;
    // res = d^2phi/dt^2 - laplacian_phi - rho/sigma_1 (sigma_1 nondimensionalized eps_0)
    for (int i = 0; i < Nx*Ny; i++) {
        l2 += std::pow(1.0/(kappa*kappa) * d2dt_phi[i].real() - laplacian_phi[i].real() - gauss_RHS[i], 2);
    }
    gaussL2_wave = std::sqrt(dx*dy*l2);

}

/**
 * Name: printFieldData
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 10/17/24 (Stephen White)
 * Description: This prints both field and particle data
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::printAllData() {
    struct timeval printBegin, printEnd;
    gettimeofday( &printBegin, NULL );

    printFieldData();
    printParticleData();

    gettimeofday( &printEnd, NULL );
    timeComponent7 += 1.0 * ( printEnd.tv_sec - printBegin.tv_sec ) + 1.0e-6 * ( printEnd.tv_usec - printBegin.tv_usec );
}

/**
 * Name: printFieldData
 * Author: Stephen White
 * Date Created: 10/17/24
 * Date Last Modified: 10/17/24 (Stephen White)
 * Description: This prints the potentials and particle information to their own files grouped by mesh refinement, labeled by field and timestep
 * Inputs: none (relies on the field and particle arrays)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::printFieldData() {

    std::ofstream phiFile, A1File, A2File;
    std::ofstream ddx_phiFile, ddx_A1File, ddx_A2File;
    std::ofstream ddy_phiFile, ddy_A1File, ddy_A2File;
    std::ofstream rhoFile, J1File, J2File;
    std::ofstream ddt_phiFile;

    std::string nstr = std::to_string(n);
    int numlen = 5;

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
    
    phiFile      << t << std::endl;
    A1File       << t << std::endl;
    A2File       << t << std::endl;
    ddx_phiFile  << t << std::endl;
    ddx_A1File   << t << std::endl;
    ddx_A2File   << t << std::endl;
    ddy_phiFile  << t << std::endl;
    ddy_A1File   << t << std::endl;
    ddy_A2File   << t << std::endl;
    rhoFile      << t << std::endl;
    J1File       << t << std::endl;
    J2File       << t << std::endl;
    ddt_phiFile  << t << std::endl;

    /*
     * The following values are all at lastStepIndex-1
     * due to the assumption that print() will be called by
     * an outside runner method, meaning the shuffleSteps()
     * method will have been called, meaning the lastStepIndex
     * slot will be junk.
     */
    for (int i = 0; i < Nx*Ny - 1; i++) {
        phiFile     << phi[lastStepIndex-1][i].real() << ",";
        A1File      << A1[lastStepIndex-1][i].real() << ",";
        A2File      << A2[lastStepIndex-1][i].real() << ",";
        ddx_phiFile << ddx_phi[lastStepIndex-1][i].real() << ",";
        ddx_A1File  << ddx_A1[lastStepIndex-1][i].real() << ",";
        ddx_A2File  << ddx_A2[lastStepIndex-1][i].real() << ",";
        ddy_phiFile << ddy_phi[lastStepIndex-1][i].real() << ",";
        ddy_A1File  << ddy_A1[lastStepIndex-1][i].real() << ",";
        ddy_A2File  << ddy_A2[lastStepIndex-1][i].real() << ",";
        rhoFile     << rho[lastStepIndex-1][i].real() << ",";
        J1File      << J1[lastStepIndex-1][i].real() << ",";
        J2File      << J2[lastStepIndex-1][i].real() << ",";
        ddt_phiFile << ddt_phi[1][i].real() << ",";
    }
    phiFile     << phi[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    A1File      << A1[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    A2File      << A2[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    ddx_phiFile << ddx_phi[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    ddx_A1File  << ddx_A1[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    ddx_A2File  << ddx_A2[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    ddy_phiFile << ddy_phi[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    ddy_A1File  << ddy_A1[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    ddy_A2File  << ddy_A2[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    rhoFile     << rho[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    J1File      << J1[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    J2File      << J2[lastStepIndex-1][Nx*Ny-1].real() << std::endl;
    ddt_phiFile << ddt_phi[1][Nx*Ny-1].real() << std::endl;

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

void MOLTEngine::printParticleData() {
    

    std::ofstream particleFile;

    std::string nstr = std::to_string(n);
    int numlen = 5;

    std::ostringstream padder;
    padder << std::internal << std::setfill('0') << std::setw(numlen) << n;
    std::string paddedNum = padder.str();

    std::string particleFileName = snapshotPath + "/particle_data/particle_" + paddedNum + ".csv";

    std::cout << particleFileName << std::endl;

    particleFile.open(particleFileName);

    particleFile << std::setprecision(16);
    for (int i = 0; i < numElectrons-1; i++) {
        particleFile << (*x_elec[lastStepIndex-1])[i] << "," << (*y_elec[lastStepIndex-1])[i] << ","
                     << (*vx_elec[lastStepIndex-1])[i] << "," << (*vy_elec[lastStepIndex-1])[i] << "\n";
    }
    particleFile << (*x_elec[lastStepIndex-1])[numElectrons-1] << "," << (*y_elec[lastStepIndex-1])[numElectrons-1] << ","
                 << (*vx_elec[lastStepIndex-1])[numElectrons-1] << "," << (*vy_elec[lastStepIndex-1])[numElectrons-1];

    particleFile.close();

}


/**
 * Name: correctGauge
 * Author: Stephen White
 * Date Created: 7/24/24
 * Date Last Modified: 7/24/24 (Stephen White)
 * Description: Corrects phi to match the Lorenz gauge condition, updates derivatives
 * using FFT or FD6
 * Inputs: none (relies on phi, A1, and A2)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::correctGauge() {

    std::vector<std::complex<double>> phi_C(Nx*Ny);

    for (int i = 0; i < Nx*Ny; i++) {
        phi_C[i] = phi[lastStepIndex-1][i] - phi[lastStepIndex][i] - kappa*kappa*dt*(ddx_A1[lastStepIndex][i] + ddy_A2[lastStepIndex][i]);
    }

    for (int i = 0; i < Nx*Ny; i++) {
        phi[lastStepIndex][i] += phi_C[i];
    }

    // Extremely clunky, need to fix.
    if (this->derivative_utility == nullptr) {

        double Lx = x[Nx-1] - x[0] + dx;
        double Ly = y[Nx-1] - y[0] + dy;

        this->derivative_utility = new FD6(Nx, Ny, Lx, Ly);

        compute_ddx_numerical(phi[lastStepIndex], ddx_phi[lastStepIndex]);
        compute_ddy_numerical(phi[lastStepIndex], ddy_phi[lastStepIndex]);

        delete this->derivative_utility;
        this->derivative_utility = nullptr;

    } else {
        compute_ddx_numerical(phi[lastStepIndex], ddx_phi[lastStepIndex]);
        compute_ddy_numerical(phi[lastStepIndex], ddy_phi[lastStepIndex]);
    }
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
        // double vx_star = 2.0*(*vx_elec[lastStepIndex-1])[i] - (*vx_elec[lastStepIndex-2])[i];
        // double vy_star = 2.0*(*vy_elec[lastStepIndex-1])[i] - (*vy_elec[lastStepIndex-2])[i];

        (*x_elec[lastStepIndex])[i] = (*x_elec[lastStepIndex-1])[i] + dt*(*vx_elec[lastStepIndex-1])[i];
        (*y_elec[lastStepIndex])[i] = (*y_elec[lastStepIndex-1])[i] + dt*(*vy_elec[lastStepIndex-1])[i];
        // (*x_elec[lastStepIndex])[i] = (*x_elec[lastStepIndex-1])[i] + dt*vx_star;
        // (*y_elec[lastStepIndex])[i] = (*y_elec[lastStepIndex-1])[i] + dt*vy_star;

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

        compute_d2dx_numerical(u, d2dx_u);
        compute_d2dy_numerical(u, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u[i] = d2dx_u[i] + d2dy_u[i];
        }

        for (int i = 0; i < Nx*Ny; i++) {
            RHS1[i] = v[i] + dt*a11*kappa*kappa*(laplacian_u[i] + S_1[i]);
        }

        solveHelmholtzEquation(RHS1, u1, alpha_1);

        compute_d2dx_numerical(u1, d2dx_u);
        compute_d2dy_numerical(u1, d2dy_u);

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

        compute_d2dx_numerical(u2, d2dx_u);
        compute_d2dy_numerical(u2, d2dy_u);

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

        compute_d2dx_numerical(u, d2dx_u);
        compute_d2dy_numerical(u, d2dy_u);

        for (int i = 0; i < Nx*Ny; i++) {
            laplacian_u[i] = d2dx_u[i] + d2dy_u[i];
        }

        for (int i = 0; i < Nx*Ny; i++) {
            RHS1[i] = v[i] + dt*a11*kappa*kappa*(laplacian_u[i] + S_1[i]);
        }

        solveHelmholtzEquation(RHS1, u1, alpha_1);

        compute_d2dx_numerical(u1, d2dx_u);
        compute_d2dy_numerical(u1, d2dy_u);

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

        compute_d2dx_numerical(u2, d2dx_u);
        compute_d2dy_numerical(u2, d2dy_u);

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

    if (this->moltMethod == MOLTMethod::Integral) {
        std::complex<double>* phi_RHS = new std::complex<double>[Nx*Ny];
        std::complex<double>*  A1_RHS = new std::complex<double>[Nx*Ny];
        std::complex<double>*  A2_RHS = new std::complex<double>[Nx*Ny];

        for (int i = 0; i < Nx*Ny; i++) {
            phi_RHS[i] = 1.0/sigma_1 * rho[lastStepIndex][i];
            A1_RHS[i] =      sigma_2 *  J1[lastStepIndex][i];
            A2_RHS[i] =      sigma_2 *  J2[lastStepIndex][i];
        }

        MOLT_combined_per_advance(phi, phi_RHS, phi[lastStepIndex], ddx_phi[lastStepIndex], ddy_phi[lastStepIndex]);
        MOLT_combined_per_advance( A1,  A1_RHS,  A1[lastStepIndex],  ddx_A1[lastStepIndex],  ddy_A1[lastStepIndex]);
        MOLT_combined_per_advance( A2,  A2_RHS,  A2[lastStepIndex],  ddx_A2[lastStepIndex],  ddy_A2[lastStepIndex]);

        delete[] phi_RHS;
        delete[] A1_RHS;
        delete[] A2_RHS;
    } else {
        if (this->updateMethod == MOLTEngine::DIRK2 || this->updateMethod == MOLTEngine::DIRK3) {

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

            if (this->updateMethod == MOLTEngine::DIRK2) {
                DIRK2_advance_per(phi[lastStepIndex-1], ddt_phi_curr, phi[lastStepIndex], ddt_phi[1], phi_src_prev, phi_src);
                DIRK2_advance_per(A1[lastStepIndex-1], ddt_A1_curr, A1[lastStepIndex], ddt_A1[1], A1_src_prev, A1_src);
                DIRK2_advance_per(A2[lastStepIndex-1], ddt_A2_curr, A2[lastStepIndex], ddt_A2[1], A2_src_prev, A2_src);
            } else if (this->updateMethod == MOLTEngine::DIRK3) {
                DIRK3_advance_per(phi[lastStepIndex-1], ddt_phi_curr, phi[lastStepIndex], ddt_phi[1], phi_src_prev, phi_src);
                DIRK3_advance_per(A1[lastStepIndex-1], ddt_A1_curr, A1[lastStepIndex], ddt_A1[1], A1_src_prev, A1_src);
                DIRK3_advance_per(A2[lastStepIndex-1], ddt_A2_curr, A2[lastStepIndex], ddt_A2[1], A2_src_prev, A2_src);
            }

        } else {
            if (this->updateMethod == MOLTEngine::BDF1) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi_src[i] = 2.0*phi[lastStepIndex-1][i] - phi[lastStepIndex-2][i] + 1.0/alpha2 * 1.0/sigma_1 * rho[lastStepIndex][i];
                    A1_src[i] = 2.0*A1[lastStepIndex-1][i] - A1[lastStepIndex-2][i] + 1.0/alpha2 * sigma_2 * J1[lastStepIndex][i];
                    A2_src[i] = 2.0*A2[lastStepIndex-1][i] - A2[lastStepIndex-2][i] + 1.0/alpha2 * sigma_2 * J2[lastStepIndex][i];
                }
            } else if (this->updateMethod == MOLTEngine::BDF2) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi_src[i] = 8.0/3.0*phi[lastStepIndex-1][i] - 22.0/9.0*phi[lastStepIndex-2][i] + 8.0/9.0*phi[lastStepIndex-3][i] - 1.0/9.0*phi[lastStepIndex-4][i] + 1.0/alpha2 * 1.0/sigma_1 * rho[lastStepIndex][i];
                    A1_src[i] = 8.0/3.0*A1[lastStepIndex-1][i] - 22.0/9.0*A1[lastStepIndex-2][i] + 8.0/9.0*A1[lastStepIndex-3][i] - 1.0/9.0*A1[lastStepIndex-4][i] + 1.0/alpha2 * sigma_2 * J1[lastStepIndex][i];
                    A2_src[i] = 8.0/3.0*A2[lastStepIndex-1][i] - 22.0/9.0*A2[lastStepIndex-2][i] + 8.0/9.0*A2[lastStepIndex-3][i] - 1.0/9.0*A2[lastStepIndex-4][i] + 1.0/alpha2 * sigma_2 * J2[lastStepIndex][i];
                }
            } else if (this->updateMethod == MOLTEngine::CDF2) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi_src[i] = 1.0/alpha2 * (rho[lastStepIndex][i] + rho[lastStepIndex-2][i]) / (sigma_1) + 2.0*phi[lastStepIndex-1][i];
                    A1_src[i]  = sigma_2/alpha2 * ( J1[lastStepIndex][i] + J1[lastStepIndex-2][i] )       + 2.0* A1[lastStepIndex-1][i];
                    A2_src[i]  = sigma_2/alpha2 * ( J2[lastStepIndex][i] + J2[lastStepIndex-2][i] )       + 2.0* A2[lastStepIndex-1][i];
                }
            } else {
                throw -1;
            }

            solveHelmholtzEquation(phi_src, phi[lastStepIndex], alpha);
            solveHelmholtzEquation( A1_src,  A1[lastStepIndex], alpha);
            solveHelmholtzEquation( A2_src,  A2[lastStepIndex], alpha);

            if (this->updateMethod == MOLTEngine::CDF2) {
                for (int i = 0; i < Nx*Ny; i++) {
                    phi[lastStepIndex][i] -= phi[lastStepIndex-2][i];
                    A1[lastStepIndex][i]  -= A1[lastStepIndex-2][i];
                    A2[lastStepIndex][i]  -= A2[lastStepIndex-2][i];
                }
            }
        }

        compute_ddx_numerical(phi[lastStepIndex], ddx_phi[lastStepIndex]);
        compute_ddy_numerical(phi[lastStepIndex], ddy_phi[lastStepIndex]);
        compute_ddx_numerical(A1[lastStepIndex],  ddx_A1[lastStepIndex]);
        compute_ddy_numerical(A1[lastStepIndex],  ddy_A1[lastStepIndex]);
        compute_ddx_numerical(A2[lastStepIndex],  ddx_A2[lastStepIndex]);
        compute_ddy_numerical(A2[lastStepIndex],  ddy_A2[lastStepIndex]);
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

    int numFields = 8;

    std::complex<double>* ddx_phi_ave = new std::complex<double>[Nx*Ny];
    std::complex<double>* ddy_phi_ave = new std::complex<double>[Nx*Ny];

    if (this->updateMethod == MOLTEngine::CDF2) {
        for (int i = 0; i < Nx*Ny; i++) {
            ddx_phi_ave[i] = (this->ddx_phi[lastStepIndex][i] + this->ddx_phi[lastStepIndex-1][i]) / 2.0;
            ddy_phi_ave[i] = (this->ddy_phi[lastStepIndex][i] + this->ddy_phi[lastStepIndex-1][i]) / 2.0;
        }
    } else {
        for (int i = 0; i < Nx*Ny; i++) {
            ddx_phi_ave[i] = this->ddx_phi[lastStepIndex][i];
            ddy_phi_ave[i] = this->ddy_phi[lastStepIndex][i];
        }
    }

    std::complex<double>* fields[numFields] = {ddx_phi_ave, ddy_phi_ave, A1[lastStepIndex], ddx_A1[lastStepIndex], ddy_A1[lastStepIndex], A2[lastStepIndex], ddx_A2[lastStepIndex], ddy_A2[lastStepIndex]};
    std::vector<std::vector<std::complex<double>>> vals(numFields, std::vector<std::complex<double>>(numElectrons));

    this->interpolate_utility->gatherFields(fields, (*x_elec[lastStepIndex]), (*y_elec[lastStepIndex]), numFields, numElectrons, vals);

    delete[] ddx_phi_ave;
    delete[] ddy_phi_ave;

    #pragma omp parallel for
    for (int i = 0; i < numElectrons; i++) {
        double ddx_phi_p = vals[0][i].real();
        double ddy_phi_p = vals[1][i].real();
        double A1_p = vals[2][i].real();
        double ddx_A1_p = vals[3][i].real();
        double ddy_A1_p = vals[4][i].real();
        double A2_p = vals[5][i].real();
        double ddx_A2_p = vals[6][i].real();
        double ddy_A2_p = vals[7][i].real();

        double vx_star = (*vx_elec[lastStepIndex-1])[i] + ( (*vx_elec[lastStepIndex-1])[i] - (*vx_elec[lastStepIndex-2])[i] );
        double vy_star = (*vy_elec[lastStepIndex-1])[i] + ( (*vy_elec[lastStepIndex-1])[i] - (*vy_elec[lastStepIndex-2])[i] );

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
    std::vector<std::vector<double>> weights(2,std::vector<double>(numElectrons));
    for (int p = 0; p < numElectrons; p++) {
        weights[0][p] = q_ele*(*vx_elec[lastStepIndex-1])[p]*w_ele/(dx*dy);
        weights[1][p] = q_ele*(*vy_elec[lastStepIndex-1])[p]*w_ele/(dx*dy);
    }
    std::complex<double>* fields[2] = {J1[lastStepIndex], J2[lastStepIndex]};

    this->interpolate_utility->scatterParticles(fields, (*x_elec[lastStepIndex]), (*y_elec[lastStepIndex]), 2, numElectrons, weights);

    if (this->rhoUpdate == MOLTEngine::CONSERVING) {

        // Compute div J
        compute_ddx_numerical(J1[lastStepIndex], ddx_J1);
        compute_ddy_numerical(J2[lastStepIndex], ddy_J2);

        // double Gamma = 0;

        // for (int i = 0; i < Nx*Ny; i++) {
        //     Gamma += ddx_J1[i].real() + ddy_J2[i].real();
        // }
        // Gamma *= -1.0/(Nx*Ny);

        // int idx;

        // for (int i = 0; i < Nx; i++) {
        //     for (int j = 0; j < Ny; j++) {
        //         idx = computeIndex(i, j);
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

        if (this->updateMethod == MOLTEngine::BDF1 ||
            this->updateMethod == MOLTEngine::CDF2) {
            for (int i = 0; i < Nx*Ny; i++) {
                rho[lastStepIndex][i] = rho[lastStepIndex-1][i] - dt*(ddx_J1[i] + ddy_J2[i]);
            }
        } else if (this->updateMethod == MOLTEngine::BDF2) {
            for (int i = 0; i < Nx*Ny; i++) {
                rho[lastStepIndex][i] = 4.0/3.0*rho[lastStepIndex-1][i] - 1.0/3.0*rho[lastStepIndex-2][i] - ((2.0/3.0)*dt)*(ddx_J1[i] + ddy_J2[i]);
            }
        } else if (this->updateMethod == MOLTEngine::DIRK2 || this->updateMethod == MOLTEngine::DIRK3) {

            double b1;
            double b2;

            double c1;
            double c2;

            if (this->updateMethod == MOLTEngine::DIRK2) {
                // Qin and Zhang's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/4.0;
                c2 = 3.0/4.0;
            } else if (this->updateMethod == MOLTEngine::DIRK3) {
                // Crouzeix's update scheme
                b1 = 1.0/2.0;
                b2 = 1.0/2.0;

                c1 = 1.0/2.0 + std::sqrt(3.0)/6.0;
                c2 = 1.0/2.0 - std::sqrt(3.0)/6.0;
            }

            compute_ddx_numerical(J1[lastStepIndex-1], ddx_J1_prev);
            compute_ddy_numerical(J2[lastStepIndex-1], ddy_J2_prev);

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
            rho_eles[i] = 0.0;
        }

        double charge_value = q_ele*w_ele/(dx*dy);

        double S [Nx*Ny] = {0};
        #pragma omp parallel for reduction(+:S[:Nx*Ny])
        for (int i = 0; i < numElectrons; i++) {

            double x_p = (*x_elec[lastStepIndex])[i];
            double y_p = (*y_elec[lastStepIndex])[i];

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

            S[ld] += (1-fx)*(1-fy)*charge_value;
            S[lu] += (1-fx)*(  fy)*charge_value;
            S[rd] += (  fx)*(1-fy)*charge_value;
            S[ru] += (  fx)*(  fy)*charge_value;
        }
        // for (int i = 0; i < Nx*Ny; i++) {
        //     rhoElecTotal += rho_eles[i].real();
        //     rhoIonsTotal += rho_eles[i].real();
        // }
        // rhoTotal = rhoElecTotal + rhoIonsTotal;

        // double volume = 1.0/(dx*dy);
        for (int i = 0; i < Nx*Ny; i++) {
            rho_eles[i] = S[i];
            // rho_eles[i] = S[i]*volume;
            rho[lastStepIndex][i] = rho_eles[i] + rho_ions[i];
        }
    }
    this->rhoTotal = 0.0;
    this->rhoElecTotal = 0.0;
    this->rhoIonsTotal = 0.0;
    // for (int i = 0; i < Nx*Ny; i++) {
    //     this->rhoTotal += rho[lastStepIndex][i].real();
    // }
    for (int i = 0; i < Nx*Ny; i++) {
        rhoElecTotal += rho_eles[i].real();
        rhoIonsTotal += rho_ions[i].real();
    }
    rhoTotal = rhoElecTotal + rhoIonsTotal;
}

void MOLTEngine::MOLT_BDF1_combined_per_advance(std::vector<std::complex<double>*> u, std::complex<double>* RHS,
                                                std::complex<double>* u_out, std::complex<double>* dudx_out, std::complex<double>* dudy_out) {

    MOLT_BDF1_advance_per(u, RHS, u_out);

    if (this->derivative_utility == nullptr) {
        MOLT_BDF1_ddx_advance_per(u, RHS, dudx_out);
        MOLT_BDF1_ddy_advance_per(u, RHS, dudy_out);
    } else {
        compute_ddx_numerical(u[lastStepIndex], dudx_out);
        compute_ddy_numerical(u[lastStepIndex], dudy_out);
    }
}

void MOLTEngine::MOLT_combined_per_advance(std::vector<std::complex<double>*> u, std::complex<double>* RHS,
                                           std::complex<double>* u_out, std::complex<double>* dudx_out, std::complex<double>* dudy_out) {
    
    const double alpha = beta/(kappa*dt);
    const double alpha2 = alpha*alpha;

    std::complex<double>* src = new std::complex<double>[Nx*Ny];

    if (this->updateMethod == MOLTEngine::BDF1) {
        for (int i = 0; i < Nx*Ny; i++) {
            src[i] = 2.0*u[lastStepIndex-1][i] - u[lastStepIndex-2][i] + 1.0/alpha2*RHS[i];
        }
    } else if (this->updateMethod == MOLTEngine::BDF2) {
        for (int i = 0; i < Nx*Ny; i++) {
            src[i] = 8.0/3.0*u[lastStepIndex-1][i] - 22.0/9.0*u[lastStepIndex-2][i] + 8.0/9.0*u[lastStepIndex-3][i] - 1.0/9.0*u[lastStepIndex-4][i] + 1.0/alpha2*RHS[i];
        }        
    } else if (this->updateMethod == MOLTEngine::CDF2) {
        for (int i = 0; i < Nx*Ny; i++) {
            src[i] = 2.0*u[lastStepIndex-1][i] + 1.0/alpha2*RHS[i];
        }
    }
    MOLT_advance_per(src, u_out);

    if (this->derivative_utility == nullptr) {
        MOLT_ddx_advance_per(src, dudx_out);
        MOLT_ddy_advance_per(src, dudy_out);
    } else {
        compute_ddx_numerical(u[lastStepIndex], dudx_out);
        compute_ddy_numerical(u[lastStepIndex], dudy_out);
    }

    if (this->updateMethod == MOLTEngine::CDF2) {
        for (int i = 0; i < Nx*Ny; i++) {
            u_out[i]    -= u[lastStepIndex-2][i];
            dudx_out[i] -= u[lastStepIndex-2][i];
            dudy_out[i] -= u[lastStepIndex-2][i];
        }
    }
}

void MOLTEngine::MOLT_advance_per(std::complex<double>* RHS, std::complex<double>* output) {

    std::complex<double>* tmp = new std::complex<double>[Nx*Ny];

    get_L_y_inverse_per(RHS, tmp);
    get_L_x_inverse_per(tmp, output);

    delete[] tmp;
}

void MOLTEngine::MOLT_ddx_advance_per(std::complex<double>* RHS, std::complex<double>* output) {

    std::complex<double>* tmp = new std::complex<double>[Nx*Ny];

    get_L_y_inverse_per(RHS, tmp);
    get_ddx_L_x_inverse_per(tmp, output);

    delete[] tmp;
}

void MOLTEngine::MOLT_ddy_advance_per(std::complex<double>* RHS, std::complex<double>* output) {

    std::complex<double>* tmp = new std::complex<double>[Nx*Ny];

    get_ddy_L_y_inverse_per(RHS, tmp);
    get_L_x_inverse_per(tmp, output);

    delete[] tmp;
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
    
    const double xMax = x[Nx-1] + dx;
    const double alpha = beta/(kappa*dt);
    const double mu_x = std::exp(-alpha*( xMax - x[0] ) );

    const int J_N = Nx+1;

    std::vector<std::complex<double>> u_ext(Nx+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    std::vector<std::complex<double>> J(J_N);

    // Go row by row
    for (int j = 0; j < Ny; j++) {

        int idx_Nm2 = computeIndex(Nx-2, j);
        int idx_Nm1 = computeIndex(Nx-1, j);

        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];

        for (int i = 0; i < Nx; i++) {
            int idx = computeIndex(i, j);
            u_ext[i+2] = u[idx];
        }

        int idx0 = computeIndex(0, j);
        int idx1 = computeIndex(1, j);
        int idx2 = computeIndex(2, j);

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

        apply_A_and_B(J, x, dx, Nx, alpha, A_x, B_x, true);

        for (int i = 0; i < Nx; i++) {
            int idx = computeIndex(i, j);
            inverseOut[idx] = J[i];
        }
    }
}

void MOLTEngine::get_L_y_inverse_per(std::complex<double>* u, std::complex<double>* inverseOut) {
    
    const double yMax = y[Ny-1] + dy;
    const double alpha = beta/(kappa*dt);
    const double mu_y = std::exp(-alpha*( yMax - y[0] ) );

    const int J_N = Ny+1;

    std::vector<std::complex<double>> u_ext(Ny+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    std::vector<std::complex<double>> J(J_N);

    // Go column by column
    for (int i = 0; i < Nx; i++) {

        int idx_Nm2 = computeIndex(i, Ny-2);
        int idx_Nm1 = computeIndex(i, Ny-1);

        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];

        for (int j = 0; j < Ny; j++) {
            int idx = computeIndex(i, j);
            u_ext[j+2] = u[idx];
        }

        int idx0 = computeIndex(i, 0);
        int idx1 = computeIndex(i, 1);
        int idx2 = computeIndex(i, 2);

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
            int idx = computeIndex(i, j);
            inverseOut[idx] = J[j];
        }
    }
}

void MOLTEngine::get_ddx_L_x_inverse_per(std::complex<double>* u, std::complex<double>* ddxOut) {

    const double xMax = x[Nx-1] + dx;
    const double alpha = beta/(kappa*dt);
    const double mu_x = std::exp(-alpha*( xMax - x[0] ));

    const int J_N = Nx+1;

    std::vector<std::complex<double>> u_ext(Nx+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    // std::vector<std::complex<double>> J(J_N);
    std::vector<std::complex<double>> ddx(J_N);
    // std::vector<std::vector<std::complex<double>>> inverse(Nx*Ny);

    // Go row by row
    for (int j = 0; j < Ny; j++) {

        int idx_Nm2 = computeIndex(Nx-2, j);
        int idx_Nm1 = computeIndex(Nx-1, j);

        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];

        for (int i = 0; i < Nx; i++) {
            int idx = computeIndex(i, j);
            u_ext[i+2] = u[idx];
        }

        int idx0 = computeIndex(0, j);
        int idx1 = computeIndex(1, j);
        int idx2 = computeIndex(2, j);

        u_ext[Nx+2] = u[idx0];
        u_ext[Nx+3] = u[idx1];
        u_ext[Nx+4] = u[idx2];

        linear5_L(u_ext, alpha, J_R);
        linear5_R(u_ext, alpha, J_L);

        fast_convolution(J_R, J_L, alpha);

        double A_x = (J_R[J_N - 1].real() + J_L[J_N - 1].real()) / (2.0 - 2.0*mu_x);
        double B_x = (J_R[0      ].real() + J_L[0      ].real()) / (2.0 - 2.0*mu_x);

        for (int i = 0; i < J_N-1; i++) {
            double mu_i1 = std::exp(-alpha*(x[i] - x[0]));
            double mu_i2 = std::exp(-alpha*(xMax - x[i]));
            ddx[i] = .5*alpha*(-J_R[i] + J_L[i]) - alpha*A_x*mu_i1 + alpha*B_x*mu_i2;
        }
        double mu_i1 = std::exp(-alpha*(xMax - x[0]));
        double mu_i2 = std::exp(-alpha*(xMax - xMax));
        ddx[J_N-1] = .5*alpha*(-J_R[J_N-1] + J_L[J_N-1]) - alpha*A_x*mu_i1 + alpha*B_x*mu_i2;

        for (int i = 0; i < Nx; i++) {
            int idx = computeIndex(i, j);
            ddxOut[idx] = ddx[i];
        }
    }
}

void MOLTEngine::get_ddy_L_y_inverse_per(std::complex<double>* u, std::complex<double>* ddyOut) {

    const double yMax = y[Ny-1] + dy;
    const double alpha = beta/(kappa*dt);
    const double mu_y = std::exp(-alpha*( yMax - y[0]));

    const int J_N = Ny+1;

    std::vector<std::complex<double>> u_ext(Ny+5);
    std::vector<std::complex<double>> J_L(J_N);
    std::vector<std::complex<double>> J_R(J_N);
    // std::vector<std::complex<double>> J(J_N);
    std::vector<std::complex<double>> ddy(J_N);

    // Go column by column
    for (int i = 0; i < Nx; i++) {

        int idx_Nm2 = computeIndex(i, Ny-2);
        int idx_Nm1 = computeIndex(i, Ny-1);

        u_ext[0] = u[idx_Nm2];
        u_ext[1] = u[idx_Nm1];

        for (int j = 0; j < Ny; j++) {
            int idx = computeIndex(i, j);
            u_ext[j+2] = u[idx];
        }

        int idx0 = computeIndex(i, 0);
        int idx1 = computeIndex(i, 1);
        int idx2 = computeIndex(i, 2);

        u_ext[Ny+2] = u[idx0];
        u_ext[Ny+3] = u[idx1];
        u_ext[Ny+4] = u[idx2];

        linear5_L(u_ext, alpha, J_R);
        linear5_R(u_ext, alpha, J_L);

        fast_convolution(J_R, J_L, alpha);

        double A_y = (J_R[J_N - 1].real() + J_L[J_N - 1].real()) / (2.0 - 2.0*mu_y);
        double B_y = (J_R[0      ].real() + J_L[0      ].real()) / (2.0 - 2.0*mu_y);

        for (int j = 0; j < J_N-1; j++) {
            double mu_j1 = std::exp(-alpha*(y[j] - y[0]));
            double mu_j2 = std::exp(-alpha*(yMax - y[j]));
            ddy[j] = .5*alpha*(-J_R[j] + J_L[j]) - alpha*A_y*mu_j1 + alpha*B_y*mu_j2;
        }
        double mu_j1 = std::exp(-alpha*(yMax - y[0]));
        double mu_j2 = std::exp(-alpha*(yMax - yMax));
        ddy[J_N-1] = .5*alpha*(-J_R[J_N-1] + J_L[J_N-1]) - alpha*A_y*mu_j1 + alpha*B_y*mu_j2;

        for (int j = 0; j < Ny; j++) {
            int idx = computeIndex(i, j);
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

void MOLTEngine::apply_A_and_B(std::vector<std::complex<double>> &I_, double* x, double dx, int N, double alpha, double A, double B, bool debug) {

    int I_len = I_.size();
    double xMax = x[N-1] + dx;

    for (int i = 0; i < I_len-1; i++) {
        I_[i] += A*std::exp(-alpha*( x[i] - x[0] ));
        I_[i] += B*std::exp(-alpha*( xMax - x[i] ));
    }
    I_[I_len-1] += A*std::exp(-alpha*( xMax - x[0] ));
    I_[I_len-1] += B*std::exp(-alpha*( xMax - xMax ));
}
