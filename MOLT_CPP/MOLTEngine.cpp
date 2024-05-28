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

using std::complex;

// This is assuming a 2D Hz mode Yee grid
class MOLTEngine {

    public:
        MOLTEngine(int Nx, int Ny, int Np, int Nh, double* x, double* y, 
                   std::vector<std::vector<double>>& x_elec, std::vector<std::vector<double>>& y_elec,
                   std::vector<std::vector<double>>& vx_elec, std::vector<std::vector<double>>& vy_elec,
                   double dx, double dy, double dt, double sigma_1, double sigma_2, double kappa, double q_elec, double m_elec, double beta);
        void step();
        void print();
        double getTime();
        int getStep();
        double getGaugeL2();
        void printTimeDiagnostics();

    private:
        int Nx;
        int Ny;
        int Np;
        int Nh;
        int lastStepIndex;
        double* x;
        double* y;
        std::vector<std::vector<double>> x_elec;
        std::vector<std::vector<double>> y_elec;
        std::vector<std::vector<double>> vx_elec;
        std::vector<std::vector<double>> vy_elec;
        std::vector<std::vector<double>> Px_elec;
        std::vector<std::vector<double>> Py_elec;
        double gaugeL2;
        double dx;
        double dy;
        double dt;
        double t;
        int n;
        double kappa;
        double sigma_1;
        double sigma_2;
        double beta;
        double elec_charge;
        double elec_mass;
        double w_elec;
        std::vector<std::vector<std::vector<std::complex<double>>>> phi;
        std::vector<std::vector<std::vector<std::complex<double>>>> ddx_phi;
        std::vector<std::vector<std::vector<std::complex<double>>>> ddy_phi;
        std::vector<std::vector<std::vector<std::complex<double>>>> A1;
        std::vector<std::vector<std::vector<std::complex<double>>>> ddx_A1;
        std::vector<std::vector<std::vector<std::complex<double>>>> ddy_A1;
        std::vector<std::vector<std::vector<std::complex<double>>>> A2;
        std::vector<std::vector<std::vector<std::complex<double>>>> ddx_A2;
        std::vector<std::vector<std::vector<std::complex<double>>>> ddy_A2;
        std::vector<std::vector<std::vector<std::complex<double>>>> rho;
        std::vector<std::vector<std::vector<std::complex<double>>>> J1;
        std::vector<std::vector<std::vector<std::complex<double>>>> J2;
        std::vector<std::vector<std::complex<double>>> ddx_J1;
        std::vector<std::vector<std::complex<double>>> ddy_J2;
        std::vector<std::vector<std::vector<std::complex<double>>>> currentFields;
        std::vector<double> kx_deriv_1, ky_deriv_1;
        std::vector<double> kx_deriv_2, ky_deriv_2;
        std::complex<double>* forwardIn;
        std::complex<double>* forwardOut;
        std::complex<double>* backwardIn;
        std::complex<double>* backwardOut;
        fftw_plan forward_plan, inverse_plan;

        double timeComponent1, timeComponent2, timeComponent3, timeComponent4, timeComponent5, timeComponent6;

        void computeGaugeL2();
        void updateParticleLocations();
        void updateParticleVelocities();
        void scatterFields();
        void updateWaves();
        void shuffleSteps();
        void updatePhi();
        void updateA1();
        void updateA2();
        double gatherField(double p_x, double p_y, std::vector<std::vector<std::complex<double>>>& field);
        void gatherFields(double p_x, double p_y, std::vector<std::vector<std::vector<std::complex<double>>>>& fields, std::vector<double>& fields_out);
        void scatterField(double p_x, double p_y, double value, std::vector<std::vector<std::complex<double>>>& field);

        void computeFirstDerivative(const std::vector<std::vector<std::complex<double>>>& inputField, 
                        std::vector<std::vector<std::complex<double>>>& derivativeField, bool isDerivativeInX);
        void computeSecondDerivative(const std::vector<std::vector<std::complex<double>>>& inputField, 
                std::vector<std::vector<std::complex<double>>>& derivativeField, bool isDerivativeInX);
        
        void solveHelmholtzEquation(std::vector<std::vector<std::complex<double>>>& RHS,
                                                std::vector<std::vector<std::complex<double>>>& LHS, double alpha);

        void compute_ddx(const std::vector<std::vector<std::complex<double>>>& inputField, 
                               std::vector<std::vector<std::complex<double>>>& derivativeField) {
                                    computeFirstDerivative(inputField, derivativeField, true);
                               }
        void compute_ddy(const std::vector<std::vector<std::complex<double>>>& inputField, 
                               std::vector<std::vector<std::complex<double>>>& derivativeField) {
                                    computeFirstDerivative(inputField, derivativeField, false);
                               }
        void compute_d2dx(const std::vector<std::vector<std::complex<double>>>& inputField, 
                                std::vector<std::vector<std::complex<double>>>& derivativeField) {
                                    computeSecondDerivative(inputField, derivativeField, true);
                                }
        void compute_d2dy(const std::vector<std::vector<std::complex<double>>>& inputField, 
                                std::vector<std::vector<std::complex<double>>>& derivativeField) {
                                    computeSecondDerivative(inputField, derivativeField, false);
                                }

        std::complex<double> to_std_complex(const fftw_complex& fc) {
            return std::complex<double>(fc[0], fc[1]);
        }
};

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
    struct timeval begin1, end1, begin2, end2, begin3, end3, begin4, end4, begin5, end5, begin6, end6;
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
    // std::cout << "Shuffling Steps" << std::endl;
    gettimeofday( &begin5, NULL );
    computeGaugeL2();
    gettimeofday( &end5, NULL );
    gettimeofday( &begin6, NULL );
    shuffleSteps();
    gettimeofday( &end6, NULL );
    // std::cout << "Rinse, Repeat" << std::endl;
    if (n % 100 == 0) {
        print();
    }
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

/**
 * Name: updateParticleVelocities
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
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            ddt_phi = (phi[lastStepIndex][i][j].real() - phi[lastStepIndex-1][i][j].real()) / dt;
            div_A = ddx_A1[lastStepIndex][i][j].real() + ddy_A2[lastStepIndex][i][j].real();
            l2 += std::pow(1/(kappa*kappa)*ddt_phi + div_A,2);
        }
    }
    gaugeL2 = std::sqrt(dx*dy*l2);
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
    std::string nxn = std::to_string(Nx-1) + "x" + std::to_string(Ny-1);
    std::string path = "results/" + nxn + "/";
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
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny-1; j++) {
            phiFile << std::to_string(phi[lastStepIndex][i][j].real()) + ",";
            A1File << std::to_string(A1[lastStepIndex][i][j].real()) + ",";
            A2File << std::to_string(A2[lastStepIndex][i][j].real()) + ",";
            // rhoFile << std::to_string(rho[lastStepIndex][i][j].real()) + ",";
            // J1File << std::to_string(J1[lastStepIndex][i][j].real()) + ",";
            // J2File << std::to_string(J2[lastStepIndex][i][j].real()) + ",";
        }
        phiFile << std::to_string(phi[lastStepIndex][i][Ny-1].real());
        A1File << std::to_string(A1[lastStepIndex][i][Ny-1].real());
        A2File << std::to_string(A2[lastStepIndex][i][Ny-1].real());
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
    for (int p = 0; p < Np; p++) {
        electronFile << std::to_string(x_elec[lastStepIndex][p]) + "," + std::to_string(y_elec[lastStepIndex][p]) + "," + std::to_string(vx_elec[lastStepIndex][p]) + "," + std::to_string(vy_elec[lastStepIndex][p]) << "\n";
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
    for (int i = 0; i < this->Np; i++) {
        vx_star = 2.0*this->vx_elec[lastStepIndex-1][i] - this->vx_elec[lastStepIndex-2][i];
        vy_star = 2.0*this->vy_elec[lastStepIndex-1][i] - this->vy_elec[lastStepIndex-2][i];

        this->x_elec[lastStepIndex][i] = this->x_elec[lastStepIndex-1][i] + dt*vx_star;
        this->y_elec[lastStepIndex][i] = this->y_elec[lastStepIndex-1][i] + dt*vy_star;

        this->x_elec[lastStepIndex][i] = this->x_elec[lastStepIndex][i] - Lx*floor((this->x_elec[lastStepIndex][i] - this->x[0]) / Lx);
        this->y_elec[lastStepIndex][i] = this->y_elec[lastStepIndex][i] - Ly*floor((this->y_elec[lastStepIndex][i] - this->y[0]) / Ly);
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
    double alpha = this->beta/(this->kappa*this->dt);
    // BDF1
    std::vector<std::vector<std::complex<double>>> phi_src(Nx, std::vector<std::complex<double>>(Ny));
    std::vector<std::vector<std::complex<double>>> A1_src(Nx, std::vector<std::complex<double>>(Ny));
    std::vector<std::vector<std::complex<double>>> A2_src(Nx, std::vector<std::complex<double>>(Ny));

    double alpha2 = alpha*alpha;

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            phi_src[i][j] = 2.0*this->phi[lastStepIndex-1][i][j] - this->phi[lastStepIndex-2][i][j] + 1.0/alpha2 * 1.0/this->sigma_1 * this->rho[lastStepIndex][i][j];
            A1_src[i][j] = 2.0*this->A1[lastStepIndex-1][i][j] - this->A1[lastStepIndex-2][i][j] + 1.0/alpha2 * this->sigma_2 * this->J1[lastStepIndex][i][j];
            A2_src[i][j] = 2.0*this->A2[lastStepIndex-1][i][j] - this->A2[lastStepIndex-2][i][j] + 1.0/alpha2 * this->sigma_2 * this->J2[lastStepIndex][i][j];
        }
    }

    solveHelmholtzEquation(phi_src, this->phi[lastStepIndex], alpha);
    solveHelmholtzEquation(A1_src,  this->A1[lastStepIndex], alpha);
    solveHelmholtzEquation(A2_src,  this->A2[lastStepIndex], alpha);

    compute_ddx(this->phi[lastStepIndex], this->ddx_phi[lastStepIndex]);
    compute_ddy(this->phi[lastStepIndex], this->ddy_phi[lastStepIndex]);
    compute_ddx(this->A1[lastStepIndex],  this->ddx_A1[lastStepIndex]);
    compute_ddy(this->A1[lastStepIndex],  this->ddy_A1[lastStepIndex]);
    compute_ddx(this->A2[lastStepIndex],  this->ddx_A2[lastStepIndex]);
    compute_ddy(this->A2[lastStepIndex],  this->ddy_A2[lastStepIndex]);
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
 *              Commented out code exists for posterity, may remove eventually, it is an eyesore.
 * Inputs: none
 * Output: none
 * Dependencies: OpenMP
 */
void MOLTEngine::updateParticleVelocities() {

    // int thread_id;
    // int maxThreads = omp_get_max_threads();
    // int numParticlesPerThread = Np / (maxThreads);

    // std::vector<double> fields_p(8);
    // double ddx_phi_p, ddy_phi_p, A1_p, ddx_A1_p, ddy_A1_p, A2_p, ddx_A2_p, ddy_A2_p;
    
    // #pragma omp parallel private(thread_id, ddx_phi_p, ddy_phi_p, A1_p, ddx_A1_p, ddy_A1_p, A2_p, ddx_A2_p, ddy_A2_p) shared(fields_p, x_elec, y_elec)
    // {

    //     thread_id = omp_get_thread_num();
    //     int startIndex = thread_id * numParticlesPerThread;
    //     for (int i = startIndex; i < startIndex + numParticlesPerThread; i++) {

    //         // std::cout << "Thread " << thread_id << " updating " << i << " particle." << std::endl;

    //         gatherFields(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], currentFields, fields_p);
    //         ddx_phi_p = fields_p[0];
    //         ddy_phi_p = fields_p[1];
    //         A1_p = fields_p[2];
    //         ddx_A1_p = fields_p[3];
    //         ddy_A1_p = fields_p[4];
    //         A2_p = fields_p[5];
    //         ddx_A2_p = fields_p[6];
    //         ddy_A2_p = fields_p[7];

    //         double vx_star = 2.0*vx_elec[lastStepIndex-1][i] - vx_elec[lastStepIndex-2][i];
    //         double vy_star = 2.0*vy_elec[lastStepIndex-1][i] - vy_elec[lastStepIndex-2][i];

    //         double rhs1 = -elec_charge*ddx_phi_p + elec_charge*( ddx_A1_p*vx_star + ddx_A2_p*vy_star );
    //         double rhs2 = -elec_charge*ddy_phi_p + elec_charge*( ddy_A1_p*vx_star + ddy_A2_p*vy_star );

    //         // Compute the new momentum
    //         Px_elec[lastStepIndex][i] = Px_elec[lastStepIndex-1][i] + dt*rhs1;
    //         Py_elec[lastStepIndex][i] = Py_elec[lastStepIndex-1][i] + dt*rhs2;

    //         double denom = std::sqrt(std::pow(Px_elec[lastStepIndex][i] - elec_charge*A1_p, 2) + std::pow(Py_elec[lastStepIndex][i] - elec_charge*A2_p, 2) + std::pow(elec_mass*kappa, 2));

    //         // Compute the new velocity using the updated momentum
    //         vx_elec[lastStepIndex][i] = (kappa*(Px_elec[lastStepIndex][i] - elec_charge*A1_p)) / denom;
    //         vy_elec[lastStepIndex][i] = (kappa*(Py_elec[lastStepIndex][i] - elec_charge*A2_p)) / denom;
    //     }
    // }
    // for (int i = numParticlesPerThread*maxThreads + numParticlesPerThread; i < Np; i++) {

    //     gatherFields(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], currentFields, fields_p);
    //     ddx_phi_p = fields_p[0];
    //     ddy_phi_p = fields_p[1];
    //     A1_p = fields_p[2];
    //     ddx_A1_p = fields_p[3];
    //     ddy_A1_p = fields_p[4];
    //     A2_p = fields_p[5];
    //     ddx_A2_p = fields_p[6];
    //     ddy_A2_p = fields_p[7];

    //     double vx_star = 2.0*vx_elec[lastStepIndex-1][i] - vx_elec[lastStepIndex-2][i];
    //     double vy_star = 2.0*vy_elec[lastStepIndex-1][i] - vy_elec[lastStepIndex-2][i];

    //     double rhs1 = -elec_charge*ddx_phi_p + elec_charge*( ddx_A1_p*vx_star + ddx_A2_p*vy_star );
    //     double rhs2 = -elec_charge*ddy_phi_p + elec_charge*( ddy_A1_p*vx_star + ddy_A2_p*vy_star );

    //     // Compute the new momentum
    //     Px_elec[lastStepIndex][i] = Px_elec[lastStepIndex-1][i] + dt*rhs1;
    //     Py_elec[lastStepIndex][i] = Py_elec[lastStepIndex-1][i] + dt*rhs2;

    //     double denom = std::sqrt(std::pow(Px_elec[lastStepIndex][i] - elec_charge*A1_p, 2) + std::pow(Py_elec[lastStepIndex][i] - elec_charge*A2_p, 2) + std::pow(elec_mass*kappa, 2));

    //     // Compute the new velocity using the updated momentum
    //     vx_elec[lastStepIndex][i] = (kappa*(Px_elec[lastStepIndex][i] - elec_charge*A1_p)) / denom;
    //     vy_elec[lastStepIndex][i] = (kappa*(Py_elec[lastStepIndex][i] - elec_charge*A2_p)) / denom;
    // }

    // std::vector<double> fields_p(8);
    // double ddx_phi_p, ddy_phi_p, A1_p, ddx_A1_p, ddy_A1_p, A2_p, ddx_A2_p, ddy_A2_p;
    // #pragma omp parallel for
    // for (int i = 0; i < Np; i++) {

    //     gatherFields(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], currentFields, fields_p);
    //     ddx_phi_p = fields_p[0];
    //     ddy_phi_p = fields_p[1];
    //     A1_p = fields_p[2];
    //     ddx_A1_p = fields_p[3];
    //     ddy_A1_p = fields_p[4];
    //     A2_p = fields_p[5];
    //     ddx_A2_p = fields_p[6];
    //     ddy_A2_p = fields_p[7];

    //     double vx_star = 2.0*vx_elec[lastStepIndex-1][i] - vx_elec[lastStepIndex-2][i];
    //     double vy_star = 2.0*vy_elec[lastStepIndex-1][i] - vy_elec[lastStepIndex-2][i];

    //     double rhs1 = -elec_charge*ddx_phi_p + elec_charge*( ddx_A1_p*vx_star + ddx_A2_p*vy_star );
    //     double rhs2 = -elec_charge*ddy_phi_p + elec_charge*( ddy_A1_p*vx_star + ddy_A2_p*vy_star );

    //     // Compute the new momentum
    //     Px_elec[lastStepIndex][i] = Px_elec[lastStepIndex-1][i] + dt*rhs1;
    //     Py_elec[lastStepIndex][i] = Py_elec[lastStepIndex-1][i] + dt*rhs2;

    //     double denom = std::sqrt(std::pow(Px_elec[lastStepIndex][i] - elec_charge*A1_p, 2) + std::pow(Py_elec[lastStepIndex][i] - elec_charge*A2_p, 2) + std::pow(elec_mass*kappa, 2));

    //     // Compute the new velocity using the updated momentum
    //     vx_elec[lastStepIndex][i] = (kappa*(Px_elec[lastStepIndex][i] - elec_charge*A1_p)) / denom;
    //     vy_elec[lastStepIndex][i] = (kappa*(Py_elec[lastStepIndex][i] - elec_charge*A2_p)) / denom;
    // }
    

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
        const double p_x = x_elec[lastStepIndex][i];
        const double p_y = y_elec[lastStepIndex][i];
        // ------------------------------
        // Gather Fields
        // We convert from cartesian to logical space
        const double x0 = this->x[0];
        const double y0 = this->y[0];
        const int lc_x = floor((p_x - x0)/dx);
        const int lc_y = floor((p_y - y0)/dy);

        const double xNode = this->x[lc_x];
        const double yNode = this->y[lc_y];

        // We compute the fractional distance of a particle from
        // the nearest node.
        // eg x=[0,.1,.2,.3], particleX = [.225]
        // The particle's fractional is 1/4
        const double fx = (p_x - xNode)/dx;
        const double fy = (p_y - yNode)/dy;

        ddx_phi_p += (1-fx)*(1-fy)*ddx_phi[lastStepIndex][lc_x][lc_y].real();
        ddx_phi_p += (1-fx)*(fy)*ddx_phi[lastStepIndex][lc_x][lc_y+1].real();
        ddx_phi_p += (fx)*(1-fy)*ddx_phi[lastStepIndex][lc_x+1][lc_y].real();
        ddx_phi_p += (fx)*(fy)*ddx_phi[lastStepIndex][lc_x+1][lc_y+1].real();

        ddy_phi_p += (1-fx)*(1-fy)*ddy_phi[lastStepIndex][lc_x][lc_y].real();
        ddy_phi_p += (1-fx)*(fy)*ddy_phi[lastStepIndex][lc_x][lc_y+1].real();
        ddy_phi_p += (fx)*(1-fy)*ddy_phi[lastStepIndex][lc_x+1][lc_y].real();
        ddy_phi_p += (fx)*(fy)*ddy_phi[lastStepIndex][lc_x+1][lc_y+1].real();

        A1_p += (1-fx)*(1-fy)*A1[lastStepIndex][lc_x][lc_y].real();
        A1_p += (1-fx)*(fy)*A1[lastStepIndex][lc_x][lc_y+1].real();
        A1_p += (fx)*(1-fy)*A1[lastStepIndex][lc_x+1][lc_y].real();
        A1_p += (fx)*(fy)*A1[lastStepIndex][lc_x+1][lc_y+1].real();

        ddx_A1_p += (1-fx)*(1-fy)*ddx_A1[lastStepIndex][lc_x][lc_y].real();
        ddx_A1_p += (1-fx)*(fy)*ddx_A1[lastStepIndex][lc_x][lc_y+1].real();
        ddx_A1_p += (fx)*(1-fy)*ddx_A1[lastStepIndex][lc_x+1][lc_y].real();
        ddx_A1_p += (fx)*(fy)*ddx_A1[lastStepIndex][lc_x+1][lc_y+1].real();

        ddy_A1_p += (1-fx)*(1-fy)*ddy_A1[lastStepIndex][lc_x][lc_y].real();
        ddy_A1_p += (1-fx)*(fy)*ddy_A1[lastStepIndex][lc_x][lc_y+1].real();
        ddy_A1_p += (fx)*(1-fy)*ddy_A1[lastStepIndex][lc_x+1][lc_y].real();
        ddy_A1_p += (fx)*(fy)*ddy_A1[lastStepIndex][lc_x+1][lc_y+1].real();

        A2_p += (1-fx)*(1-fy)*A2[lastStepIndex][lc_x][lc_y].real();
        A2_p += (1-fx)*(fy)*A2[lastStepIndex][lc_x][lc_y+1].real();
        A2_p += (fx)*(1-fy)*A2[lastStepIndex][lc_x+1][lc_y].real();
        A2_p += (fx)*(fy)*A2[lastStepIndex][lc_x+1][lc_y+1].real();

        ddx_A2_p += (1-fx)*(1-fy)*ddx_A2[lastStepIndex][lc_x][lc_y].real();
        ddx_A2_p += (1-fx)*(fy)*ddx_A2[lastStepIndex][lc_x][lc_y+1].real();
        ddx_A2_p += (fx)*(1-fy)*ddx_A2[lastStepIndex][lc_x+1][lc_y].real();
        ddx_A2_p += (fx)*(fy)*ddx_A2[lastStepIndex][lc_x+1][lc_y+1].real();

        ddy_A2_p += (1-fx)*(1-fy)*ddy_A2[lastStepIndex][lc_x][lc_y].real();
        ddy_A2_p += (1-fx)*(fy)*ddy_A2[lastStepIndex][lc_x][lc_y+1].real();
        ddy_A2_p += (fx)*(1-fy)*ddy_A2[lastStepIndex][lc_x+1][lc_y].real();
        ddy_A2_p += (fx)*(fy)*ddy_A2[lastStepIndex][lc_x+1][lc_y+1].real();

        double vx_star = 2.0*vx_elec[lastStepIndex-1][i] - vx_elec[lastStepIndex-2][i];
        double vy_star = 2.0*vy_elec[lastStepIndex-1][i] - vy_elec[lastStepIndex-2][i];

        double rhs1 = -elec_charge*ddx_phi_p + elec_charge*( ddx_A1_p*vx_star + ddx_A2_p*vy_star );
        double rhs2 = -elec_charge*ddy_phi_p + elec_charge*( ddy_A1_p*vx_star + ddy_A2_p*vy_star );

        // Compute the new momentum
        Px_elec[lastStepIndex][i] = Px_elec[lastStepIndex-1][i] + dt*rhs1;
        Py_elec[lastStepIndex][i] = Py_elec[lastStepIndex-1][i] + dt*rhs2;

        double denom = std::sqrt(std::pow(Px_elec[lastStepIndex][i] - elec_charge*A1_p, 2) + std::pow(Py_elec[lastStepIndex][i] - elec_charge*A2_p, 2) + std::pow(elec_mass*kappa, 2));

        // Compute the new velocity using the updated momentum
        vx_elec[lastStepIndex][i] = (kappa*(Px_elec[lastStepIndex][i] - elec_charge*A1_p)) / denom;
        vy_elec[lastStepIndex][i] = (kappa*(Py_elec[lastStepIndex][i] - elec_charge*A2_p)) / denom;
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
void MOLTEngine::solveHelmholtzEquation(std::vector<std::vector<std::complex<double>>>& RHS,
                                        std::vector<std::vector<std::complex<double>>>& LHS, double alpha) {

    int Nx = this->Nx - 1;
    int Ny = this->Ny - 1;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            forwardIn[i * Ny + j] = RHS[i][j];
        }
    }
        
    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
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
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            LHS[i][j] = backwardOut[i * Ny + j] / double(Nx * Ny);
        }
    }
    for (int i = 0; i < this->Nx; i++) {
        LHS[i][this->Ny-1] = LHS[i][0];
    }
    for (int j = 0; j < this->Ny; j++) {
        LHS[this->Nx-1][j] = LHS[0][j];
    }
}

/**
 * Name: shuffleSteps
 * Author: Stephen White
 * Date Created: 5/28/24
 * Date Last Modified: 5/28/24 (Stephen White)
 * Description: We use time history throughout this simulation, which means we need a copy of Nh previous timesteps.
 *              At the end of each iteration we shuffle the results down one timestep, making room for the next.
 * Inputs: none (relies on global values x_elec, y_elec, vx_elec, vy_elec, Px_elec, Py_elec, phi, ddx_phi, ddy_phi, A1, ddx_A1, ddy_A1, A2, ddx_A2, ddy_A2, and currentFields)
 * Output: none
 * Dependencies: none
 */
void MOLTEngine::shuffleSteps() {
    for (int h = 0; h < lastStepIndex; h++) {
        x_elec[h].assign(x_elec[h+1].begin(), x_elec[h+1].end());
        y_elec[h].assign(y_elec[h+1].begin(), y_elec[h+1].end());
        vx_elec[h].assign(vx_elec[h+1].begin(), vx_elec[h+1].end());
        vy_elec[h].assign(vy_elec[h+1].begin(), vy_elec[h+1].end());
        Px_elec[h].assign(Px_elec[h+1].begin(), Px_elec[h+1].end());
        Py_elec[h].assign(Py_elec[h+1].begin(), Py_elec[h+1].end());

        phi[h].assign(phi[h+1].begin(), phi[h+1].end());
        ddx_phi[h].assign(ddx_phi[h+1].begin(), ddx_phi[h+1].end());
        ddy_phi[h].assign(ddy_phi[h+1].begin(), ddy_phi[h+1].end());
        A1[h].assign(A1[h+1].begin(), A1[h+1].end());
        ddx_A1[h].assign(ddx_A1[h+1].begin(), ddx_A1[h+1].end());
        ddy_A1[h].assign(ddy_A1[h+1].begin(), ddy_A1[h+1].end());
        A2[h].assign(A2[h+1].begin(), A2[h+1].end());
        ddx_A2[h].assign(ddx_A2[h+1].begin(), ddx_A2[h+1].end());
        ddy_A2[h].assign(ddy_A2[h+1].begin(), ddy_A2[h+1].end());

        rho[h].assign(rho[h+1].begin(), rho[h+1].end());
        J1[h].assign(J1[h+1].begin(), J1[h+1].end());
        J2[h].assign(J2[h+1].begin(), J2[h+1].end());
    }
    currentFields[0] = ddx_phi[lastStepIndex];
    currentFields[1] = ddy_phi[lastStepIndex];
    currentFields[2] = A1[lastStepIndex];
    currentFields[3] = ddx_A1[lastStepIndex];
    currentFields[4] = ddy_A1[lastStepIndex];
    currentFields[5] = A2[lastStepIndex];
    currentFields[6] = ddx_A2[lastStepIndex];
    currentFields[7] = ddy_A2[lastStepIndex];
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
void MOLTEngine::gatherFields(double p_x, double p_y, std::vector<std::vector<std::vector<std::complex<double>>>>& fields, std::vector<double>& fields_out) {
    // We convert from cartesian to logical space
    const double x0 = this->x[0];
    const double y0 = this->y[0];
    const int lc_x = floor((p_x - x0)/dx);
    const int lc_y = floor((p_y - y0)/dy);

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
        field_00 = fields[i][lc_x][lc_y].real();
        field_01 = fields[i][lc_x][lc_y+1].real();
        field_10 = fields[i][lc_x+1][lc_y].real();
        field_11 = fields[i][lc_x+1][lc_y+1].real();

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
    for (int i = 0; i < Nx; i++) {
        std::fill(J1[lastStepIndex][i].begin(), J1[lastStepIndex][i].end(), 0.0);
        std::fill(J2[lastStepIndex][i].begin(), J2[lastStepIndex][i].end(), 0.0);
    }

    for (int i = 0; i < Np; i++) {
        // double vx_star = 2.0*this->vx_elec[lastStepIndex-1][i] - this->vx_elec[lastStepIndex-2][i];
        // double vy_star = 2.0*this->vy_elec[lastStepIndex-1][i] - this->vy_elec[lastStepIndex-2][i];

        // double x_value = this->elec_charge*vx_star*this->w_elec;
        // double y_value = this->elec_charge*vy_star*this->w_elec;

        double x_value = this->elec_charge*vx_elec[lastStepIndex-1][i]*this->w_elec;
        double y_value = this->elec_charge*vy_elec[lastStepIndex-1][i]*this->w_elec;

        scatterField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], x_value, this->J1[lastStepIndex]);
        scatterField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], y_value, this->J2[lastStepIndex]);
    }
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            J1[lastStepIndex][i][j] /= dx*dy;
            J2[lastStepIndex][i][j] /= dx*dy;
        }
    }

    // Enforce periodicity
    for (int i = 0; i < this->Nx; i++) {
        J1[lastStepIndex][i][0] += J1[lastStepIndex][i][this->Ny-1];
        J1[lastStepIndex][i][this->Ny-1] = J1[lastStepIndex][i][0];

        J2[lastStepIndex][i][0] += J2[lastStepIndex][i][this->Ny-1];
        J2[lastStepIndex][i][this->Ny-1] = J2[lastStepIndex][i][0];
    }
    for (int j = 0; j < this->Ny; j++) {
        J1[lastStepIndex][0][j] += J1[lastStepIndex][this->Nx-1][j];
        J1[lastStepIndex][this->Nx-1][j] = J1[lastStepIndex][0][j];
        
        J2[lastStepIndex][0][j] += J2[lastStepIndex][this->Nx-1][j];
        J2[lastStepIndex][this->Nx-1][j] = J2[lastStepIndex][0][j];
    }

    // Compute div J
    compute_ddx(J1[lastStepIndex], ddx_J1);
    compute_ddy(J2[lastStepIndex], ddy_J2);

    for (int i = 0; i < this->Nx; i++) {
        for (int j = 0; j < this->Ny; j++) {
            rho[lastStepIndex][i][j] = this->rho[lastStepIndex-1][i][j] - dt*(ddx_J1[i][j] + ddy_J2[i][j]);
        }
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
double MOLTEngine::gatherField(double p_x, double p_y, std::vector<std::vector<std::complex<double>>>& field) {
    // We convert from cartesian to logical space
    double x0 = this->x[0];
    double y0 = this->y[0];
    int lc_x = floor((p_x - x0)/dx);
    int lc_y = floor((p_y - y0)/dy);

    double xNode = this->x[lc_x];
    double yNode = this->y[lc_y];

    // We compute the fractional distance of a particle from
    // the nearest node.
    // eg x=[0,.1,.2,.3], particleX = [.225]
    // The particle's fractional is 1/4
    double fx = (p_x - xNode)/dx;
    double fy = (p_y - yNode)/dy;

    // Now we acquire the field values at the surrounding nodes
    const double field_00 = field[lc_x][lc_y].real();
    const double field_01 = field[lc_x][lc_y+1].real();
    const double field_10 = field[lc_x+1][lc_y].real();
    const double field_11 = field[lc_x+1][lc_y+1].real();

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
void MOLTEngine::scatterField(double p_x, double p_y, double value, std::vector<std::vector<std::complex<double>>>& field) {

    // We convert from cartesian to logical space
    double x0 = this->x[0];
    double y0 = this->y[0];

    int lc_x = floor((p_x - x0)/dx);
    int lc_y = floor((p_y - y0)/dy);

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
    field[lc_x][lc_y]     += (1-fx)*(1-fy)*value;
    field[lc_x][lc_y+1]   += (1-fx)*(fy)*value;
    field[lc_x+1][lc_y]   += (fx)*(1-fy)*value;
    field[lc_x+1][lc_y+1] += (fx)*(fy)*value;
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
void MOLTEngine::computeFirstDerivative(const std::vector<std::vector<std::complex<double>>>& inputField, 
                                        std::vector<std::vector<std::complex<double>>>& derivativeField,
                                        bool isDerivativeInX) {

    int Nx = this->Nx - 1;
    int Ny = this->Ny - 1;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            forwardIn[i * Ny + j] = inputField[i][j];
        }
    }

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = isDerivativeInX ? kx_deriv_1 : ky_deriv_1;

    // Apply the derivative operator in the frequency domain
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
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
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            derivativeField[i][j] = backwardOut[i * Ny + j] / double(Nx * Ny);
        }
    }

    // Periodic BC
    for (int i = 0; i < this->Nx; i++) {
        derivativeField[i][this->Ny-1] = derivativeField[i][0];
    }
    for (int j = 0; j < this->Ny; j++) {
        derivativeField[this->Nx-1][j] = derivativeField[0][j];
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
void MOLTEngine::computeSecondDerivative(const std::vector<std::vector<std::complex<double>>>& inputField, 
                                         std::vector<std::vector<std::complex<double>>>& derivativeField,
                                         bool isDerivativeInX) {

    int Nx = this->Nx - 1;
    int Ny = this->Ny - 1;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            forwardIn[i * Ny + j] = inputField[i][j];
        }
    }

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = isDerivativeInX ? kx_deriv_2 : ky_deriv_2;

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
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
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            derivativeField[i][j] = backwardOut[i * Ny + j] / double(Nx * Ny);
        }
    }

    for (int i = 0; i < this->Nx; i++) {
        derivativeField[i][this->Ny-1] = derivativeField[i][0];
    }
    for (int j = 0; j < this->Ny; j++) {
        derivativeField[this->Nx-1][j] = derivativeField[0][j];
    }
}