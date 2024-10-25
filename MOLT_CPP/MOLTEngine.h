#include <cmath>
#include <iostream>
#include <fstream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <vector>
#include <complex.h>
#include <fftw3.h>
#include <sys/time.h>

#include "Derivative.h"
#include "Interpolate.h"
#include "LinearInterpolate.h"
#include "QuadraticInterpolate.h"
#include "FFT.h"
#include "FD6.h"

class MOLTEngine {

    public:
        enum NumericalMethod { BDF1, BDF2, BDF3, DIRK2, DIRK3, CDF2 };
        enum MOLTMethod { Integral, Helmholtz };
        enum RhoUpdate { CONSERVING, NAIVE };
        
        MOLTEngine(int Nx, int Ny, int numElectrons, int numIons, int Nh, double* x, double* y,
                   std::vector<std::vector<double>*>& x_elec, std::vector<std::vector<double>*>& y_elec,
                   std::vector<std::vector<double>*>& vx_elec, std::vector<std::vector<double>*>& vy_elec,
                   std::vector<double>& x_ion, std::vector<double>& y_ion,
                   std::vector<double>& vx_ion, std::vector<double>& vy_ion,
                   double dx, double dy, double dt, double sigma_1, double sigma_2, double kappa,
                   double q_elec, double m_elec, double q_ions, double m_ions,
                   double w_elec, double w_ion,
                   MOLTEngine::NumericalMethod updateMethod, MOLTEngine::MOLTMethod moltMethod, MOLTEngine::RhoUpdate rhoUpdate,
                   Derivative::DerivativeMethod derivativeMethod, Interpolate::InterpolateMethod interpolateMethod,
                   bool correctGauge, std::string savePath, std::string debugPath, bool debugViaMatlab=false) {

            this->Nx = Nx;
            this->Ny = Ny;
            this->numElectrons = numElectrons;
            this->numIons = numIons;
            this->Nh = Nh;
            this->lastStepIndex = Nh-1;
            this->x = x;
            this->y = y;
            this->dx = dx;
            this->dy = dy;
            this->dt = dt;

            this->sigma_1 = sigma_1;
            this->sigma_2 = sigma_2;
            this->kappa = kappa;

            this->q_ele = q_elec;
            this->m_ele = m_elec;
            this->q_ion = q_ions;
            this->m_ion = m_ions;
            this->t = 0;
            this->n = 0;

            this->x_elec = x_elec;
            this->y_elec = y_elec;
            this->vx_elec = vx_elec;
            this->vy_elec = vy_elec;

            this->x_ion = x_ion;
            this->y_ion = y_ion;
            this->vx_ion = vx_ion;
            this->vy_ion = vy_ion;

            this->w_ele = w_elec;
            this->w_ion = w_ion;

            this->correctTheGauge = correctGauge;

            double Lx = x[Nx-1] - x[0] + dx;
            double Ly = y[Nx-1] - y[0] + dy;

            if (derivativeMethod == Derivative::DerivativeMethod::FD6) {
                std::cout << "Creating FD6" << std::endl;
                derivative_utility = new FD6(Nx, Ny, Lx, Ly);
            } else if (derivativeMethod == Derivative::DerivativeMethod::FFT) {
                std::cout << "Creating FFT" << std::endl;
                derivative_utility = new FFT(Nx, Ny, Lx, Ly);
            } else {
                derivative_utility = nullptr;
                // Derivatives will be solved by MOLT
            }

            if (interpolateMethod == Interpolate::Linear) {
                interpolate_utility = new LinearInterpolate(Nx, Ny, x, y);
            } else if (interpolateMethod == Interpolate::Quadratic) {
                interpolate_utility = new QuadraticInterpolate(Nx, Ny, x, y);
            } else {
                std::cout << "NO SUCH INTERPOLATE METHOD" << std::endl;
                throw -1;
            }

            if (updateMethod == MOLTEngine::BDF1) {
                beta = 1.0;
            } else if (updateMethod == MOLTEngine::BDF2) {
                beta = 1.0 / (2.0/3.0);
            } else if (updateMethod == MOLTEngine::BDF3) {
                beta = 1.0 / (6.0/11.0);
            } else if (updateMethod == MOLTEngine::DIRK2) {

            } else if (updateMethod == MOLTEngine::DIRK3) {

            } else if (updateMethod == MOLTEngine::CDF2) {
                beta = std::sqrt(2);
            } else {
                throw -1;
            }

            std::vector<std::vector<double>*> Px_elec(Nh);
            std::vector<std::vector<double>*> Py_elec(Nh);
            std::vector<double> Px_ion(numIons);
            std::vector<double> Py_ion(numIons);

            for (int h = 0; h < Nh; h++) {
                Px_elec[h] = new std::vector<double>(numElectrons);
                Py_elec[h] = new std::vector<double>(numElectrons);
                for (int i = 0; i < numElectrons; i++) {
                    double v_mag2 = std::sqrt((*vx_elec[h])[i]*(*vx_elec[h])[i] + (*vy_elec[h])[i]*(*vy_elec[h])[i]);
                    double gamma = 1.0 / std::sqrt(1.0 - std::pow(v_mag2 / kappa, 2));
                    (*Px_elec[h])[i] = gamma * m_ele * (*vx_elec[h])[i];
                    (*Py_elec[h])[i] = gamma * m_ele * (*vy_elec[h])[i];
                }
            }

            for (int i = 0; i < numIons; i++) {
                double v_mag2 = std::sqrt(vx_ion[i]*vx_ion[i] + vy_ion[i]*vy_ion[i]);
                double gamma = 1.0 / std::sqrt(1.0 - std::pow(v_mag2 / kappa, 2));
                Px_ion[i] = gamma * m_ion * vx_ion[i];
                Py_ion[i] = gamma * m_ion * vy_ion[i];
            }

            this->Px_elec = Px_elec;
            this->Py_elec = Py_elec;
            this->Px_ion = Px_ion;
            this->Py_ion = Py_ion;

            std::vector<std::complex<double>*> phi(Nh);
            std::vector<std::complex<double>*> ddx_phi(Nh);
            std::vector<std::complex<double>*> ddy_phi(Nh);
            std::vector<std::complex<double>*> A1(Nh);
            std::vector<std::complex<double>*> ddx_A1(Nh);
            std::vector<std::complex<double>*> ddy_A1(Nh);
            std::vector<std::complex<double>*> A2(Nh);
            std::vector<std::complex<double>*> ddx_A2(Nh);
            std::vector<std::complex<double>*> ddy_A2(Nh);

            std::vector<std::complex<double>*> ddt_phi(2);
            std::vector<std::complex<double>*> ddt_A1(2);
            std::vector<std::complex<double>*> ddt_A2(2);

            std::vector<std::complex<double>*> rho(Nh);
            std::vector<std::complex<double>*> J1(Nh);
            std::vector<std::complex<double>*> J2(Nh);

            // std::vector<std::vector<std::complex<double>>> IC(Nx, std::vector<std::complex<double>>(Ny));

            for (int h = 0; h < Nh; h++) {
                phi[h]     = new std::complex<double>[Nx*Ny];
                ddx_phi[h] = new std::complex<double>[Nx*Ny];
                ddy_phi[h] = new std::complex<double>[Nx*Ny];
                A1[h]      = new std::complex<double>[Nx*Ny];
                ddx_A1[h]  = new std::complex<double>[Nx*Ny];
                ddy_A1[h]  = new std::complex<double>[Nx*Ny];
                A2[h]      = new std::complex<double>[Nx*Ny];
                ddx_A2[h]  = new std::complex<double>[Nx*Ny];
                ddy_A2[h]  = new std::complex<double>[Nx*Ny];

                rho[h] = new std::complex<double>[Nx*Ny];
                J1[h]  = new std::complex<double>[Nx*Ny];
                J2[h]  = new std::complex<double>[Nx*Ny];

                for (int i = 0; i < Nx*Ny; i++) {
                    phi[h][i]     = 0;
                    ddx_phi[h][i] = 0;
                    ddy_phi[h][i] = 0;
                    A1[h][i]      = 0;
                    ddx_A1[h][i]  = 0;
                    ddy_A1[h][i]  = 0;
                    A2[h][i]      = 0;
                    ddx_A2[h][i]  = 0;
                    ddy_A2[h][i]  = 0;

                    rho[h][i] = 0;
                    J1[h][i]  = 0;
                    J2[h][i]  = 0;
                }
            }

            ddt_phi[0] = new std::complex<double>[Nx*Ny];
            ddt_phi[1] = new std::complex<double>[Nx*Ny];
            ddt_A1[0] = new std::complex<double>[Nx*Ny];
            ddt_A1[1] = new std::complex<double>[Nx*Ny];
            ddt_A2[0] = new std::complex<double>[Nx*Ny];
            ddt_A2[1] = new std::complex<double>[Nx*Ny];

            // Used to compute Gauss' Law
            d2dx_phi_curr = new std::complex<double>[Nx*Ny];
            d2dy_phi_curr = new std::complex<double>[Nx*Ny];
            d2dx_phi_prev = new std::complex<double>[Nx*Ny];
            d2dy_phi_prev = new std::complex<double>[Nx*Ny];

            laplacian_phi = new std::complex<double>[Nx*Ny];

            gauss_RHS = new double[Nx*Ny];

            for (int i = 0; i < Nx*Ny; i++) {
                ddt_phi[0][i] = 0;
                ddt_phi[1][i] = 0;
                ddt_A1[0][i] = 0;
                ddt_A1[1][i] = 0;
                ddt_A2[0][i] = 0;
                ddt_A2[1][i] = 0;
            }
            
            this->ddx_J1 = new std::complex<double>[Nx*Ny];
            this->ddy_J2 = new std::complex<double>[Nx*Ny];
            this->ddx_J1_prev = new std::complex<double>[Nx*Ny];
            this->ddy_J2_prev = new std::complex<double>[Nx*Ny];

            this->F1 = new double[Nx*Ny];
            this->F2 = new double[Nx*Ny];

            this->phi = phi;
            this->ddx_phi = ddx_phi;
            this->ddy_phi = ddy_phi;
            this->A1 = A1;
            this->ddx_A1 = ddx_A1;
            this->ddy_A1 = ddy_A1;
            this->A2 = A2;
            this->ddx_A2 = ddx_A2;
            this->ddy_A2 = ddy_A2;

            this->gaugeL2 = 0;
            this->gaussL2_divA = 0;
            this->gaussL2_divE = 0;
            this->gaussL2_wave = 0;

            this->kineticEnergy = 0;
            this->ionKineticEnergy = 0;
            this->eleKineticEnergy = 0;

            this->potentialEnergy = 0;
            this->ionPotentialEnergy = 0;
            this->elePotentialEnergy = 0;

            this->totalEnergy = 0;
            this->ionTotalEnergy = 0;
            this->eleTotalEnergy = 0;
            this->eleForce = 0;
            this->totalMass = 0;
            this->ionTotalMass = 0;
            this->eleTotalMass = 0;
            this->temperature = 0;
            this->totalMomentum = 0;
            this->magneticMagnitude = 0;

            this->rho = rho;
            this->J1 = J1;
            this->J2 = J2;

            this->rho_ions = new std::complex<double>[Nx*Ny];
            this->rho_eles = new std::complex<double>[Nx*Ny];

            for (int i = 0; i < Nx*Ny; i++) {
                rho_ions[i] = 0.0;
                rho_eles[i] = 0.0;
            }

            std::cout << "w_ele*q_ele/(dx*dy): " << w_ele*q_ele/(dx*dy) << std::endl;
            std::cout << "w_ion*q_ion/(dx*dy): " << w_ele*q_ele/(dx*dy) << std::endl;

            std::vector<std::vector<double>> electron_weights(1, std::vector<double>(numElectrons, w_ele*q_ele/(dx*dy)));
            std::vector<std::vector<double>> ion_weights(1, std::vector<double>(numElectrons, w_ion*q_ion/(dx*dy)));

            std::cout << "Scattering Particles" << std::endl;
            interpolate_utility->scatterParticles(&rho_eles, *x_elec[lastStepIndex], *y_elec[lastStepIndex], 1, numElectrons, electron_weights);
            interpolate_utility->scatterParticles(&rho_ions, *x_elec[lastStepIndex], *y_elec[lastStepIndex], 1, numElectrons, ion_weights);
            std::cout << "Scattered Particles" << std::endl;

            for (int i = 0; i < Nx*Ny; i++) {
                for (int h = 0; h <= lastStepIndex; h++) {
                    rho[h][i] = rho_eles[i] + rho_ions[i];
                }
            }

            for (int i = 0; i < numElectrons; i++) {
                totalMomentum += std::sqrt( std::pow((*Px_elec[lastStepIndex])[i], 2) + std::pow((*Py_elec[lastStepIndex])[i], 2) );
            }

            this->rhoTotal = 0.0;
            this->rhoElecTotal = 0.0;
            this->rhoIonsTotal = 0.0;

            for (int i = 0; i < Nx*Ny; i++) {
                rhoElecTotal += rho_eles[i].real();
                rhoIonsTotal += rho_ions[i].real();
            }
            rhoTotal = rhoElecTotal + rhoIonsTotal;

            this->phi_src = new std::complex<double>[Nx * Ny];
            this->A1_src  = new std::complex<double>[Nx * Ny];
            this->A2_src  = new std::complex<double>[Nx * Ny];

            this->ddt_phi = ddt_phi;
            this->ddt_A1 = ddt_A1;
            this->ddt_A2 = ddt_A2;

            this->E1  = new std::complex<double>[Nx * Ny];
            this->E2  = new std::complex<double>[Nx * Ny];

            this->ddx_E1  = new std::complex<double>[Nx * Ny];
            this->ddy_E2  = new std::complex<double>[Nx * Ny];

            // Begin DIRK Variables
            this->S_1 = new std::complex<double>[Nx*Ny];
            this->S_2 = new std::complex<double>[Nx*Ny];

            this->d2dx_u = new std::complex<double>[Nx*Ny];
            this->d2dy_u = new std::complex<double>[Nx*Ny];

            this->laplacian_u = new std::complex<double>[Nx*Ny];
            this->laplacian_u1 = new std::complex<double>[Nx*Ny];
            this->laplacian_u2 = new std::complex<double>[Nx*Ny];

            this->RHS1 = new std::complex<double>[Nx*Ny];
            this->RHS2 = new std::complex<double>[Nx*Ny];

            this->u1 = new std::complex<double>[Nx*Ny];
            this->v1 = new std::complex<double>[Nx*Ny];
            this->u2 = new std::complex<double>[Nx*Ny];
            this->v2 = new std::complex<double>[Nx*Ny];
            
            this->phi_src_prev = new std::complex<double>[Nx*Ny];
            this->A1_src_prev = new std::complex<double>[Nx*Ny];
            this->A2_src_prev = new std::complex<double>[Nx*Ny];

            this->ddt_phi_curr = new std::complex<double>[Nx*Ny];
            this->d2dt_phi = new std::complex<double>[Nx*Ny];
            this->ddt_A1_curr = new std::complex<double>[Nx*Ny];
            this->ddt_A2_curr = new std::complex<double>[Nx*Ny];
            this->ddt_divA_curr = new std::complex<double>[Nx*Ny];
            // End DIRK Variables

            this->timeComponent1 = 0;
            this->timeComponent2 = 0;
            this->timeComponent3 = 0;
            this->timeComponent4 = 0;
            this->timeComponent5 = 0;
            this->timeComponent6 = 0;

            this->updateMethod = updateMethod;
            this->rhoUpdate = rhoUpdate;
            this->moltMethod = moltMethod;

            if (moltMethod == MOLTMethod::Integral && (updateMethod == NumericalMethod::DIRK2 || updateMethod == NumericalMethod::DIRK2)) {
                std::cout << "Currently do not have the capability to do DIRK with Fast Convolution Method." << std::endl;
                throw -1;
            }

            this->snapshotPath = savePath;
            this->debugPath = debugPath;

            this->debugViaMatlab = debugViaMatlab;

            std::cout << "Clearing Files" << std::endl;

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
            
            std::ofstream fileClear;
            fileClear.open(electronFileName, std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(phiFileName     , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(A1FileName      , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(A2FileName      , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(ddx_phiFileName , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(ddy_phiFileName , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(ddx_A1FileName  , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(ddy_A1FileName  , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(ddx_A2FileName  , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(ddy_A2FileName  , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(rhoFileName     , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(J1FileName      , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(J2FileName      , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();
            fileClear.open(ddt_phiFileName , std::ofstream::out | std::ofstream::trunc);
            fileClear.close();

            std::cout << "Cleared Files" << std::endl;

            computePhysicalDiagnostics();
        }
        void step();
        void print();
        double getTime();
        int getStep();
        double getGaugeL2();
        double getGaussL2_divE();
        double getGaussL2_divA();
        double getGaussL2_wave();
        double getTotalCharge();
        double getElecCharge();
        double getIonsCharge();
        double getTotalMass();
        double getForce();
        double getKineticEnergy();
        double getPotentialEnergy();
        double getTotalEnergy();
        double getTemperature();
        double getMagneticMagnitude();
        double getTotalMomentum();
        void saveParticleInformation();
        void printTimeDiagnostics();

        std::vector<std::vector<std::complex<double>>> getRho() {
            std::vector<std::vector<std::complex<double>>> currRho;
            for (int i = 0; i < Nx; i++) {
                std::vector<std::complex<double>> col(Nx);
                for (int j = 0; j < Ny; j++) {
                    int idx = computeIndex(i, j);
                    col[j] = rho[lastStepIndex][idx];
                }
                currRho.push_back(col);
            }
            return currRho;
        }

        void computePhysicalDiagnostics();

    private:
        int Nx; // Number of nodes, noninclusive of right boundary
        int Ny; // Number of nodes, noninclusive of lower boundary
        int numElectrons;
        int numIons;
        int Nh;
        int lastStepIndex;
        double* x;
        double* y;
        std::vector<std::vector<double>*> x_elec;
        std::vector<std::vector<double>*> y_elec;
        std::vector<std::vector<double>*> vx_elec;
        std::vector<std::vector<double>*> vy_elec;
        std::vector<std::vector<double>*> Px_elec;
        std::vector<std::vector<double>*> Py_elec;
        std::vector<double> x_ion;
        std::vector<double> y_ion;
        std::vector<double> vx_ion;
        std::vector<double> vy_ion;
        std::vector<double> Px_ion;
        std::vector<double> Py_ion;
        double gaugeL2;
        double gaussL2_divE;
        double gaussL2_wave;
        double gaussL2_divA;
        double rhoElecTotal;
        double rhoIonsTotal;
        double rhoTotal;
        double kineticEnergy;
        double ionKineticEnergy;
        double eleKineticEnergy;
        double potentialEnergy;
        double ionPotentialEnergy;
        double elePotentialEnergy;
        double totalEnergy;
        double ionTotalEnergy;
        double eleTotalEnergy;
        double eleForce;
        double totalMass;
        double ionTotalMass;
        double eleTotalMass;
        double temperature;
        double totalMomentum;
        double magneticMagnitude;
        double dx;
        double dy;
        double dt;
        double t;
        int n;
        double kappa;
        double sigma_1;
        double sigma_2;
        double beta;
        double q_ele;
        double m_ele;
        double q_ion;
        double m_ion;
        double w_ele;
        double w_ion;
        std::vector<std::complex<double>*> phi;
        std::vector<std::complex<double>*> ddx_phi;
        std::vector<std::complex<double>*> ddy_phi;
        std::vector<std::complex<double>*> A1;
        std::vector<std::complex<double>*> ddx_A1;
        std::vector<std::complex<double>*> ddy_A1;
        std::vector<std::complex<double>*> A2;
        std::vector<std::complex<double>*> ddx_A2;
        std::vector<std::complex<double>*> ddy_A2;
        std::vector<std::complex<double>*> rho;
        std::vector<std::complex<double>*> J1;
        std::vector<std::complex<double>*> J2;
        std::vector<std::complex<double>*> ddt_phi;
        std::vector<std::complex<double>*> ddt_A1;
        std::vector<std::complex<double>*> ddt_A2;
        std::complex<double>* ddx_J1;
        std::complex<double>* ddy_J2;
        std::complex<double>* ddx_J1_prev;
        std::complex<double>* ddy_J2_prev;
        std::complex<double>* phi_src;
        std::complex<double>* A1_src;
        std::complex<double>* A2_src;
        std::complex<double>* E1;
        std::complex<double>* E2;
        std::complex<double>* ddx_E1;
        std::complex<double>* ddy_E2;
        std::complex<double>* d2dt_phi;
        std::complex<double>* rho_ions;
        std::complex<double>* rho_eles;
        double* F1;
        double* F2;
        double* gauss_RHS;

        // ============= Variables for DIRK methods ===============
        std::complex<double>* S_1;
        std::complex<double>* S_2;
        std::complex<double>* d2dx_u;
        std::complex<double>* d2dy_u;
        std::complex<double>* laplacian_u;
        std::complex<double>* laplacian_u1;
        std::complex<double>* laplacian_u2;
        std::complex<double>* d2dx_phi_curr;
        std::complex<double>* d2dy_phi_curr;
        std::complex<double>* d2dx_phi_prev;
        std::complex<double>* d2dy_phi_prev;
        std::complex<double>* laplacian_phi;
        std::complex<double>* RHS1;
        std::complex<double>* RHS2;
        std::complex<double>* u1;
        std::complex<double>* v1;
        std::complex<double>* u2;
        std::complex<double>* v2;
        std::complex<double>* phi_src_prev;
        std::complex<double>* A1_src_prev;
        std::complex<double>* A2_src_prev;
        std::complex<double>* ddt_phi_curr;
        std::complex<double>* ddt_A1_curr;
        std::complex<double>* ddt_A2_curr;
        std::complex<double>* ddt_divA_curr;
        // ============= Variables for DIRK methods ===============

        bool correctTheGauge;
        bool debugViaMatlab;
        MOLTEngine::NumericalMethod updateMethod;
        MOLTEngine::RhoUpdate rhoUpdate;
        MOLTEngine::MOLTMethod moltMethod;
        std::string snapshotPath;
        std::string debugPath;

        Derivative* derivative_utility;
        Interpolate* interpolate_utility;

        // ============= Variables for pointers to delete in shuffle method ===============
        std::vector<double>* x_elec_dlt_ptr;
        std::vector<double>* y_elec_dlt_ptr;
        std::vector<double>* vx_elec_dlt_ptr;
        std::vector<double>* vy_elec_dlt_ptr;
        std::vector<double>* Px_elec_dlt_ptr;
        std::vector<double>* Py_elec_dlt_ptr;
        std::complex<double>* phi_dlt_ptr;
        std::complex<double>* ddx_phi_dlt_ptr;
        std::complex<double>* ddy_phi_dlt_ptr;
        std::complex<double>* A1_dlt_ptr;
        std::complex<double>* ddx_A1_dlt_ptr;
        std::complex<double>* ddy_A1_dlt_ptr;
        std::complex<double>* A2_dlt_ptr;
        std::complex<double>* ddx_A2_dlt_ptr;
        std::complex<double>* ddy_A2_dlt_ptr;
        std::complex<double>* rho_dlt_ptr;
        std::complex<double>* J1_dlt_ptr;
        std::complex<double>* J2_dlt_ptr;
        std::complex<double>* ddt_phi_dlt_ptr;
        std::complex<double>* ddt_A1_dlt_ptr;
        std::complex<double>* ddt_A2_dlt_ptr;
        // ============= Variables for pointers to delete in shuffle method ===============

        double timeComponent1, timeComponent2, timeComponent3, timeComponent4, timeComponent5, timeComponent6, timeComponent7;

        void debug();
        void computeForce();
        void computeGaugeL2();
        void computeGaussL2();
        void computeTotalEnergy();
        void computeTotalMass();
        void computeTemperature();
        void computeMagneticMagnitude();
        void computeTotalMomentum();
        void correctGauge();
        void updateParticleLocations();
        void updateParticleVelocities();
        void scatterFields();
        void updateWaves();
        void DIRK2_advance_per(std::complex<double>* u, std::complex<double>* v,
                               std::complex<double>* u_next, std::complex<double>* v_next,
                               std::complex<double>* src_prev, std::complex<double>* src_curr);
        void DIRK3_advance_per(std::complex<double>* u, std::complex<double>* v,
                               std::complex<double>* u_next, std::complex<double>* v_next,
                               std::complex<double>* src_prev, std::complex<double>* src_curr);
        void MOLT_BDF1_advance_per(std::vector<std::complex<double>*> input_field_hist, std::complex<double>* RHS, std::complex<double>* output);
        void MOLT_BDF1_combined_per_advance(std::vector<std::complex<double>*> u, std::complex<double>* RHS,
                                            std::complex<double>* u_out, std::complex<double>* dudx_out, std::complex<double>* dudy_out);
        void MOLT_BDF1_ddx_advance_per(std::vector<std::complex<double>*> input_field_hist, std::complex<double>* RHS, std::complex<double>* output);
        void MOLT_BDF1_ddy_advance_per(std::vector<std::complex<double>*> input_field_hist, std::complex<double>* RHS, std::complex<double>* output);
        void MOLT_combined_per_advance(std::vector<std::complex<double>*> u, std::complex<double>* RHS,
                                       std::complex<double>* u_out, std::complex<double>* dudx_out, std::complex<double>* dudy_out);
        void MOLT_advance_per(std::complex<double>* RHS, std::complex<double>* output);
        void MOLT_ddx_advance_per(std::complex<double>* RHS, std::complex<double>* output);
        void MOLT_ddy_advance_per(std::complex<double>* RHS, std::complex<double>* output);
        void get_L_x_inverse_per(std::complex<double>* u, std::complex<double>* inverse);
        void get_L_y_inverse_per(std::complex<double>* u, std::complex<double>* inverse);
        void get_ddy_L_y_inverse_per(std::complex<double>* u, std::complex<double>* ddyOut);
        void get_ddx_L_x_inverse_per(std::complex<double>* u, std::complex<double>* ddyOut);
        void linear5_L(std::vector<std::complex<double>> v_ext, double gamma, std::vector<std::complex<double>>& J_L);
        void linear5_R(std::vector<std::complex<double>> v_ext, double gamma, std::vector<std::complex<double>>& J_R);
        void fast_convolution(std::vector<std::complex<double>> &I_L, std::vector<std::complex<double>> &I_R, double alpha);
        void apply_A_and_B(std::vector<std::complex<double>> &I_, double* x, double dx, int N, double alpha, double A, double B, bool debug=false);
        double gatherField(double p_x, double p_y, std::complex<double>* field);
        void gatherFields(double p_x, double p_y, std::vector<std::vector<std::complex<double>>>& fields, std::vector<double>& fields_out);
        void scatterField(double p_x, double p_y, double value, std::complex<double>* field);
        void shuffleSteps();
        void updatePhi();
        void updateA1();
        void updateA2();
        std::vector<double> compute_wave_numbers(int N, double L);

        int computeIndex(int i, int j) {
            return j*Nx + i;
        }

        void compute_ddx_numerical(std::complex<double>* inputField, std::complex<double>* derivativeField) {
            derivative_utility->computeFirstDerivative(inputField, derivativeField, true);
        }

        void compute_ddy_numerical(std::complex<double>* inputField, std::complex<double>* derivativeField) {
            derivative_utility->computeFirstDerivative(inputField, derivativeField, false);
        }

        void compute_d2dx_numerical(std::complex<double>* inputField, std::complex<double>* derivativeField) {
            derivative_utility->computeSecondDerivative(inputField, derivativeField, true);
        }

        void compute_d2dy_numerical(std::complex<double>* inputField, std::complex<double>* derivativeField) {
            derivative_utility->computeSecondDerivative(inputField, derivativeField, false);
        }

        void solveHelmholtzEquation(std::complex<double>* RHS,
                                    std::complex<double>* LHS, double alpha) {
            derivative_utility->solveHelmholtzEquation(RHS, LHS, alpha); 
        }

        double dirk_qin_zhang_rhs(double rhs_prev, double rhs_curr) {
            double b1 = .5;  // 1/2;
            double b2 = .5;  // 1/2;

            double c1 = .25; // 1/4;
            double c2 = .75; // 3/4;

            double RHS_1 = (1-c1)*rhs_prev + c1*rhs_curr;
            double RHS_2 = (1-c2)*rhs_prev + c2*rhs_curr;

            return b1*RHS_1 + b2*RHS_2;
        }

        std::complex<double> to_std_complex(const fftw_complex& fc) {
            return std::complex<double>(fc[0], fc[1]);
        }

        // Helper function to compute the wave numbers in one dimension
        std::vector<double> compute_wave_numbers(int N, double L, bool first_derivative = true) {
            std::vector<double> k(N);
            double dk = 2 * M_PI / L;
            for (int i = 0; i < N / 2; ++i) {
                k[i] = i * dk;
            }
            k[N / 2] = first_derivative ? 0 : (N/2)*dk;
            for (int i = N / 2 + 1; i < N; ++i) {
                k[i] = (i - N) * dk;
            }
            return k;
        }

        void BDF1_advance_per(std::complex<double>* input_field, std::complex<double>* RHS);

        void debugLocations();
        void debugScatteredFields();
        void debugComputedFields();
        void debugVelocities();
        void debugMomentums();
};