#include <cmath>
#include <iostream>
#include <fstream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <complex.h>
#include <fftw3.h>
#include <vector>

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
        double gatherField(double p_x, double p_y, std::vector<std::vector<std::complex<double>>>& field);
        void scatterField(double p_x, double p_y, double value, std::vector<std::vector<std::complex<double>>>& field);

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
        double* kx_deriv_1;
        double* ky_deriv_1;
        double* kx_deriv_2;
        double* ky_deriv_2;
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
        void computeGaugeL2();
        void updateParticleLocations();
        void updateParticleVelocities();
        void scatterFields();
        void updateWaves();
        void shuffleSteps();
        void updatePhi();
        void updateA1();
        void updateA2();

        void compute_derivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                        std::vector<std::vector<std::complex<double>>>& derivative_field, bool derivative_in_x);
        void compute_second_derivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                std::vector<std::vector<std::complex<double>>>& derivative_field, bool derivative_in_x);
        
        void solveHelmholtzEquation(std::vector<std::vector<std::complex<double>>>& RHS,
                                                std::vector<std::vector<std::complex<double>>>& LHS, double alpha);

        void compute_ddx(const std::vector<std::vector<std::complex<double>>>& input_field, 
                               std::vector<std::vector<std::complex<double>>>& derivative_field) {
                                    compute_derivative(input_field, derivative_field, true);
                               }
        void compute_ddy(const std::vector<std::vector<std::complex<double>>>& input_field, 
                               std::vector<std::vector<std::complex<double>>>& derivative_field) {
                                    compute_derivative(input_field, derivative_field, false);
                               }
        void compute_d2dx(const std::vector<std::vector<std::complex<double>>>& input_field, 
                                std::vector<std::vector<std::complex<double>>>& derivative_field) {
                                    compute_second_derivative(input_field, derivative_field, true);
                                }
        void compute_d2dy(const std::vector<std::vector<std::complex<double>>>& input_field, 
                                std::vector<std::vector<std::complex<double>>>& derivative_field) {
                                    compute_second_derivative(input_field, derivative_field, false);
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
};

/**
 * Name: step
 * Author: Stephen White
 * Date Created: 9/28/22
 * Date Last Modified: 9/28/22 (Stephen White)
 * Description: 2D update equation derived from the curl equations in Maxwell's Equations (TE mode of the Yee Scheme)
 * Inputs: NA
 * Output: NA
 * Dependencies: scatterFields, shuffleSteps, updateParticleLocations, updateParticleVelocities, updateWaves
 */
void MOLTEngine::step() {
    // for (int i = 0; i < 10; i++) {
    //     std::cout << x_elec[lastStepIndex-1][i] << ", " << y_elec[lastStepIndex-1][i] << ", " << vx_elec[lastStepIndex-1][i] << ", " << vy_elec[lastStepIndex-1][i] << std::endl;
    // }
    // std::cout << "Updating Particle Locations" << std::endl;
    updateParticleLocations();
    // for (int i = 0; i < 10; i++) {
    //     std::cout << x_elec[lastStepIndex][i] << ", " << y_elec[lastStepIndex][i] << ", " << vx_elec[lastStepIndex][i] << ", " << vy_elec[lastStepIndex][i] << std::endl;
    // }
    // std::cout << "Scattering Fields" << std::endl;
    scatterFields();
    // std::cout << "Updating Waves" << std::endl;
    updateWaves();
    // std::cout << "Updating Particle Velocities" << std::endl;
    updateParticleVelocities();
    // std::cout << "Shuffling Steps" << std::endl;
    computeGaugeL2();
    shuffleSteps();
    // std::cout << "Rinse, Repeat" << std::endl;
    if (n % 100 == 0) {
        print();
    }
    n++;
    t += dt;
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

void MOLTEngine::updateParticleLocations() {
    double Lx = x[Nx-1] - x[0];
    double Ly = y[Ny-1] - y[0];

    for (int i = 0; i < this->Np; i++) {
        double vx_star = 2.0*this->vx_elec[lastStepIndex-1][i] - this->vx_elec[lastStepIndex-2][i];
        double vy_star = 2.0*this->vy_elec[lastStepIndex-1][i] - this->vy_elec[lastStepIndex-2][i];

        this->x_elec[lastStepIndex][i] = this->x_elec[lastStepIndex-1][i] + dt*vx_star;
        this->y_elec[lastStepIndex][i] = this->y_elec[lastStepIndex-1][i] + dt*vy_star;

        this->x_elec[lastStepIndex][i] = this->x_elec[lastStepIndex][i] - Lx*floor((this->x_elec[lastStepIndex][i] - this->x[0]) / Lx);
        this->y_elec[lastStepIndex][i] = this->y_elec[lastStepIndex][i] - Ly*floor((this->y_elec[lastStepIndex][i] - this->y[0]) / Ly);
    }
    // for (int i = 0; i < 100; i++) {
    //     double vx_star = 2.0*this->vx_elec[lastStepIndex-1][i] - this->vx_elec[lastStepIndex-2][i];
    //     double vy_star = 2.0*this->vy_elec[lastStepIndex-1][i] - this->vy_elec[lastStepIndex-2][i];
    //     std::cout << vx_star << ", " << vy_star << std::endl;
    // }
    // std::cout << "===========" << std::endl;
    // for (int i = 0; i < 100; i++) {
    //     std::cout << vx_elec[lastStepIndex-1][i] << ", " << vx_elec[lastStepIndex-2][i] << std::endl;
    // }
    // for (int i = 0; i < 100; i++) {
    //     std::cout << x_elec[lastStepIndex][i] << ", " << y_elec[lastStepIndex][i] << std::endl;
    // }
}

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

    // std::cout << alpha << std::endl;
    // std::cout << "[";
    // for (int i = 0; i < Nx; i++) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; j++) {
    //         std::cout << rho[lastStepIndex][i][j].real() << ", ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "]" << std::endl;
    // std::cout << "====================" << std::endl;
    // std::cout << "[";
    // for (int i = 0; i < Nx; i++) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; j++) {
    //         std::cout << phi_src[i][j].real() << ", ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "]" << std::endl;

    solveHelmholtzEquation(phi_src, this->phi[lastStepIndex], alpha);
    solveHelmholtzEquation(A1_src,  this->A1[lastStepIndex], alpha);
    solveHelmholtzEquation(A2_src,  this->A2[lastStepIndex], alpha);
    // std::cout << "[";
    // for (int i = 0; i < Nx; i++) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; j++) {
    //         std::cout << A2[lastStepIndex][i][j].real() << " + " << A2[lastStepIndex][i][j].imag() << "i, ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "]" << std::endl;

    compute_ddx(this->phi[lastStepIndex], this->ddx_phi[lastStepIndex]);
    compute_ddy(this->phi[lastStepIndex], this->ddy_phi[lastStepIndex]);
    compute_ddx(this->A1[lastStepIndex],  this->ddx_A1[lastStepIndex]);
    compute_ddy(this->A1[lastStepIndex],  this->ddy_A1[lastStepIndex]);
    compute_ddx(this->A2[lastStepIndex],  this->ddx_A2[lastStepIndex]);
    compute_ddy(this->A2[lastStepIndex],  this->ddy_A2[lastStepIndex]);
}

void MOLTEngine::updateParticleVelocities() {
    for (int i = 0; i < Np; i++) {
        double ddx_phi_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], ddx_phi[lastStepIndex]);
        double ddy_phi_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], ddy_phi[lastStepIndex]);

        double A1_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], A1[lastStepIndex]);
        double ddx_A1_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], ddx_A1[lastStepIndex]);
        double ddy_A1_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], ddy_A1[lastStepIndex]);

        double A2_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], A2[lastStepIndex]);
        double ddx_A2_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], ddx_A2[lastStepIndex]);
        double ddy_A2_p = gatherField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], ddy_A2[lastStepIndex]);

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
    // for (int i = 0; i < 10; i++) {
    //     std::cout << Px_elec[lastStepIndex][i] << " " << Py_elec[lastStepIndex][i] << std::endl;
    // }
}

void MOLTEngine::solveHelmholtzEquation(std::vector<std::vector<std::complex<double>>>& RHS,
                                        std::vector<std::vector<std::complex<double>>>& LHS, double alpha) {

    int Nx = this->Nx - 1;
    int Ny = this->Ny - 1;
    double Lx = this->x[Nx-1] - this->x[0];
    double Ly = this->y[Ny-1] - this->y[0];

    // Flatten the 2D input field into a 1D array for FFTW
    std::vector<std::complex<double>> in(Nx * Ny);
    std::vector<std::complex<double>> out(Nx * Ny);
    std::vector<std::complex<double>> derivative(Nx * Ny);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            in[i * Ny + j] = RHS[i][j];
        }
    }

    // Create FFTW plans for the forward and inverse FFT
    fftw_plan forward_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(in.data()), 
        reinterpret_cast<fftw_complex*>(out.data()), 
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(out.data()), 
        reinterpret_cast<fftw_complex*>(derivative.data()), 
        FFTW_BACKWARD, FFTW_ESTIMATE);
        
    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the wave numbers in the appropriate direction
    std::vector<double> kx = compute_wave_numbers(Nx, Lx, false); 
    std::vector<double> ky = compute_wave_numbers(Ny, Ly, false);

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(out.data())[index]);
            double k_val =  kx[i]*kx[i] + ky[j]*ky[j];
            freq_component /= (1 + 1/(alpha*alpha) * k_val); // Invert the helmholtz operator (I - (d^2/dx^2 + d^2/dy^2)) ==Fourier==> (I + (kx^2 + ky^2)))
            reinterpret_cast<fftw_complex*>(out.data())[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(out.data())[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            derivative[i * Ny + j] /= (Nx * Ny);
            LHS[i][j] = derivative[i * Ny + j];
        }
    }
    for (int i = 0; i < this->Nx; i++) {
        LHS[i][this->Ny-1] = LHS[i][0];
    }
    for (int j = 0; j < this->Ny; j++) {
        LHS[this->Nx-1][j] = LHS[0][j];
    }

    // Clean up FFTW plans
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
}

void MOLTEngine::scatterFields() {

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            J1[lastStepIndex][i][j] = 0;
            J2[lastStepIndex][i][j] = 0;
        }
    }

    for (int i = 0; i < Np; i++) {
        double vx_star = 2.0*this->vx_elec[lastStepIndex-1][i] - this->vx_elec[lastStepIndex-2][i];
        double vy_star = 2.0*this->vy_elec[lastStepIndex-1][i] - this->vy_elec[lastStepIndex-2][i];

        double x_value = this->elec_charge*vx_star*this->w_elec;
        double y_value = this->elec_charge*vy_star*this->w_elec;

        // std::cout << x_elec[lastStepIndex][i] << std::endl;

        scatterField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], x_value, this->J1[lastStepIndex]);
        scatterField(x_elec[lastStepIndex][i], y_elec[lastStepIndex][i], y_value, this->J2[lastStepIndex]);
    }
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            J1[lastStepIndex][i][j] /= dx*dy;
            J2[lastStepIndex][i][j] /= dx*dy;
        }
    }
    // std::cout << this->elec_charge*this->w_elec << std::endl;
    // std::cout << dx*dy << std::endl;
    // std::cout << "[";
    // for (int i = 0; i < Nx; i++) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; j++) {
    //         std::cout << J1[lastStepIndex][i][j].real() << ", ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "]" << std::endl;

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
            // div_J[i][j] = ddx_J1[i][j] + ddy_J2[i][j];
            rho[lastStepIndex][i][j] = this->rho[lastStepIndex-1][i][j] - dt*(ddx_J1[i][j] + ddy_J2[i][j]);
        }
    }
    // std::cout << "[";
    // for (int i = 0; i < Nx; i++) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; j++) {
    //         // std::cout << std::setprecision (15) << rho[lastStepIndex][i][j].real() << " + " << rho[lastStepIndex][i][j].imag() << "i, ";
    //         std::cout << std::setprecision (15) << rho[lastStepIndex][i][j].real() << ", ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "]" << std::endl;
    // std::cout << "[";
    // for (int i = 0; i < Nx; i++) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; j++) {
    //         std::cout << ddx_J1[i][j].real() << " + " << ddx_J1[i][j].imag() << "i, ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "]" << std::endl;
    // std::cout << "[";
    // for (int i = 0; i < Nx; i++) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; j++) {
    //         std::cout << ddy_J2[i][j].real() << " + " << ddy_J2[i][j].imag() << "i, ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // std::cout << "]" << std::endl;
}

void MOLTEngine::shuffleSteps() {
    for (int h = 0; h < lastStepIndex; h++) {
        for (int i = 0; i < Np; i++) {
            this->x_elec[h][i] = this->x_elec[h+1][i];
            this->y_elec[h][i] = this->y_elec[h+1][i];
            this->vx_elec[h][i] = this->vx_elec[h+1][i];
            this->vy_elec[h][i] = this->vy_elec[h+1][i];
            this->Px_elec[h][i] = this->Px_elec[h+1][i];
            this->Py_elec[h][i] = this->Py_elec[h+1][i];
        }
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                this->phi[h][i][j] = this->phi[h+1][i][j];
                this->ddx_phi[h][i][j] = this->ddx_phi[h+1][i][j];
                this->ddy_phi[h][i][j] = this->ddy_phi[h+1][i][j];
                this->A1[h][i][j] = this->A1[h+1][i][j];
                this->ddx_A1[h][i][j] = this->ddx_A1[h+1][i][j];
                this->ddy_A1[h][i][j] = this->ddy_A1[h+1][i][j];
                this->A2[h][i][j] = this->A2[h+1][i][j];
                this->ddx_A2[h][i][j] = this->ddx_A2[h+1][i][j];
                this->ddy_A2[h][i][j] = this->ddy_A2[h+1][i][j];

                this->rho[h][i][j] = this->rho[h+1][i][j];
                this->J1[h][i][j] = this->J1[h+1][i][j];
                this->J2[h][i][j] = this->J2[h+1][i][j];
            }
        }
    }
}

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

    // std::cout << p_x << ", " << p_y << std::endl;
    // std::cout << lc_x << ", " << lc_y << std::endl;
    // std::cout << fx << ", " << fy << std::endl;
    // std::cout << xNode << ", " << yNode << std::endl;
    // std::cout << p_x << " - " << xNode << " = " << (p_x - xNode) << ", " << p_y << " - " << yNode << " = " << (p_y - yNode) << std::endl;
    // std::cout << "=================" << std::endl;

    // Now we acquire the particle value and add it to the corresponding field
    field[lc_x][lc_y]     += (1-fx)*(1-fy)*value;
    field[lc_x][lc_y+1]   += (1-fx)*(fy)*value;
    field[lc_x+1][lc_y]   += (fx)*(1-fy)*value;
    field[lc_x+1][lc_y+1] += (fx)*(fy)*value;
}

std::complex<double> to_std_complex(const fftw_complex& fc) {
    return std::complex<double>(fc[0], fc[1]);
}

// Helper function to compute the wave numbers in one dimension
std::vector<double> compute_wave_numbers(int N, double L) {
    std::vector<double> k(N);
    double dk = 2 * M_PI / L;
    for (int i = 0; i <= N / 2; ++i) {
        k[i] = i * dk;
    }
    for (int i = N / 2 + 1; i < N; ++i) {
        k[i] = (i - N) * dk;
    }
    return k;
}

void MOLTEngine::compute_derivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                                    std::vector<std::vector<std::complex<double>>>& derivative_field,
                                    bool derivative_in_x) {

    int Nx = this->Nx - 1;
    int Ny = this->Ny - 1;
    double Lx = this->x[Nx-1] - this->x[0];
    double Ly = this->y[Ny-1] - this->y[0];

    // Flatten the 2D input field into a 1D array for FFTW
    std::vector<std::complex<double>> in(Nx * Ny);
    std::vector<std::complex<double>> out(Nx * Ny);
    std::vector<std::complex<double>> derivative(Nx * Ny);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            in[i * Ny + j] = input_field[i][j];
        }
    }

    // Create FFTW plans for the forward and inverse FFT
    fftw_plan forward_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(in.data()), 
        reinterpret_cast<fftw_complex*>(out.data()), 
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(out.data()), 
        reinterpret_cast<fftw_complex*>(derivative.data()), 
        FFTW_BACKWARD, FFTW_ESTIMATE);

    // std::cout << "BEFORE" << std::endl;
    
    // std::cout << "[";
    // for (int i = 0; i < Nx; ++i) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; ++j) {
    //         double re = input_field.data()[i][j].real();
    //         double im = input_field.data()[i][j].imag();
    //         std::cout << re << " + " << im << "i" << ", ";
    //     }
    //     std::cout << "];" << std::endl;
    // }
    // std::cout << "]" << std::endl;

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // std::cout << "Transformed: " << std::endl;
    // std::cout << "[";
    // for (int i = 0; i < Nx; ++i) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; ++j) {
    //         int index = i * Ny + j;
    //         double re = reinterpret_cast<fftw_complex*>(out.data())[index][0];
    //         double im = reinterpret_cast<fftw_complex*>(out.data())[index][1];
    //         std::cout << re << " + " << im << "i" << ", ";
    //     }
    //     std::cout << "];" << std::endl;
    // }
    // std::cout << "]" << std::endl;

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = derivative_in_x ? compute_wave_numbers(Nx, Lx) : compute_wave_numbers(Ny, Ly);

    // Apply the derivative operator in the frequency domain
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(out.data())[index]);
            if (derivative_in_x) {
                freq_component *= std::complex<double>(0, k[i]); // Multiply by i * kx
            } else {
                freq_component *= std::complex<double>(0, k[j]); // Multiply by i * ky
            }
            reinterpret_cast<fftw_complex*>(out.data())[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(out.data())[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            derivative[i * Ny + j] /= (Nx * Ny);
            derivative_field[i][j] = derivative[i * Ny + j];
        }
    }

    for (int i = 0; i < this->Nx; i++) {
        derivative_field[i][this->Ny-1] = derivative_field[i][0];
    }
    for (int j = 0; j < this->Ny; j++) {
        derivative_field[this->Nx-1][j] = derivative_field[0][j];
    }

    // std::cout << "Derivative: " << std::endl;
    // std::cout << "[";
    // for (int i = 0; i < Nx; ++i) {
    //     std::cout << "[";
    //     for (int j = 0; j < Ny; ++j) {
    //         int index = i * Ny + j;
    //         double re = derivative_field[i][j].real();
    //         double im = derivative_field[i][j].imag();
    //         std::cout << re << " + " << im << "i" << ", ";
    //     }
    //     std::cout << "];" << std::endl;
    // }
    // std::cout << "]" << std::endl;

    // Clean up FFTW plans
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
}

void MOLTEngine::compute_second_derivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                               std::vector<std::vector<std::complex<double>>>& derivative_field,
                               bool derivative_in_x) {

    int Nx = this->Nx - 1;
    int Ny = this->Ny - 1;
    double Lx = this->x[Nx-1] - this->x[0];
    double Ly = this->y[Ny-1] - this->y[0];

    // Flatten the 2D input field into a 1D array for FFTW
    std::vector<std::complex<double>> in(Nx * Ny);
    std::vector<std::complex<double>> out(Nx * Ny);
    std::vector<std::complex<double>> derivative(Nx * Ny);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            in[i * Ny + j] = input_field[i][j];
        }
    }

    // Create FFTW plans for the forward and inverse FFT
    fftw_plan forward_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(in.data()), 
        reinterpret_cast<fftw_complex*>(out.data()), 
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_2d(Nx, Ny, 
        reinterpret_cast<fftw_complex*>(out.data()), 
        reinterpret_cast<fftw_complex*>(derivative.data()), 
        FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the wave numbers in the appropriate direction
    std::vector<double> k = derivative_in_x ? compute_wave_numbers(Nx, Lx, false) : compute_wave_numbers(Ny, Ly, false);

    // Apply the second derivative operator in the frequency domain
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int index = i * Ny + j;
            std::complex<double> freq_component = to_std_complex(reinterpret_cast<fftw_complex*>(out.data())[index]);
            double k_val = derivative_in_x ? k[i] : k[j];
            freq_component *= -k_val * k_val; // Multiply by -k^2
            reinterpret_cast<fftw_complex*>(out.data())[index][0] = freq_component.real();
            reinterpret_cast<fftw_complex*>(out.data())[index][1] = freq_component.imag();
        }
    }

    // Execute the inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse FFT output
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            derivative[i * Ny + j] /= (Nx * Ny);
            derivative_field[i][j] = derivative[i * Ny + j];
        }
    }

    for (int i = 0; i < this->Nx; i++) {
        derivative_field[i][this->Ny-1] = derivative_field[i][0];
    }
    for (int j = 0; j < this->Ny; j++) {
        derivative_field[this->Nx-1][j] = derivative_field[0][j];
    }

    // Clean up FFTW plans
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
}