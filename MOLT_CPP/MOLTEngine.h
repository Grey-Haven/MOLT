#include <cmath>
#include <iostream>
#include <fstream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <complex.h>
#include <fftw3.h>

class MOLTEngine {

    public:
        MOLTEngine(int Nx, int Ny, int Np, int Nh, double* x, double* y,
                   std::vector<std::vector<double>>& x_elec, std::vector<std::vector<double>>& y_elec,
                   std::vector<std::vector<double>>& vx_elec, std::vector<std::vector<double>>& vy_elec,
                   double dx, double dy, double dt, double sigma_1, double sigma_2, double kappa, double q_elec, double m_elec, double beta) {
            this->Nx = Nx;
            this->Ny = Ny;
            this->Np = Np;
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
            this->elec_charge = q_elec;
            this->elec_mass = m_elec;
            this->beta = beta;
            this->t = 0;
            this->n = 0;

            this->x_elec = x_elec;
            this->y_elec = y_elec;
            this->vx_elec = vx_elec;
            this->vy_elec = vy_elec;

            this->w_elec = 10*((x[Nx-1] - x[0])*(y[Ny-1] - y[0]))/Np;

            std::vector<std::vector<double>> Px_elec(Nh, std::vector<double>(Np));
            std::vector<std::vector<double>> Py_elec(Nh, std::vector<double>(Np));

            for (int h = 0; h < Nh; h++) {
                for (int i = 0; i < Np; i++) {
                    Px_elec[h][i] = elec_mass * vx_elec[h][i];
                    Py_elec[h][i] = elec_mass * vy_elec[h][i];
                }
            }

            this->Px_elec = Px_elec;
            this->Py_elec = Py_elec;

            std::vector<std::vector<std::vector<std::complex<double>>>> phi(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> ddx_phi(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> ddy_phi(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> A1(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> ddx_A1(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> ddy_A1(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> A2(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> ddx_A2(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> ddy_A2(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));

            std::vector<std::vector<std::vector<std::complex<double>>>> rho(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> J1(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            std::vector<std::vector<std::vector<std::complex<double>>>> J2(Nh, std::vector<std::vector<std::complex<double>>>(Nx, std::vector<std::complex<double>>(Ny)));
            
            std::vector<std::vector<std::complex<double>>> ddx_J1(Nx, std::vector<std::complex<double>>(Ny));
            std::vector<std::vector<std::complex<double>>> ddy_J2(Nx, std::vector<std::complex<double>>(Ny));

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

            this->rho = rho;
            this->J1 = J1;
            this->J2 = J2;

            this->ddx_J1 = ddx_J1;
            this->ddy_J2 = ddy_J2;
        }
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
        void gatherFields();
        void shuffleSteps();
        void updatePhi();
        void updateA1();
        void updateA2();
        std::complex<double> to_std_complex(const fftw_complex& fc);
        std::vector<double> compute_wave_numbers(int N, double L);
        void compute_derivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                        std::vector<std::vector<std::complex<double>>>& derivative_field, bool derivative_in_x);
        void compute_second_derivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                std::vector<std::vector<std::complex<double>>>& derivative_field, bool derivative_in_x);

        void compute_ddx(const std::vector<std::vector<std::complex<double>>>& input_field, 
                                std::vector<std::vector<std::complex<double>>>& derivative_field,
                                int Nx, int Ny, double Lx, double Ly) {
                                    compute_derivative(input_field, derivative_field, true);
                                }
        void compute_ddy(const std::vector<std::vector<std::complex<double>>>& input_field, 
                                std::vector<std::vector<std::complex<double>>>& derivative_field,
                                int Nx, int Ny, double Lx, double Ly) {
                                    compute_derivative(input_field, derivative_field, false);
                                }
        void compute_d2dx(const std::vector<std::vector<std::complex<double>>>& input_field, 
                                std::vector<std::vector<std::complex<double>>>& derivative_field,
                                int Nx, int Ny, double Lx, double Ly) {
                                    compute_second_derivative(input_field, derivative_field, true);
                                }
        void compute_d2dy(const std::vector<std::vector<std::complex<double>>>& input_field, 
                                std::vector<std::vector<std::complex<double>>>& derivative_field,
                                int Nx, int Ny, double Lx, double Ly) {
                                    compute_second_derivative(input_field, derivative_field, false);
                                }

};