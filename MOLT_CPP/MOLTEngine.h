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

class MOLTEngine {

    public:
        MOLTEngine(int Nx, int Ny, int Np, int Nh, double* x, double* y,
                   std::vector<std::vector<double>*>& x_elec, std::vector<std::vector<double>*>& y_elec,
                   std::vector<std::vector<double>*>& vx_elec, std::vector<std::vector<double>*>& vy_elec,
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

            std::vector<std::vector<double>*> Px_elec(Nh);
            std::vector<std::vector<double>*> Py_elec(Nh);

            for (int h = 0; h < Nh; h++) {
                Px_elec[h] = new std::vector<double>(Np);
                Py_elec[h] = new std::vector<double>(Np);
                for (int i = 0; i < Np; i++) {
                    (*Px_elec[h])[i] = elec_mass * (*vx_elec[h])[i];
                    (*Py_elec[h])[i] = elec_mass * (*vy_elec[h])[i];
                }
            }

            this->Px_elec = Px_elec;
            this->Py_elec = Py_elec;

            std::vector<std::vector<std::complex<double>>>** phi = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** ddx_phi = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** ddy_phi = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** A1 = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** ddx_A1 = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** ddy_A1 = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** A2 = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** ddx_A2 = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** ddy_A2 = new std::vector<std::vector<std::complex<double>>>*[Nh];

            std::vector<std::vector<std::complex<double>>>** rho = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** J1 = new std::vector<std::vector<std::complex<double>>>*[Nh];
            std::vector<std::vector<std::complex<double>>>** J2 = new std::vector<std::vector<std::complex<double>>>*[Nh];

            // std::vector<std::vector<std::complex<double>>> IC(Nx, std::vector<std::complex<double>>(Ny));

            for (int h = 0; h < Nh; h++) {
                phi[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                ddx_phi[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                ddy_phi[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                A1[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                ddx_A1[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                ddy_A1[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                A2[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                ddx_A2[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                ddy_A2[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));

                rho[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                J1[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
                J2[h] = new std::vector<std::vector<std::complex<double>>>(Ny, std::vector<std::complex<double>>(Nx, 0));
            }
            
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

            // std::vector<std::vector<std::vector<std::complex<double>>>> fields{ddx_phi[lastStepIndex], ddy_phi[lastStepIndex], A1[lastStepIndex], ddx_A1[lastStepIndex], ddy_A1[lastStepIndex], A2[lastStepIndex], ddx_A2[lastStepIndex], ddy_A2[lastStepIndex]};
            // this->currentFields = fields;

            this->gaugeL2 = 0;

            this->rho = rho;
            this->J1 = J1;
            this->J2 = J2;

            this->ddx_J1 = ddx_J1;
            this->ddy_J2 = ddy_J2;

            std::vector<double> kx_deriv_1 = compute_wave_numbers(Nx, x[Nx-1] - x[0], true);
            std::vector<double> ky_deriv_1 = compute_wave_numbers(Ny, y[Ny-1] - y[0], true);

            std::vector<double> kx_deriv_2 = compute_wave_numbers(Nx, x[Nx-1] - x[0], false);
            std::vector<double> ky_deriv_2 = compute_wave_numbers(Ny, y[Ny-1] - y[0], false);

            this->kx_deriv_1 = kx_deriv_1;
            this->ky_deriv_1 = ky_deriv_1;

            this->kx_deriv_2 = kx_deriv_2;
            this->ky_deriv_2 = ky_deriv_2;

            this->forwardIn   = new std::complex<double>[(Nx-1) * (Ny-1)];
            this->forwardOut  = new std::complex<double>[(Nx-1) * (Ny-1)];
            this->backwardIn  = new std::complex<double>[(Nx-1) * (Ny-1)];
            this->backwardOut = new std::complex<double>[(Nx-1) * (Ny-1)];

            this->timeComponent1 = 0;
            this->timeComponent2 = 0;
            this->timeComponent3 = 0;
            this->timeComponent4 = 0;
            this->timeComponent5 = 0;
            this->timeComponent6 = 0;

            // Create FFTW plans for the forward and inverse FFT
            forward_plan = fftw_plan_dft_2d(Nx-1, Ny-1,
                reinterpret_cast<fftw_complex*>(forwardIn), 
                reinterpret_cast<fftw_complex*>(forwardOut), 
                FFTW_FORWARD, FFTW_ESTIMATE);
            inverse_plan = fftw_plan_dft_2d(Nx-1, Ny-1,
                reinterpret_cast<fftw_complex*>(backwardIn), 
                reinterpret_cast<fftw_complex*>(backwardOut), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        }
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
        std::vector<std::vector<double>*> x_elec;
        std::vector<std::vector<double>*> y_elec;
        std::vector<std::vector<double>*> vx_elec;
        std::vector<std::vector<double>*> vy_elec;
        std::vector<std::vector<double>*> Px_elec;
        std::vector<std::vector<double>*> Py_elec;
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
        std::vector<std::vector<std::complex<double>>>** phi;
        std::vector<std::vector<std::complex<double>>>** ddx_phi;
        std::vector<std::vector<std::complex<double>>>** ddy_phi;
        std::vector<std::vector<std::complex<double>>>** A1;
        std::vector<std::vector<std::complex<double>>>** ddx_A1;
        std::vector<std::vector<std::complex<double>>>** ddy_A1;
        std::vector<std::vector<std::complex<double>>>** A2;
        std::vector<std::vector<std::complex<double>>>** ddx_A2;
        std::vector<std::vector<std::complex<double>>>** ddy_A2;
        std::vector<std::vector<std::complex<double>>>** rho;
        std::vector<std::vector<std::complex<double>>>** J1;
        std::vector<std::vector<std::complex<double>>>** J2;
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

        double timeComponent1, timeComponent2, timeComponent3, timeComponent4, timeComponent5, timeComponent6, timeComponent7;

        void computeGaugeL2();
        void updateParticleLocations();
        void updateParticleVelocities();
        void scatterFields();
        void updateWaves();
        double gatherField(double p_x, double p_y, std::vector<std::vector<std::complex<double>>>& field);
        void gatherFields(double p_x, double p_y, std::vector<std::vector<std::vector<std::complex<double>>>>& fields, std::vector<double>& fields_out);
        void scatterField(double p_x, double p_y, double value, std::vector<std::vector<std::complex<double>>>& field);
        void shuffleSteps();
        void updatePhi();
        void updateA1();
        void updateA2();
        std::vector<double> compute_wave_numbers(int N, double L);
        void computeFirstDerivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                        std::vector<std::vector<std::complex<double>>>& derivative_field, bool derivative_in_x);
        void computeSecondDerivative(const std::vector<std::vector<std::complex<double>>>& input_field, 
                std::vector<std::vector<std::complex<double>>>& derivative_field, bool derivative_in_x);
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