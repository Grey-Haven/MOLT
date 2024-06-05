#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <cmath>
#include <random>
#include "MOLTEngine.h"

#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    // Physical constants
    const double M_electron = 9.109e-31; // [kg]
    const double Q_electron = 1.602e-19; // [C] (intentionally positive)
    const double c = 299792458;
    const double mu_0 = 1.25663706e-6;
    const double eps_0 = 8.854187817e-12;
    const double k_B = 1.38064852e-23; // Boltzmann constant

    // Nondimensionalized units
    const double M = M_electron; // [kg]
    const double Q = Q_electron; // [C]

    const double q_elec = -Q_electron / Q;
    const double m_elec = M_electron / M;
    
    const double T_bar = 10000;                                       // Average temperature [K]
    const double n_bar = 1e13;                                        // Average macroscopic number density [m^-3]
    const double lambda_D = sqrt((eps_0 * k_B * T_bar)/(n_bar*(Q*Q)));  // Debye length [m]
    const double w_p = sqrt((n_bar * pow(Q, 2)) / (M * eps_0));       // angular frequency

    // More nondimensionalized units
    const double L = lambda_D; // In meters [m]
    const double T = 1 / w_p;  // In seconds/radians [s/r]
    const double V = L / T;    // In [m/s] (thermal velocity lam_D*w_p)

    const double kappa = c/V;  // Nondimensional wave speed [m/s]

    // nondimensionalization parameters for involutions
    const double sig_1 = (M * eps_0) / (pow(Q, 2) * pow(T, 2) * n_bar);
    const double sig_2 = mu_0 * pow(Q, 2) * pow(L, 2) * n_bar / M;

    // const double beta_BDF1 = 1;
    // const double beta_BDF2 = 1 / (2.0/3.0);
    // const double beta_BDF3 = 1 / (6.0/11.0);
    const double beta_CDF1 = sqrt(2);

    const double T_final = 1; // normalized wrt 1/w_p (plasma period)

    const int N_h = 6;

    // Physical grid parameters
    const double L_x = 1;
    const double L_y = 1;

    const double a_x = -L_x/2;
    const double b_x =  L_x/2;

    const double a_y = -L_y/2;
    const double b_y =  L_y/2;
    // End physical grid parameters

    // Set up nondimensional grid
    const int N = 64;
    const int Nx = N;
    const int Ny = N;
    const double dx = (b_x-a_x)/(Nx);
    const double dy = (b_y-a_y)/(Ny);

    double x[Nx]; // [a_x,b_x)
    double y[Ny]; // [a_x,b_x)

    for (int i = 0; i < Nx; i++) {
        x[i] = a_x + i*dx;
        y[i] = a_y + i*dy;
    }
    // End set up of nondimensional grid

    // Particle Setup
    // const int numParticles = 2.5e4;
    const int numParticles = N*N*100;
    // const int numParticles = 5;

    // const double v1_drift = kappa / 100;
    // const double v2_drift = kappa / 100;

    const double sigma_x = .05*(b_x - a_x);
    const double sigma_v = 1;

    const double x_0 = (a_x + b_x) / 2;
    const double v_0 = 0;

    std::default_random_engine generator;
    std::normal_distribution<double> location_distribution(x_0, sigma_x);
    std::normal_distribution<double> velocity_distribution(v_0, sigma_v);
    // End Particle Setup

    std::vector<std::vector<double>*> x1_elec_hist(N_h);
    std::vector<std::vector<double>*> x2_elec_hist(N_h);
    
    std::vector<double> x1_ion(numParticles);
    std::vector<double> x2_ion(numParticles);
    
    std::vector<std::vector<double>*> v1_elec_hist(N_h);
    std::vector<std::vector<double>*> v2_elec_hist(N_h);

    for (int h = 0; h < N_h; h++) {
        x1_elec_hist[h] = new std::vector<double>(numParticles);
        x2_elec_hist[h] = new std::vector<double>(numParticles);

        v1_elec_hist[h] = new std::vector<double>(numParticles);
        v2_elec_hist[h] = new std::vector<double>(numParticles);
    }

    // Distribute particles across phase space
    double v1_drift = kappa/100;
    double v2_drift = kappa/100;

    for (int i = 0; i < numParticles; i++) {

        double x_p = location_distribution(generator);
        double y_p = location_distribution(generator);

        if (x_p < a_x) {
            x_p += L_x;
        }
        if (x_p > b_x) {
            x_p -= L_x;
        }
        if (y_p < a_y) {
            y_p += L_y;
        }
        if (y_p > b_y) {
            y_p -= L_y;
        }

        (*x1_elec_hist[N_h - 1])[i] = x_p;
        (*x2_elec_hist[N_h - 1])[i] = y_p;
        (*x1_elec_hist[N_h - 2])[i] = x_p;
        (*x2_elec_hist[N_h - 2])[i] = y_p;
        x1_ion[i] = x_p;
        x2_ion[i] = y_p;

        double vx_p = velocity_distribution(generator) + v1_drift;
        double vy_p = velocity_distribution(generator) + v2_drift;

        (*v1_elec_hist[N_h - 1])[i] = vx_p;
        (*v2_elec_hist[N_h - 1])[i] = vy_p;
        (*v1_elec_hist[N_h - 2])[i] = vx_p;
        (*v2_elec_hist[N_h - 2])[i] = vy_p;
        (*v1_elec_hist[N_h - 3])[i] = vx_p;
        (*v2_elec_hist[N_h - 3])[i] = vy_p;
    }
    // dxs[g] = dx;
    double dt = dx / (sqrt(2) * kappa);

    // std::cout << std::setprecision(15) << "dx = " << dx << " kappa = " << kappa << std::endl;
    // std::cout << std::setprecision(15) << "sqrt(2)*kappa = " << sqrt(2)*kappa << std::endl;
    // std::cout << std::setprecision(15) << "dt = " << dt << std::endl;

    int N_steps = floor(T_final / dt);
   
    std::string nxn = std::to_string(Nx) + "x" + std::to_string(Ny);

    std::string elec_file_path = "./initial_conditions/" + nxn + "/elec_0.csv";

    std::cout << elec_file_path << std::endl;
    
    // std::cout << elec_file_path << std::endl;

    std::string line;
    std::vector<std::vector<std::string>> data;

    std::ifstream elec_file(elec_file_path);

    while (std::getline(elec_file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string value;

        // Parse each line
        while (std::getline(ss, value, ',')) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    // Close the file
    elec_file.close();
    // int numCols = data.size();

    // for (int i = 0; i < numCols; i++) {
    //     double x1  = stod(data[i][0]);
    //     double x2  = stod(data[i][1]);
    //     double vx1 = stod(data[i][2]);
    //     double vx2 = stod(data[i][3]);

    //     (*x1_elec_hist[N_h - 1])[i] = x1;
    //     (*x2_elec_hist[N_h - 1])[i] = x2;
    //     (*x1_elec_hist[N_h - 2])[i] = x1;
    //     (*x2_elec_hist[N_h - 2])[i] = x2;

    //     (*v1_elec_hist[N_h - 1])[i] = vx1;
    //     (*v2_elec_hist[N_h - 1])[i] = vx2;
    //     (*v1_elec_hist[N_h - 2])[i] = vx1;
    //     (*v2_elec_hist[N_h - 2])[i] = vx2;
    //     (*v1_elec_hist[N_h - 3])[i] = vx1;
    //     (*v2_elec_hist[N_h - 3])[i] = vx2;
    // }

    // std::cout << "MOLTing" << std::endl;
    MOLTEngine molt(Nx, Ny, numParticles, N_h, x, y, x1_elec_hist, x2_elec_hist, v1_elec_hist, v2_elec_hist, dx, dy, dt, sig_1, sig_2, kappa, q_elec, m_elec, beta_CDF1);

    struct timeval begin, end;
    gettimeofday( &begin, NULL );
    
    // Kokkos::Timer timer;

    // N_steps = 1;

    std::vector<double> gauge_L2(N_steps+1);
    gauge_L2[0] = molt.getGaugeL2();

    for (int n = 0; n < N_steps; n++) {
        if (n % 100 == 0) {
            std::cout << "Step: " << n << "/" << N_steps << " = " << 100*(double(n)/double(N_steps)) << "\% complete" << std::endl;
        }
        molt.step();
        gauge_L2[n+1] = molt.getGaugeL2();
    }
    std::cout << "Done running!" << std::endl;
    gettimeofday( &end, NULL );
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    // double time = timer.seconds();
    std::cout << "Grid size: " << Nx << "x" << Ny << " ran for " << time << " seconds." << std::endl;
    // }
    molt.printTimeDiagnostics();

    std::ofstream gaugeFile;
    gaugeFile.open("./results/CDF1/gauge_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        gaugeFile << std::setprecision(16) << std::to_string(dt*n) << "," << gauge_L2[n] << std::endl;
    }
    gaugeFile.close();

    return 0;
}
