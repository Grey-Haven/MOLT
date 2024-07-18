#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <filesystem>

#include <cmath>
#include <random>
#include "MOLTEngine.h"

#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    // Physical constants
    // const double M_electron = 9.109e-31; // [kg]
    // const double Q_electron = 1.602e-19; // [C] (intentionally positive)
    const double c = 299792458;
    const double mu_0 = 1.25663706e-6;
    const double eps_0 = 8.854187817e-12;
    const double k_B = 1.38064852e-23; // Boltzmann constant

    // Particle species mass parameters
    const double ion_electron_mass_ratio = 10000.0;

    const double electron_charge_mass_ratio = -175882008800.0; // Units of C/kg
    const double ion_charge_mass_ratio = -electron_charge_mass_ratio/ion_electron_mass_ratio; // Units of C/kg

    const double M_electron = (-1.602e-19)/electron_charge_mass_ratio;
    const double M_ion = ion_electron_mass_ratio*M_electron;

    const double Q_electron = electron_charge_mass_ratio*M_electron;
    const double Q_ion = ion_charge_mass_ratio*M_ion;

    // Nondimensionalized units
    const double M = M_electron; // [kg]
    const double Q = -Q_electron; // [C]
    
    // Normalized charges
    const double q_ions = Q_ion/Q;
    const double q_elec = Q_electron/Q;

    // Normalized masses
    const double m_ions = M_ion/M;
    const double m_elec = M_electron/M;
    
    const double T_bar = 10000;                                         // Average temperature [K]
    const double n_bar = 2.5e12;                                        // Average macroscopic number density [m^-3]
    const double lambda_D = sqrt((eps_0 * k_B * T_bar)/(n_bar*(Q*Q)));  // Debye length [m]
    const double w_p = sqrt((n_bar * pow(Q, 2)) / (M * eps_0));         // angular frequency

    // More nondimensionalized units
    const double L = lambda_D; // In meters [m]
    const double T = 1 / w_p;  // In seconds/radians [s/r]
    const double V = L / T;    // In [m/s] (thermal velocity lam_D*w_p)

    const double kappa = c/V;  // Nondimensional wave speed [m/s]

    // nondimensionalization parameters for involutions
    const double sig_1 = (M * eps_0) / (pow(Q, 2) * pow(T, 2) * n_bar);
    const double sig_2 = mu_0 * pow(Q, 2) * pow(L, 2) * n_bar / M;

    // const double T_final = argv > 3 ? int[argc[3]] : 80;

    const double T_final = 1000.0; // normalized wrt 1/w_p (plasma period)

    const int N_h = 5;

    // Physical grid parameters
    const double L_x = 16;
    const double L_y = 16;

    const double a_x = -L_x/2;
    const double b_x =  L_x/2;

    const double a_y = -L_y/2;
    const double b_y =  L_y/2;
    // End physical grid parameters

    // Set up nondimensional grid
    int N;
    sscanf (argv[1],"%d",&N);
    // const int N = int(argv[1]);
    const int Nx = N;
    const int Ny = N;
    const double dx = (b_x-a_x)/(Nx);
    const double dy = (b_y-a_y)/(Ny);

    std::cout << "dx: " << double(L_x) / double(N) << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "lambda_D: " << lambda_D << std::endl;
    std::cout << "w_p^-1: " << double(1)/w_p << std::endl;
    std::cout << "lambda_D/dx : " << lambda_D / dx << std::endl;
    std::cout << "lambda_D/N : " << lambda_D / double(N) << std::endl;

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
    std::vector<double> v1_ion(numParticles);
    std::vector<double> v2_ion(numParticles);
    
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

        v1_ion[i] = 0;
        v2_ion[i] = 0;
    }
    // dxs[g] = dx;
    double dt = dx / (sqrt(2) * kappa);

    // std::cout << std::setprecision(15) << "dx = " << dx << " kappa = " << kappa << std::endl;
    // std::cout << std::setprecision(15) << "sqrt(2)*kappa = " << sqrt(2)*kappa << std::endl;
    // std::cout << std::setprecision(15) << "dt = " << dt << std::endl;

    int N_steps = floor(T_final / dt);
   
    std::string nxn = std::to_string(Nx) + "x" + std::to_string(Ny);

    // std::string elec_file_path = "./initial_conditions/" + nxn + "/elec_0.csv";

    // std::cout << elec_file_path << std::endl;
    
    // std::cout << elec_file_path << std::endl;

    // std::string line;
    // std::vector<std::vector<std::string>> data;

    // std::ifstream elec_file(elec_file_path);

    // while (std::getline(elec_file, line)) {
    //     std::vector<std::string> row;
    //     std::stringstream ss(line);
    //     std::string value;

    //     // Parse each line
    //     while (std::getline(ss, value, ',')) {
    //         row.push_back(value);
    //     }
    //     data.push_back(row);
    // }

    // Close the file
    // elec_file.close();
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
    MOLTEngine::RhoUpdate rhoUpdate;
    MOLTEngine::NumericalMethod method;

    if (strcmp(argv[2], "CONSERVING") == 0) {
        rhoUpdate = MOLTEngine::CONSERVING;
    } else if (strcmp(argv[2], "NAIVE") == 0) {
        rhoUpdate = MOLTEngine::NAIVE;
    } else {
        throw -1;
    }

    if (strcmp(argv[3], "BDF1") == 0) {
        method = MOLTEngine::BDF1;
    } else if (strcmp(argv[3], "BDF2") == 0) {
        method = MOLTEngine::BDF2;
    } else if (strcmp(argv[3], "CDF1") == 0) {
        method = MOLTEngine::CDF1;
    } else if (strcmp(argv[3], "DIRK2") == 0) {
        method = MOLTEngine::DIRK2;
    } else if (strcmp(argv[3], "DIRK3") == 0) {
        method = MOLTEngine::DIRK3;
    } else if (strcmp(argv[3], "MOLT_BDF1") == 0) {
        method = MOLTEngine::MOLT_BDF1;
    } else {
        throw -1;
    }
    std::string subpath = "";

    if (rhoUpdate == MOLTEngine::CONSERVING) {
        subpath += "conserving";
    } else if (rhoUpdate == MOLTEngine::NAIVE) {
        subpath += "naive";
    }

    if (method == MOLTEngine::BDF1) {
        subpath += "/BDF1";
    } else if (method == MOLTEngine::BDF2) {
        subpath += "/BDF2";
    } else if (method == MOLTEngine::DIRK2) {
        subpath += "/DIRK2";
    } else if (method == MOLTEngine::DIRK3) {
        subpath += "/DIRK3";
    } else if (method == MOLTEngine::CDF1) {
        subpath += "/CDF1";
    } else if (method == MOLTEngine::MOLT_BDF1) {
        subpath += "/MOLT_BDF1";
    }

    std::string path = "./results/moving_blob/" + subpath + "/" + nxn;
    std::cout << path << std::endl;

    // Create results folder
    std::filesystem::create_directories(path);
    std::filesystem::create_directories(path + "/particles");
    std::filesystem::create_directories(path + "/phi");
    std::filesystem::create_directories(path + "/A1");
    std::filesystem::create_directories(path + "/A2");
    std::filesystem::create_directories(path + "/ddx_phi");
    std::filesystem::create_directories(path + "/ddx_A1");
    std::filesystem::create_directories(path + "/ddx_A2");
    std::filesystem::create_directories(path + "/ddy_phi");
    std::filesystem::create_directories(path + "/ddy_A1");
    std::filesystem::create_directories(path + "/ddy_A2");
    std::filesystem::create_directories(path + "/rho");
    std::filesystem::create_directories(path + "/J1");
    std::filesystem::create_directories(path + "/J2");

    MOLTEngine molt(Nx, Ny, numParticles, N_h, x, y, x1_elec_hist,
                    x2_elec_hist, v1_elec_hist, v2_elec_hist,
                    x1_ion, x2_ion, v1_ion, v2_ion,
                    dx, dy, dt, sig_1, sig_2, kappa, q_elec, m_elec, q_ions, m_ions, method, rhoUpdate, path);

    struct timeval begin, end;
    gettimeofday( &begin, NULL );
    
    // Kokkos::Timer timer;

    // N_steps = 1;

    std::vector<double> gauge_L2(N_steps+1);
    std::vector<double> rho_total(N_steps+1);
    gauge_L2[0] = molt.getGaugeL2();
    rho_total[0] = molt.getTotalCharge();

    std::ofstream gaugeFile;
    std::ofstream rhoFile;

    // N_steps = 5;

    for (int n = 0; n < N_steps; n++) {
        if (n % 1000 == 0) {
            std::cout << "Step: " << n << "/" << N_steps << " = " << 100*(double(n)/double(N_steps)) << "\% complete" << std::endl;
        }
        if (n % (N_steps / 100) == 0) {
            gaugeFile.open(path + "/gauge_" + nxn + "_unfinished_recent" + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                gaugeFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << gauge_L2[n_sub] << std::endl;
            }
            gaugeFile.close();
        }
        if (n % (N_steps / 100) == 0) {
            rhoFile.open(path + "/rho_" + nxn + "_unfinished_recent" + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                rhoFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << rho_total[n_sub] << std::endl;
            }
            rhoFile.close();
        }
        molt.step();
        gauge_L2[n+1] = molt.getGaugeL2();
        rho_total[n+1] = molt.getTotalCharge();
    }
    std::cout << "Done running!" << std::endl;
    gettimeofday( &end, NULL );
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    // double time = timer.seconds();
    std::cout << "Grid size: " << Nx << "x" << Ny << " ran for " << time << " seconds." << std::endl;
    // }
    molt.printTimeDiagnostics();

    gaugeFile.open(path + "/gauge_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        gaugeFile << std::setprecision(16) << std::to_string(dt*n) << "," << gauge_L2[n] << std::endl;
    }
    gaugeFile.close();
    rhoFile.open(path + "/rho_" + nxn + ".csv");
    for (int n_sub = 0; n_sub < N_steps+1; n_sub++) {
        rhoFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << rho_total[n_sub] << std::endl;
    }
    rhoFile.close();

    return 0;
}
