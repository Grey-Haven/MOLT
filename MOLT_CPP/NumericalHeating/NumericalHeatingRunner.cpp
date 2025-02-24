#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <filesystem>
#include <cassert>

#include <cmath>
#include <random>
#include "../MOLTEngine.h"
#include "../Derivative.h"

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
    
    const double T_bar = 2.371698e6;                                    // Average temperature [K]
    const double n_bar = 1.1297e14;                                     // Average macroscopic number density [m^-3]
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
    const double L_x = 50;
    const double L_y = 50;

    const double a_x = -L_x/2;
    const double b_x =  L_x/2;

    const double a_y = -L_y/2;
    const double b_y =  L_y/2;
    // End physical grid parameters

    // Set up nondimensional grid
    int N;
    sscanf (argv[1],"%d",&N);
    const int Nx = N;
    const int Ny = N;
    const double dx = (b_x-a_x)/(Nx);
    const double dy = (b_y-a_y)/(Ny);

    double x[Nx]; // [a_x,b_x)
    double y[Ny]; // [a_x,b_x)

    std::string nxn = std::to_string(Nx) + "x" + std::to_string(Ny);

    for (int i = 0; i < Nx; i++) {
        x[i] = a_x + i*dx;
        y[i] = a_y + i*dy;
    }
    // End set up of nondimensional grid

    // Particle Setup
    /* 
     * We double the number of particles and halve their weight.
     * The initial setup will place two electrons on top of each other
     * cancelling out any current.
     */
    const int numHalfParticles = 250632;
    const int numParticles = 2*numHalfParticles;
    
    double macroparticleWeight = (L_x*L_y) / numParticles;
    double macroparticleWeight_elec = macroparticleWeight;
    double macroparticleWeight_ion  = 2*macroparticleWeight;
    // const int numParticles = N*N*100;

    // const double v1_drift = kappa / 100;
    // const double v2_drift = kappa / 100;

    const double sigma_v = 1;

    const double v_0 = 0;

    std::default_random_engine generator;
    // std::normal_distribution<double> location_distribution(x_0, sigma_x);
    std::normal_distribution<double> velocity_distribution(v_0, sigma_v);
    std::uniform_real_distribution<double> x_distribution(a_x, b_x);
    std::uniform_real_distribution<double> y_distribution(a_y, b_y);
    // End Particle Setup

    std::vector<std::vector<double>*> x1_elec_hist(N_h);
    std::vector<std::vector<double>*> x2_elec_hist(N_h);
    
    std::vector<double> x1_ion(numHalfParticles);
    std::vector<double> x2_ion(numHalfParticles);
    std::vector<double> v1_ion(numHalfParticles);
    std::vector<double> v2_ion(numHalfParticles);

    std::vector<std::vector<double>*> v1_elec_hist(N_h);
    std::vector<std::vector<double>*> v2_elec_hist(N_h);

    for (int h = 0; h < N_h; h++) {
        x1_elec_hist[h] = new std::vector<double>(numParticles);
        x2_elec_hist[h] = new std::vector<double>(numParticles);

        v1_elec_hist[h] = new std::vector<double>(numParticles);
        v2_elec_hist[h] = new std::vector<double>(numParticles);
    }
 
    srandom(time(NULL));

    // for (int i = 0; i < numParticles; i++) {

    //     double x_p = x_distribution(generator);
    //     double y_p = y_distribution(generator);

    //     (*x1_elec_hist[N_h - 1])[i] = x_p;
    //     (*x2_elec_hist[N_h - 1])[i] = y_p;
    //     (*x1_elec_hist[N_h - 2])[i] = x_p;
    //     (*x2_elec_hist[N_h - 2])[i] = y_p;
    //     x1_ion[i] = x_p;
    //     x2_ion[i] = y_p;

    //     double vx_p = velocity_distribution(generator);
    //     double vy_p = velocity_distribution(generator);

    //     (*v1_elec_hist[N_h - 1])[i] = vx_p;
    //     (*v2_elec_hist[N_h - 1])[i] = vy_p;
    //     (*v1_elec_hist[N_h - 2])[i] = vx_p;
    //     (*v2_elec_hist[N_h - 2])[i] = vy_p;
    //     (*v1_elec_hist[N_h - 3])[i] = vx_p;
    //     (*v2_elec_hist[N_h - 3])[i] = vy_p;

    //     v1_ion[i] = 0;
    //     v2_ion[i] = 0;
    // }

    std::string elec_file_path = "./initial_conditions/particles_0.csv";

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
    int numCols = data.size();

    std::cout << numCols << " == " << numHalfParticles << std::endl;
    assert(numCols == numHalfParticles && "Loading particles from cold storage gives the expected number of particles.");

    for (int i = 0; i < numCols; i++) {
        double x1  = stod(data[i][0]);
        double x2  = stod(data[i][1]);
        double vx1 = stod(data[i][2]);
        double vx2 = stod(data[i][3]);

        (*x1_elec_hist[N_h - 1])[i] = x1;
        (*x2_elec_hist[N_h - 1])[i] = x2;
        (*x1_elec_hist[N_h - 2])[i] = x1;
        (*x2_elec_hist[N_h - 2])[i] = x2;
        (*x1_elec_hist[N_h - 3])[i] = x1;
        (*x2_elec_hist[N_h - 3])[i] = x2;

        (*v1_elec_hist[N_h - 1])[i] = vx1;
        (*v2_elec_hist[N_h - 1])[i] = vx2;
        (*v1_elec_hist[N_h - 2])[i] = vx1;
        (*v2_elec_hist[N_h - 2])[i] = vx2;
        (*v1_elec_hist[N_h - 3])[i] = vx1;
        (*v2_elec_hist[N_h - 3])[i] = vx2;

        (*x1_elec_hist[N_h - 1])[i+numHalfParticles] = x1;
        (*x2_elec_hist[N_h - 1])[i+numHalfParticles] = x2;
        (*x1_elec_hist[N_h - 2])[i+numHalfParticles] = x1;
        (*x2_elec_hist[N_h - 2])[i+numHalfParticles] = x2;
        (*x1_elec_hist[N_h - 3])[i+numHalfParticles] = x1;
        (*x2_elec_hist[N_h - 3])[i+numHalfParticles] = x2;

        (*v1_elec_hist[N_h - 1])[i+numHalfParticles] = -vx1;
        (*v2_elec_hist[N_h - 1])[i+numHalfParticles] = -vx2;
        (*v1_elec_hist[N_h - 2])[i+numHalfParticles] = -vx1;
        (*v2_elec_hist[N_h - 2])[i+numHalfParticles] = -vx2;
        (*v1_elec_hist[N_h - 3])[i+numHalfParticles] = -vx1;
        (*v2_elec_hist[N_h - 3])[i+numHalfParticles] = -vx2;

        x1_ion[i] = x1;
        x2_ion[i] = x2;
        v1_ion[i] = 0;
        v2_ion[i] = 0;
    }

    // for (int i = 0; i < x1_ion.size(); i++) {
    //     std::cout << x1_ion[i] << ", " << x2_ion[i] << std::endl;
    // }
    // return 0;

    // dxs[g] = dx;
    // double dt = dx / (sqrt(2) * kappa);
    // int N_steps = int(T_final / dt);
    const int N_steps = 1e6;
    const double dt = double(T_final) / double(N_steps);
    const double MAX_DT = (L_x/double(N))/60.0; // for first paper CFL = 1, 2, 4, 8. Fixed mesh of 64x64. Do same with sheath.

    std::cout << dt << " < " << MAX_DT << std::endl;
    if (dt >= MAX_DT) {
        std::cout << dt << "TOO HIGH" << std::endl;
        throw -1;
    }

    std::cout << "Numerical Reference Scalings" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << " L (Debye Length) [m]: " << L << std::endl;
    std::cout << " T (Angular Plasma Period) [s/rad]: " << T << std::endl;
    std::cout << " V (Thermal Velocity) [m/s]: " << V << std::endl;
    std::cout << " n_bar (average number density) [m^{-3}]: " << n_bar << std::endl;
    std::cout << " T_bar (average temperature) [K]: " << T_bar << std::endl;

    std::cout << "============================" << std::endl;
    std::cout << "Timestepping Information" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << " N_steps: " << N_steps << std::endl;
    std::cout << " Approx. CFL (field): " << kappa*dt/dx << std::endl;
    std::cout << " Approx. CFL (particle): " << 10*dt/dx << std::endl;

    std::cout << "============================" << std::endl;
    std::cout << "Dimensional Quantities" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << " Domain length [m]: " << L*L_x << std::endl;
    std::cout << " Plasma period [s]: " << 2*M_PI*T << std::endl;
    std::cout << " Final time [s]: " << 2*M_PI*T_final*T << std::endl;
    std::cout << " dt [s] = " << 2*M_PI*T*dt << std::endl;
    std::cout << " dx [m] = " << L*dx << std::endl;

    std::cout << "============================" << std::endl;
    std::cout << "Non-Dimensional Quantities" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << " Domain length [non-dimensional]: " << L_x << std::endl;
    std::cout << " kappa [non-dimensional] = " << kappa << std::endl;
    std::cout << " Final time [non-dimensional]: " << T_final << std::endl;
    std::cout << " dt [non-dimensional] = " << dt << std::endl;
    std::cout << " dx [non-dimensional] = " << dx << std::endl;
    std::cout << " Grid cells per Debye length [non-dimensional]: " << 1.0/dx << std::endl;
    std::cout << " Timesteps per plasma period [non-dimensional]: " << 1.0/dt << std::endl;


    std::cout << "============================" << std::endl;
    std::cout << "MISC" << std::endl;
    std::cout << "============================" << std::endl;


    std::cout << "dx: " << dx << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "lambda_D: " << lambda_D << std::endl;
    std::cout << "w_p^-1: " << double(1)/w_p << std::endl;
    std::cout << "lambda_D/dx : " << lambda_D / dx << std::endl;
    std::cout << "v_th = lambda_D*w_p : " << lambda_D*w_p << std::endl;
    std::cout << "sigma_1 : " << sig_1 << std::endl;
    std::cout << "sigma_2 : " << sig_2 << std::endl;
    std::cout << "kappa: " << kappa << std::endl;

    MOLTEngine::RhoUpdate rhoUpdate;
    Derivative::DerivativeMethod derivativeMethod;
    Interpolate::InterpolateMethod interpolateMethod;
    MOLTEngine::NumericalMethod numericalMethod;
    MOLTEngine::MOLTMethod moltMethod;

    if (strcmp(argv[2], "INTEGRAL") == 0) {
        moltMethod = MOLTEngine::Integral;
    } else if (strcmp(argv[2], "HELMHOLTZ") == 0) {
        moltMethod = MOLTEngine::Helmholtz;
    } else {
        throw -1;
    }

    if (strcmp(argv[3], "CONSERVING") == 0) {
        rhoUpdate = MOLTEngine::CONSERVING;
    } else if (strcmp(argv[3], "NAIVE") == 0) {
        rhoUpdate = MOLTEngine::NAIVE;
    } else {
        throw -1;
    }

    if (strcmp(argv[4], "BDF1") == 0) {
        numericalMethod = MOLTEngine::BDF1;
    } else if (strcmp(argv[4], "BDF2") == 0) {
        numericalMethod = MOLTEngine::BDF2;
    } else if (strcmp(argv[4], "CDF2") == 0) {
        numericalMethod = MOLTEngine::CDF2;
    } else if (strcmp(argv[4], "DIRK2") == 0) {
        numericalMethod = MOLTEngine::DIRK2;
    } else if (strcmp(argv[4], "DIRK3") == 0) {
        numericalMethod = MOLTEngine::DIRK3;
    } else {
        throw -1;
    }

    if (strcmp(argv[5], "MOLT") == 0) {
        derivativeMethod = Derivative::MOLT;
    } else if (strcmp(argv[5], "FFT") == 0) {
        derivativeMethod = Derivative::FFT;
    } else if (strcmp(argv[5], "FD6") == 0) {
        derivativeMethod = Derivative::FD6;
    } else {
        throw -1;
    }

    bool correctGauge = false;

    if (strcmp(argv[6], "CORRECT_GAUGE") == 0) {
        correctGauge = true;
    } else {
        correctGauge = false;
    }

    // if (strcmp(argv[7], "QUADRATIC") == 0) {
    interpolateMethod = Interpolate::Quadratic;
    // } else {
    //     throw -1;
    // }

    std::string subpath = "";

    if (moltMethod == MOLTEngine::Integral) {
        subpath += "Integral";
    } else if (moltMethod == MOLTEngine::Helmholtz) {
        subpath += "Helmholtz";
    }

    if (rhoUpdate == MOLTEngine::CONSERVING) {
        subpath += "/conserving";
    } else if (rhoUpdate == MOLTEngine::NAIVE) {
        subpath += "/naive";
    }

    if (numericalMethod == MOLTEngine::BDF1) {
        subpath += "/BDF1";
    } else if (numericalMethod == MOLTEngine::BDF2) {
        subpath += "/BDF2";
    } else if (numericalMethod == MOLTEngine::DIRK2) {
        subpath += "/DIRK2";
    } else if (numericalMethod == MOLTEngine::DIRK3) {
        subpath += "/DIRK3";
    } else if (numericalMethod == MOLTEngine::CDF2) {
        subpath += "/CDF2";
    }

    if (derivativeMethod == Derivative::DerivativeMethod::MOLT) {
        subpath += "/MOLT_deriv";
    } else if (derivativeMethod == Derivative::DerivativeMethod::FFT) {
        subpath += "/FFT_deriv";
    } else if (derivativeMethod == Derivative::DerivativeMethod::FD6) {
        subpath += "/FD6_deriv";
    }

    if (interpolateMethod == Interpolate::Linear) {
        subpath += "/linear_interpolate";
    } else if (interpolateMethod == Interpolate::Quadratic) {
        subpath += "/quadratic_interpolate";
    }

    std::cout << "Arguments:" << std::endl;

    std::cout << argv[1] << std::endl;
    std::cout << argv[2] << std::endl;
    std::cout << argv[3] << std::endl;
    std::cout << argv[4] << std::endl;
    std::cout << argv[5] << std::endl;
    std::cout << argv[6] << std::endl;

    // uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::string savePath = "./results/" + subpath + "/" + nxn + "/" + "run" + (correctGauge ? "_correct_gauge" : "");
    std::string debugPath = "./debug/" + subpath + "/" + nxn;
    std::cout << savePath << std::endl;

    // Create results folder
    std::filesystem::create_directories(savePath);

    MOLTEngine molt(Nx, Ny, numParticles, numHalfParticles, N_h, x, y,
                    x1_elec_hist, x2_elec_hist, v1_elec_hist, v2_elec_hist,
                    x1_ion, x2_ion, v1_ion, v2_ion,
                    dx, dy, dt, sig_1, sig_2, kappa, q_elec, m_elec, q_ions, m_ions,
                    macroparticleWeight, macroparticleWeight,
                    numericalMethod, moltMethod, rhoUpdate, derivativeMethod, interpolateMethod, correctGauge, savePath, debugPath, false);

    struct timeval begin, end;
    gettimeofday( &begin, NULL );
    
    // Kokkos::Timer timer;

    std::vector<double> gauge_L2(N_steps+1);
    std::vector<double> gauss_L2_divE(N_steps+1);
    std::vector<double> gauss_L2_divA(N_steps+1);
    std::vector<double> gauss_L2_wave(N_steps+1);
    std::vector<double> rho_total(N_steps+1);
    std::vector<double> rho_elec(N_steps+1);
    std::vector<double> rho_ions(N_steps+1);
    std::vector<double> mass_total(N_steps+1);
    std::vector<double> kinetic_energy(N_steps+1);
    std::vector<double> potential_energy(N_steps+1);
    std::vector<double> energy_total(N_steps+1);
    std::vector<double> temperature(N_steps+1);
    std::vector<double> totalMomentum(N_steps+1);
    
    molt.computePhysicalDiagnostics();

    gauss_L2_divE[0] = molt.getGaussL2_divE();
    gauss_L2_divA[0] = molt.getGaussL2_divA();
    gauss_L2_wave[0] = molt.getGaussL2_wave();
    gauge_L2[0] = molt.getGaugeL2();
    rho_elec[0] = molt.getElecCharge();
    rho_ions[0] = molt.getIonsCharge();
    rho_total[0] = molt.getTotalCharge();
    kinetic_energy[0] = molt.getKineticEnergy();
    potential_energy[0] = molt.getPotentialEnergy();
    energy_total[0] = molt.getTotalEnergy();
    mass_total[0] = molt.getTotalMass();
    temperature[0] = molt.getTemperature();
    totalMomentum[0] = molt.getTotalMomentum();

    std::ofstream gaugeFile;
    std::ofstream gaussFile;
    std::ofstream chargeFile;
    std::ofstream energyFile;
    std::ofstream massFile;
    std::ofstream tempFile;
    std::ofstream momFile;

    for (int n = 0; n < N_steps; n++) {
        // if (n % 500 == 0) {
        //     molt.print();
        // }
        if (n % 1000 == 0) {
            std::cout << "Step: " << n << "/" << N_steps << " = " << 100*(double(n)/double(N_steps)) << "\% complete" << std::endl;
        }
        if (n % (N_steps / 100) == 0) {
            gaugeFile.open(savePath + "/gauge_" + nxn + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                gaugeFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << gauge_L2[n_sub] << std::endl;
            }
            gaugeFile.close();
            gaussFile.open(savePath + "/gauss_" + nxn + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                gaussFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << gauss_L2_divE[n_sub] << "," << gauss_L2_divA[n_sub] << "," << gauss_L2_wave[n_sub] << std::endl;
            }
            gaussFile.close();
            chargeFile.open(savePath + "/charge_" + nxn + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                chargeFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << rho_elec[n_sub] << "," << rho_ions[n_sub] << "," << rho_total[n_sub] << std::endl;
            }
            chargeFile.close();
            energyFile.open(savePath + "/energy_" + nxn + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                energyFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << kinetic_energy[n_sub] << "," << potential_energy[n_sub] << "," << energy_total[n_sub] << std::endl;
            }
            energyFile.close();
            massFile.open(savePath + "/mass_" + nxn + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                massFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << mass_total[n_sub] << std::endl;
            }
            massFile.close();
            tempFile.open(savePath + "/temperature_" + nxn + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                tempFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << temperature[n_sub] << std::endl;
            }
            tempFile.close();
            momFile.open(savePath + "/total_momentum_" + nxn + ".csv");
            for (int n_sub = 0; n_sub < n; n_sub++) {
                momFile << std::setprecision(16) << std::to_string(dt*n_sub) << "," << totalMomentum[n_sub] << std::endl;
            }
            momFile.close();
        }
        molt.step();
        gauge_L2[n+1] = molt.getGaugeL2();
        gauss_L2_divE[n+1] = molt.getGaussL2_divE();
        gauss_L2_divA[n+1] = molt.getGaussL2_divA();
        gauss_L2_wave[n+1] = molt.getGaussL2_wave();
        rho_elec[n+1] = molt.getElecCharge();
        rho_ions[n+1] = molt.getIonsCharge();
        rho_total[n+1] = molt.getTotalCharge();
        mass_total[n+1] = molt.getTotalMass();
        kinetic_energy[n+1] = molt.getKineticEnergy();
        potential_energy[n+1] = molt.getPotentialEnergy();
        energy_total[n+1] = molt.getTotalEnergy();
        temperature[n+1] = molt.getTemperature();
        totalMomentum[n+1] = molt.getTotalMomentum();

    }
    std::cout << "Done running!" << std::endl;

    gettimeofday( &end, NULL );
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    // double time = timer.seconds();
    std::cout << "Grid size: " << Nx << "x" << Ny << " ran for " << time << " seconds." << std::endl;
    // }
    molt.printTimeDiagnostics();

    gaugeFile.open(savePath + "/gauge_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        gaugeFile << std::setprecision(16) << std::to_string(dt*n) << "," << gauge_L2[n] << std::endl;
    }
    gaugeFile.close();
    gaussFile.open(savePath + "/gauss_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        gaussFile << std::setprecision(16) << std::to_string(dt*n) << "," << gauss_L2_divE[n] << "," << gauss_L2_divA[n] << "," << gauss_L2_wave[n] << std::endl;
    }
    gaussFile.close();
    chargeFile.open(savePath + "/charge_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        chargeFile << std::setprecision(16) << std::to_string(dt*n) << ","  << rho_elec[n] << "," << rho_ions[n] << "," << rho_total[n] << std::endl;
    }
    chargeFile.close();
    energyFile.open(savePath + "/energy_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        energyFile << std::setprecision(16) << std::to_string(dt*n) << "," << kinetic_energy[n] << "," << potential_energy[n] << "," << energy_total[n] << std::endl;
    }
    energyFile.close();
    massFile.open(savePath + "/mass_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        massFile << std::setprecision(16) << std::to_string(dt*n) << "," << mass_total[n] << std::endl;
    }
    massFile.close();
    tempFile.open(savePath + "/temperature_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        tempFile << std::setprecision(16) << std::to_string(dt*n) << "," << temperature[n] << std::endl;
    }
    tempFile.close();
    momFile.open(savePath + "/total_momentum_" + nxn + ".csv");
    for (int n = 0; n < N_steps+1; n++) {
        momFile << std::setprecision(16) << std::to_string(dt*n) << "," << totalMomentum[n] << std::endl;
    }
    momFile.close();

    return 0;
}
