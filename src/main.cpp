#include "classes/NavierStokes.hpp"

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Spatial dimension (2 or 3)
    const unsigned int dim = 2;

    // Taylor-Hood elements: P2/P1
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;

    // Physical parameters
    const double Re      = 40.0;
    const double T_final = 8.0;
    const double U_m     = 1.5; // Max inlet velocity

    // Time discretization:  TimeScheme::BackwardEuler  oppure  TimeScheme::CrankNicolson
    // Nonlinear solver:     NonlinearMethod::Newton    oppure  NonlinearMethod::Linearized
    const TimeScheme      time_scheme      = TimeScheme::CrankNicolson;
    const NonlinearMethod nonlinear_method = NonlinearMethod::Newton;

    // Time step: set a positive value to use manually, or <= 0 for automatic
    // selection based on Re (Re<=20 -> 0.1, Re<=50 -> 0.05, Re<=100 -> 0.02, Re<=150 -> 0.01, Re>150 -> 0.005)
    const double deltat = 1.0;

    // Mesh file
    const std::string mesh_file_name = "../meshes/mesh-2D-100.msh";

    NavierStokes<dim> solver(mesh_file_name,
                              degree_velocity,
                              degree_pressure,
                              deltat,
                              T_final,
                              Re,
                              U_m,
                              time_scheme,
                              nonlinear_method);

    solver.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception: " << exc.what() << std::endl
              << "Aborting!" << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception! Aborting!" << std::endl;
    return 1;
  }

  return 0;
}