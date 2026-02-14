#include "classes/NavierStokesCN.hpp"

// ============================================================================
// main_cn.cc — Navier-Stokes with Crank-Nicolson semi-implicit scheme
// ============================================================================
//
// Uses NavierStokesCN<dim> instead of NavierStokes<dim>.
// Key differences from Newton-based main.cc:
//   - No Newton iteration per time step — one linear solve only
//   - Crank-Nicolson O(Δt²) time discretization
//   - 2nd-order extrapolation of transport velocity
//
// Output: forces_cn.txt, solution_*.vtu
// ============================================================================

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // ------- Configuration -------
      const unsigned int dim = 2;

      // Taylor-Hood P2/P1
      const unsigned int degree_velocity = 2;
      const unsigned int degree_pressure = 1;

      // Time parameters
      const double T_final = 8.0;
      const double deltat  = 0.05; // Slightly smaller for CN stability
      const double Re      = 100.0;

      // Mesh
      const std::string mesh_file_name = "../meshes/mesh-2D";

      // ------- Solver -------
      NavierStokesCN<dim> solver(mesh_file_name,
                                 degree_velocity,
                                 degree_pressure,
                                 deltat,
                                 T_final,
                                 Re,
                                 1.5); // U_m = 1.5

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
