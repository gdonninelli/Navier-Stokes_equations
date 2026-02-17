#include "classes/TestCases.hpp"

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Schäfer-Turek 1996 benchmark test cases.
    //
    // Available 2D cases (NavierStokes<2>):
    //   TestCases::make_2D_1  — Steady,   Re=20,        U_m=0.3
    //   TestCases::make_2D_2  — Unsteady, Re=100,       U_m=1.5  (constant inlet)
    //   TestCases::make_2D_3  — Unsteady, Re(t)∈[0,100], U_m=1.5  (sin(πt/8))
    //
    // Available 3D cases (NavierStokes<3>):
    //   TestCases::make_3D_1Z — Steady,   Re=20,        U_m=0.45
    //   TestCases::make_3D_2Z — Unsteady, Re=100,       U_m=2.25 (constant inlet)
    //   TestCases::make_3D_3Z — Unsteady, Re(t)∈[0,100], U_m=2.25 (sin(πt/8))
    //
    // Optional arguments (after mesh_file):
    //   TimeScheme:      BackwardEuler / CrankNicolson
    //   NonlinearMethod: Newton / Linearized
    //   deltat:          positive value or <= 0 for automatic

    // const std::string mesh_2d = "../meshes/mesh-2D-200.msh";
    const std::string mesh_3d = "../meshes/mesh-3D-5.msh";

    // auto tc = TestCases::make_2D_1(mesh_2d);
    // auto tc = TestCases::make_2D_2(mesh_2d);
    // auto tc = TestCases::make_2D_3(mesh_2d);

    auto tc = TestCases::make_3D_1Z(mesh_3d);
    // auto tc = TestCases::make_3D_2Z(mesh_3d);
    // auto tc = TestCases::make_3D_3Z(mesh_3d);

    NavierStokes<3> solver(tc);
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