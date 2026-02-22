#include "classes/TestCases.hpp"

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    //! Select the mesh and test case to run by uncommenting the desired line below.
    
    // const std::string mesh_2d = "../meshes/mesh-2D-200.msh";
    const std::string mesh_3d = "../meshes/mesh-3D-5.msh";

    // auto tc = TestCases::make_2D_1(mesh_2d);
    // auto tc = TestCases::make_2D_2(mesh_2d);
    // auto tc = TestCases::make_2D_3(mesh_2d);

    // auto tc = TestCases::make_3D_1Z(mesh_3d);
    auto tc = TestCases::make_3D_2Z(mesh_3d);
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
