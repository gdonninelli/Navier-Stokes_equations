#include "NavierStokes.hpp"

// Punto di ingresso del programma
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      
      // 1. Inizializzazione MPI
      // Il terzo argomento '1' indica di usare un solo thread per processo MPI (consigliato per deal.II)
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // --- CONFIGURAZIONE TEST RAPIDO ---
      const unsigned int dim = 2; // Testiamo in 2D

      // Parametri numerici (Taylor-Hood: Velocità grado 2, Pressione grado 1)
      const unsigned int degree_velocity = 2;
      const unsigned int degree_pressure = 1;

      // Parametri fisici e temporali per il test
      const double T_final = 0.1;   // Tempo finale BREVE solo per testare (metti 8.0 per la run vera)
      const double deltat  = 0.005; // Passo temporale (dt)
      const double Re      = 100.0; // Reynolds instazionario (Vedi Project PDF [cite: 111])

      // Nome del file mesh generato da Gmsh
      const std::string mesh_file_name = "../meshes/mesh-2D";

      // 2. Istanziazione del Solver
      // Nota: U_m (velocità max ingresso) è 1.5 come da benchmark 2D-2/3 [cite: 111]
      NavierStokes<dim> flow_solver(mesh_file_name,
                                    degree_velocity,
                                    degree_pressure,
                                    deltat,
                                    T_final,
                                    Re,
                                    1.5); // U_m = 1.5

      // 3. Avvio Simulazione
      flow_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}