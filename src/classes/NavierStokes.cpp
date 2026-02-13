#include "NavierStokes.hpp"

namespace
{
  bool is_in(const unsigned int val, const std::vector<unsigned int> &vec)
  {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
  }
}

template <unsigned int dim>
void NavierStokes<dim>::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Setup..." << std::endl;

  nu = U_m * D / Re;
  pcout << "  Reynolds number: " << Re << std::endl;
  pcout << "  Kinematic viscosity: " << nu << std::endl;
  pcout << "  Inlet Velocity (Um): " << U_m << std::endl;
  pcout << "  Time step: " << deltat << std::endl;

  // 1. DoFHandler 
  dof_handler.distribute_dofs(*fe);

  // 2. Blocks enumeration (velocity and pressure)
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1; // Pressure is the last block
  DoFRenumbering::component_wise(dof_handler, block_component);

  // 3. DoFs counting
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  pcout << "  Number of active cells: " << mesh.n_active_cells() << std::endl;
  pcout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
        << " (" << n_u << " + " << n_p << ")" << std::endl;

  // 4. IndexSets initialization (parallel management)
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  block_owned_dofs.resize(2);
  block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
  block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);

  block_relevant_dofs.resize(2);
  block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
  block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

  // 5. Vectors initialization
  // solution_owned: contains only local DoFs (for the solver)
  solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
  system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
  newton_update.reinit(block_owned_dofs, MPI_COMM_WORLD);

  // solution, solution_old, current_solution: contain also ghost DoFs (for assembly)
  solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  solution_old.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  current_solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);

  // Set initial conditions
  solution = 0.0;
  solution_old = 0.0;
  current_solution = 0.0;
  
  // TODO: Maybe set initial condition here 

  // 6. Matrices initialization
  // Dynamic sparsity pattern
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, locally_relevant_dofs, false);
  
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.n_locally_owned_dofs_per_processor(),
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);

  system_matrix.reinit(block_owned_dofs, dsp, MPI_COMM_WORLD);
  pressure_mass.reinit(block_owned_dofs, dsp, MPI_COMM_WORLD);

  pcout << "Setup complete." << std::endl;
}


template <unsigned int dim>
void NavierStokes<dim>::assemble_newton_system()
{
  system_matrix = 0;
  system_rhs    = 0;
  pressure_mass = 0; // TODO: Recompute if necessary

  const QGauss<dim> quadrature_formula(degree_velocity + 1);

  FEValues<dim> fe_values(*fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass(dofs_per_cell, dofs_per_cell); // Solo blocco p-p
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Estrattori per FEValues (gestiscono vettori e scalari)
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  // Valori della soluzione all'iterazione k (current) e al tempo n-1 (old)
  std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
  std::vector<double>         current_pressure_values(n_q_points);
  
  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);

  // Vettori di supporto per phi_u, grad_phi_u, phi_p
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs    = 0;
          cell_pressure_mass = 0;

          fe_values.reinit(cell);

          // Estraggo i valori dai vettori globali nei punti di quadratura
          fe_values[velocities].get_function_values(current_solution, current_velocity_values);
          fe_values[velocities].get_function_gradients(current_solution, current_velocity_gradients);
          fe_values[pressure].get_function_values(current_solution, current_pressure_values);
          
          fe_values[velocities].get_function_values(solution_old, old_velocity_values);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const double JxW = fe_values.JxW(q);

              // Precalcolo funzioni di base per efficienza
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_u[k]      = fe_values[velocities].value(k, q);
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                  phi_p[k]      = fe_values[pressure].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // --- Assemblaggio Residuo (RHS) ---
                  // R = F - (Termini inerziali + Viscosi + Convettivi + Pressione)
                  // Nota: portiamo tutto a destra, quindi i segni sono opposti al LHS standard

                  // Termine temporale: (u_k - u_old) / dt * v
                  const double time_term = (current_velocity_values[q] - old_velocity_values[q]) * phi_u[i] / deltat;
                  
                  // Termine convettivo: (u_k . grad u_k) * v
                  const double conv_term = (current_velocity_values[q] * current_velocity_gradients[q]) * phi_u[i];

                  // Termine viscoso: nu * (grad u_k : grad v)
                  const double visc_term = nu * scalar_product(current_velocity_gradients[q], grad_phi_u[i]);

                  // Termine pressione: - p_k * div v
                  const double pres_term = -current_pressure_values[q] * fe_values[velocities].divergence(i, q);
                  
                  // Termine incompressibilità: - q * div u_k
                  const double div_term = -phi_p[i] * trace(current_velocity_gradients[q]);

                  cell_rhs(i) += (-time_term - conv_term - visc_term - pres_term - div_term) * JxW;


                  // --- Assemblaggio Jacobiano (Matrice di Sistema) ---
                  // Linearizzazione di Newton
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // 1. Massa (da derivata temporale): 1/dt * (u, v)
                      double val = (phi_u[i] * phi_u[j]) / deltat;

                      // 2. Viscosità: nu * (grad u, grad v)
                      val += nu * scalar_product(grad_phi_u[i], grad_phi_u[j]);

                      // 3. Convezione Linearizzata:
                      // (u_k . grad du) * v + (du . grad u_k) * v
                      val += (current_velocity_values[q] * grad_phi_u[j]) * phi_u[i] +
                             (phi_u[j] * current_velocity_gradients[q]) * phi_u[i];

                      // 4. Pressione: -(p, div v)
                      val -= phi_p[j] * fe_values[velocities].divergence(i, q);

                      // 5. Divergenza: -(q, div u)
                      val -= phi_p[i] * fe_values[velocities].divergence(j, q);

                      cell_matrix(i, j) += val * JxW;

                      // Matrice massa pressione per precondizionatore
                      cell_pressure_mass(i, j) += phi_p[i] * phi_p[j] * JxW;
                    }
                }
            }

          cell->get_dof_indices(local_dof_indices);

          // Gestione Constraints (Dirichlet)
          // Attenzione: In Newton, se un DoF è vincolato (Dirichlet), 
          // la matrice deve avere 1 sulla diagonale e il RHS deve essere 0 
          // (perché l'aggiornamento delta_u deve essere nullo sul bordo Dirichlet).
          
          AffineConstraints<double> constraints;
          constraints.clear();
          
          // Inlet
          inlet_velocity->set_time(time); // Aggiorno tempo
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   inlet_boundary_id,
                                                   *inlet_velocity,
                                                   constraints);
          // Walls & Cylinder (No-slip)
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   wall_boundary_id,
                                                   Functions::ZeroFunction<dim>(dim + 1),
                                                   constraints);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   cylinder_boundary_id,
                                                   Functions::ZeroFunction<dim>(dim + 1),
                                                   constraints);
          
          constraints.close();
          constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
          
          // Per la pressure mass non applichiamo constraints di velocità
          constraints.distribute_local_to_global(cell_pressure_mass, local_dof_indices, pressure_mass);
        }
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);
}


template <unsigned int dim>
void NavierStokes<dim>::solve_newton_system()
{
  SolverControl solver_control(10000, 1e-6 * system_rhs.l2_norm());

  // Scegli il precondizionatore (BlockDiagonal o BlockTriangular o Identity)
  // Qui uso BlockInverse come esempio generico basato sull'HPP
  
  PreconditionBlockDiagonal preconditioner;
  preconditioner.initialize(system_matrix, pressure_mass);

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
  
  // Risolviamo per newton_update (delta u)
  newton_update = 0.0;
  solver.solve(system_matrix, newton_update, system_rhs, preconditioner);

  // Distribuisco i ghost
  AffineConstraints<double> constraints;
  // ... (re-interpolare constraints per coerenza se necessario, ma qui update è 0 su Dirichlet)
  constraints.close();
  constraints.distribute(newton_update);
}


template <unsigned int dim>
void NavierStokes<dim>::compute_lift_drag(double &drag_coeff, double &lift_coeff) const
{
  // Calcolo delle forze sul cilindro
  // F = Integral_S (sigma * n) dS
  // sigma = -pI + nu * (grad u + grad u^T)

  double force_x = 0.0;
  double force_y = 0.0;

  const QGauss<dim - 1> face_quadrature_formula(degree_velocity + 1);

  FEFaceValues<dim> fe_face_values(*fe,
                                   face_quadrature_formula,
                                   update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values |
                                   update_normal_vectors);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> velocity_values(face_quadrature_formula.size());
  std::vector<Tensor<2, dim>> velocity_gradients(face_quadrature_formula.size());
  std::vector<double>         pressure_values(face_quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == cylinder_boundary_id)
                {
                  fe_face_values.reinit(cell, f);

                  fe_face_values[velocities].get_function_values(current_solution, velocity_values);
                  fe_face_values[velocities].get_function_gradients(current_solution, velocity_gradients);
                  fe_face_values[pressure].get_function_values(current_solution, pressure_values);

                  for (unsigned int q = 0; q < face_quadrature_formula.size(); ++q)
                    {
                      const double         p      = pressure_values[q];
                      const Tensor<2, dim> grad_u = velocity_gradients[q];
                      const Tensor<1, dim> normal = fe_face_values.normal_vector(q);
                      const double         JxW    = fe_face_values.JxW(q);

                      // Tensore degli sforzi: sigma = -pI + rho*nu*(grad_u + grad_u^T)
                      // Nota: rho=1 nel codice, ma usiamo la variabile membro rho per correttezza
                      Tensor<2, dim> stress;
                      for (unsigned int i = 0; i < dim; ++i)
                        stress[i][i] = -p;
                      
                      // Parte viscosa (Tensor simmetrico)
                      // stress += rho * nu * (grad_u + transpose(grad_u));
                      // In deal.II:
                      stress += rho * nu * (grad_u + transpose(grad_u));

                      // Forza locale = stress * normal
                      Tensor<1, dim> force_loc = stress * normal;

                      force_x += force_loc[0] * JxW;
                      force_y += force_loc[1] * JxW; // Lift direction
                    }
                }
            }
        }
    }

  // Somma MPI
  force_x = Utilities::MPI::sum(force_x, MPI_COMM_WORLD);
  force_y = Utilities::MPI::sum(force_y, MPI_COMM_WORLD);

  // Coefficienti adimensionali
  // Cd = 2 * Fx / (rho * U_mean^2 * D * H) -- Formula variabile a seconda della definizione del benchmark
  // La formula classica Schaefer-Turek è:
  // Cd = 2 * Fd / (rho * U_mean^2 * D)  (in 2D)
  // Qui implemento quella generica basata su HPP variables
  const double den = 0.5 * rho * U_m * U_m * D; // * H se necessario in 3D per area riferimento
  
  drag_coeff = force_x / den;
  lift_coeff = force_y / den;
}

template <unsigned int dim>
void NavierStokes<dim>::output(const unsigned int time_step)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  
  // Partizionamento vettori per output
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.push_back("pressure");
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(current_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  
  data_out.build_patches();

  std::string filename = "solution-" + Utilities::int_to_string(time_step, 4) + ".vtu";
  data_out.write_vtu_with_pvtu_record("./", filename, time_step, MPI_COMM_WORLD);
}


template <unsigned int dim>
void NavierStokes<dim>::run()
{
  setup();

  // Inizializzo solution_old con la condizione iniziale
  // Qui assume 0, se non diverso
  current_solution = solution_old;

  double time = 0.0;
  unsigned int time_step = 0;

  std::ofstream forces_file;
  if (mpi_rank == 0)
    {
      forces_file.open("forces.txt");
      forces_file << "Time Drag Lift" << std::endl;
    }

  // Loop temporale
  while (time < T)
    {
      time += deltat;
      time_step++;
      
      pcout << "Time step " << time_step << " at t=" << time << std::endl;

      // Aggiorno BC temporali
      inlet_velocity->set_time(time);

      // Loop di Newton
      double residual_norm = 1e10;
      unsigned int newton_iter = 0;

      while (residual_norm > newton_tolerance && newton_iter < newton_max_iterations)
        {
          assemble_newton_system();
          
          residual_norm = system_rhs.l2_norm();
          pcout << "  Newton iter " << newton_iter << ": residual = " << residual_norm << std::endl;

          if (residual_norm < newton_tolerance)
            {
              pcout << "  Converged!" << std::endl;
              break;
            }

          solve_newton_system();

          // Update soluzione: u_{k+1} = u_k + delta_u
          current_solution.add(1.0, newton_update);
          
          // Applico constraints non-omogenei (Inlet variabile)
          // Nota: nel sistema di Newton risolviamo per l'update con BC omogenee,
          // ma current_solution deve rispettare le BC reali.
          // Spesso in Newton si fa: u = u + du. 
          // E le BC sono imposte imponendo u_k=g sul bordo e du=0.
          // La mia assemble gestisce questo tramite AffineConstraints sul residuo e matrice.
          
          newton_iter++;
        }

      // Fine time step: aggiorno la storia
      solution_old = current_solution;

      // Calcolo Lift e Drag
      double drag, lift;
      compute_lift_drag(drag, lift);
      if (mpi_rank == 0)
        {
           pcout << "  Cd: " << drag << ", Cl: " << lift << std::endl;
           forces_file << time << " " << drag << " " << lift << std::endl;
        }

      if (time_step % 10 == 0) // Output ogni 10 step
        output(time_step);
    }
}

// Explicit instantiation
template class NavierStokes<2>;
template class NavierStokes<3>;