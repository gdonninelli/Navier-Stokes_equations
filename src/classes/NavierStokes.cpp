#include "NavierStokes.hpp"

template <unsigned int dim>
void NavierStokes<dim>::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Setup..." << std::endl;

  // TODO: Controllare se va bene, proabilmente mesh non nativamente supprotata da deal.II, quindi dobbiamo leggere in un triangulation seriale e poi distribuirla.
  // 0. Mesh loading
  // For parallel::fullydistributed::Triangulation we must:
  //   a) Read the mesh into a serial Triangulation
  //   b) Partition it
  //   c) Create the distributed triangulation from the description
  {
    Triangulation<dim> serial_mesh;
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_mesh);

    std::ifstream input_file(mesh_file_name);
    AssertThrow(input_file.is_open(),
                ExcMessage("Could not open mesh file: " + mesh_file_name));

    // Read the file, replacing $ParametricNodes with $Nodes
    // because deal.II does not support $ParametricNodes
    std::stringstream ss;
    std::string line;
    bool in_parametric_nodes = false;
    bool first_line_after_header = false; // the count line right after $ParametricNodes
    while (std::getline(input_file, line))
      {
        // Strip trailing carriage return (Windows line endings)
        if (!line.empty() && line.back() == '\r')
          line.pop_back();

        if (line == "$ParametricNodes")
          {
            ss << "$Nodes\n";
            in_parametric_nodes = true;
            first_line_after_header = true;
          }
        else if (line == "$EndParametricNodes")
          {
            ss << "$EndNodes\n";
            in_parametric_nodes = false;
          }
        else if (in_parametric_nodes)
          {
            if (first_line_after_header)
              {
                // This is the node count line, pass through
                ss << line << "\n";
                first_line_after_header = false;
              }
            else
              {
                // Node line: "id x y z [parametric_data...]"
                // Keep only the first 4 fields: id x y z
                std::istringstream iss(line);
                int id;
                double x, y, z;
                iss >> id >> x >> y >> z;
                ss << id << " " << x << " " << y << " " << z << "\n";
              }
          }
        else
          {
            ss << line << "\n";
          }
      }

    grid_in.read_msh(ss);

    // Partition the serial mesh for MPI distribution
    GridTools::partition_triangulation(mpi_size, serial_mesh);

    // Build the fully distributed triangulation
    auto construction_data =
      TriangulationDescription::Utilities::create_description_from_triangulation(
        serial_mesh, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);
  }

  double U_mean = 0.0;
  if (dim == 2)
    U_mean = (2.0 / 3.0) * U_m;
  else
    U_mean = (4.0 / 9.0) * U_m; // Check paper equation for specific 3D case

  // Re = (U_mean * D) / nu  ->  nu = (U_mean * D) / Re
  nu = (U_mean * D) / Re;

  pcout << "  Reynolds number: " << Re << std::endl;
  pcout << "  U_max (Inlet param): " << U_m << std::endl;
  pcout << "  U_mean (Reference): " << U_mean << std::endl;
  pcout << "  Cylinder Diameter (D): " << D << std::endl;
  pcout << "  Computed Kinematic viscosity (nu): " << nu << std::endl;
  pcout << "  Time step: " << deltat << std::endl;

  // 1. DoFHandler 
  dof_handler.reinit(mesh);
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

  // Verification of Boundary IDs (Cause of errors in assembly if mismatch between mesh and code)
  {
    const std::vector<types::boundary_id> ids = mesh.get_boundary_ids();
    pcout << "  Boundary IDs found in mesh: ";
    for (auto id : ids)
      pcout << id << " ";
    pcout << std::endl;

    pcout << "  Expected IDs: Inlet=" << inlet_boundary_id 
          << ", Walls=" << wall_boundary_id 
          << ", Cylinder=" << cylinder_boundary_id << std::endl;

    // Safety check: if expected IDs are missing, assign them geometrically
    bool has_inlet = false, has_walls = false, has_cylinder = false;
    for (auto id : ids)
      {
        if (id == inlet_boundary_id)    has_inlet    = true;
        if (id == wall_boundary_id)     has_walls    = true;
        if (id == cylinder_boundary_id) has_cylinder = true;
      }

    if (!has_inlet || !has_walls || !has_cylinder)
      {
        pcout << "  WARNING: Expected boundary IDs not found! "
              << "Assigning boundary IDs geometrically..." << std::endl;

        const double tol   = 1e-6;
        const double cx    = 0.2;
        const double cy    = 0.2;
        const double L     = 2.2;
        const double r_cyl = D / 2.0;

        for (const auto &cell : mesh.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                if (!cell->face(f)->at_boundary())
                  continue;

                const auto center = cell->face(f)->center();

                if constexpr (dim == 2)
                  {
                    const double x = center[0];
                    const double y = center[1];
                    const double dist_cyl = std::sqrt((x - cx) * (x - cx) +
                                                       (y - cy) * (y - cy));

                    if (dist_cyl < r_cyl + 0.02)
                      cell->face(f)->set_boundary_id(cylinder_boundary_id);
                    else if (std::abs(x) < tol)
                      cell->face(f)->set_boundary_id(inlet_boundary_id);
                    else if (std::abs(x - L) < tol)
                      cell->face(f)->set_boundary_id(outlet_boundary_id);
                    else
                      cell->face(f)->set_boundary_id(wall_boundary_id);
                  }
              }
          }

        const std::vector<types::boundary_id> new_ids = mesh.get_boundary_ids();
        pcout << "  Boundary IDs after geometric assignment: ";
        for (auto id : new_ids)
          pcout << id << " ";
        pcout << std::endl;
      }
  }

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
  
  // TODO: Maybe set initial condition here (understand if it's nedeed)

  // 6. Matrices initialization
  // Block dynamic sparsity pattern
  BlockDynamicSparsityPattern bdsp(block_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, bdsp);


  // TODO: Stesso problema per la mesh
  system_matrix.reinit(block_owned_dofs, bdsp, MPI_COMM_WORLD);
  pressure_mass.reinit(block_owned_dofs, bdsp, MPI_COMM_WORLD);

  pcout << "Setup complete." << std::endl;
}


template <unsigned int dim>
void NavierStokes<dim>::assemble_newton_system()
{
  system_matrix = 0;
  system_rhs    = 0;
  pressure_mass = 0; // TODO: Recompute if necessary (maybe)

  const QGaussSimplex<dim> quadrature_formula(degree_velocity + 1);

  FEValues<dim> fe_values(*mapping,
                          *fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values); // as usual

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass(dofs_per_cell, dofs_per_cell); 
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Extractors for FEValues (handle vectors and scalars)
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  // Values of the solution at iteration k (current) and at time n-1 (old)
  std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
  std::vector<double>         current_pressure_values(n_q_points);
  
  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);

  // Support vectors for phi_u, grad_phi_u, phi_p
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  // Homogeneous constraints for the Newton UPDATE δu.
  // Only velocity DOFs are constrained (δu_velocity = 0 on Dirichlet
  // boundaries).  Pressure must NOT be constrained here: it has no
  // Dirichlet condition and over-constraining it leads to a singular
  // or trivially-zero solution.
  const ComponentMask velocity_mask = fe->component_mask(velocities);

  AffineConstraints<double> constraints;
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(*mapping,
                                           dof_handler,
                                           inlet_boundary_id,
                                           Functions::ZeroFunction<dim>(dim + 1),
                                           constraints,
                                           velocity_mask);
  VectorTools::interpolate_boundary_values(*mapping,
                                           dof_handler,
                                           wall_boundary_id,
                                           Functions::ZeroFunction<dim>(dim + 1),
                                           constraints,
                                           velocity_mask);
  VectorTools::interpolate_boundary_values(*mapping,
                                           dof_handler,
                                           cylinder_boundary_id,
                                           Functions::ZeroFunction<dim>(dim + 1),
                                           constraints,
                                           velocity_mask);
  constraints.close();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs    = 0;
          cell_pressure_mass = 0;

          fe_values.reinit(cell);

          // Get current solution 
          fe_values[velocities].get_function_values(current_solution, current_velocity_values);
          fe_values[velocities].get_function_gradients(current_solution, current_velocity_gradients);
          fe_values[pressure].get_function_values(current_solution, current_pressure_values);
          
          fe_values[velocities].get_function_values(solution_old, old_velocity_values);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const double JxW = fe_values.JxW(q);

              // Precompute basis functions for efficiency
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_u[k]      = fe_values[velocities].value(k, q);
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                  phi_p[k]      = fe_values[pressure].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // Now residual and Jacobian assembly.
                  // Note: we move everything to the right-hand side, so the signs are opposite to the standard LHS

                  // Time term: (u_k - u_old) / dt * v
                  const double time_term = (current_velocity_values[q] - old_velocity_values[q]) * phi_u[i] / deltat;
                  
                  // Convective term: (u_k . grad u_k) * v
                  const double conv_term = (current_velocity_values[q] * current_velocity_gradients[q]) * phi_u[i];

                  // Viscous term: nu * (grad u_k : grad v)
                  const double visc_term = nu * scalar_product(current_velocity_gradients[q], grad_phi_u[i]);

                  // Pressure term: - p_k * div v
                  const double pres_term = -current_pressure_values[q] * fe_values[velocities].divergence(i, q);
                  
                  // Incompressibility term: - q * div u_k
                  const double div_term = -phi_p[i] * trace(current_velocity_gradients[q]);

                  cell_rhs(i) += (-time_term - conv_term - visc_term - pres_term - div_term) * JxW;


                  // Newton linearization
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // 1. Mass (from time derivative): 1/dt * (u, v)
                      double val = (phi_u[i] * phi_u[j]) / deltat;

                      // 2. Viscosity: nu * (grad u, grad v)
                      val += nu * scalar_product(grad_phi_u[i], grad_phi_u[j]);

                      // 3. Linearized Convection:
                      // (u_k . grad du) * v + (du . grad u_k) * v
                      val += (current_velocity_values[q] * grad_phi_u[j]) * phi_u[i] +
                             (phi_u[j] * current_velocity_gradients[q]) * phi_u[i];

                      // 4. Pressure: -(p, div v)
                      val -= phi_p[j] * fe_values[velocities].divergence(i, q);

                      // 5. Divergence: -(q, div u)
                      val -= phi_p[i] * fe_values[velocities].divergence(j, q);

                      cell_matrix(i, j) += val * JxW;

                      // Pressure mass matrix for preconditioner
                      cell_pressure_mass(i, j) += phi_p[i] * phi_p[j] * JxW;
                    }
                }
            }

          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
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
  const double rhs_norm = system_rhs.l2_norm();
  SolverControl solver_control(20000, 1e-4 * rhs_norm);

  // Use block-triangular preconditioner with AMG on velocity block
  // The pressure Schur complement for time-dependent NS is:
  //   S = -(dt/rho) M_p  =>  S^{-1} ≈ -(rho/dt) M_p^{-1}
  // So the pressure_scaling must be -rho/deltat.
  const double pressure_scaling = -rho / deltat;

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            system_matrix.block(1, 0),
                            pressure_scaling);

  // GMRES with restart=150 for saddle-point systems
  typename SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData
    gmres_data(150 /*restart after*/);
  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control, gmres_data);

  // Solve for newton_update (delta u)
  newton_update = 0.0;
  solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
}

// --- ADDED FUNCTION FOR PRESSURE DROP CALCULATION ---
template <unsigned int dim>
double NavierStokes<dim>::compute_pressure_difference() 
{
    // Benchmark points coordinates (Schaefer-Turek)
    // 2D: front=(0.15, 0.2), end=(0.25, 0.2)
    // 3D: front=(0.45, 0.2, 0.205), end=(0.55, 0.2, 0.205)
    
    Point<dim> p_front, p_end;
    if (dim == 2) {
        p_front = Point<dim>(0.15, 0.2);
        p_end   = Point<dim>(0.25, 0.2);
    } else {
        p_front = Point<dim>(0.45, 0.2, 0.205);
        p_end   = Point<dim>(0.55, 0.2, 0.205);
    }

    double press_front = 0.0;
    double press_end   = 0.0;

    // Use VectorTools::point_value. 
    // Note: throws exception if the point is not in the local cell.
    
    // Front pressure
    try {
        // Extract pressure component (dim)
        Vector<double> value(dim+1);
        VectorTools::point_value(*mapping, dof_handler, current_solution, p_front, value);
        press_front = value[dim];
    } catch (...) {
        // Point not in this process
        press_front = 0.0;
    }

    // End pressure
    try {
        Vector<double> value(dim+1);
        VectorTools::point_value(*mapping, dof_handler, current_solution, p_end, value);
        press_end = value[dim];
    } catch (...) {
        press_end = 0.0;
    }

    // MPI sum to get the value from the process that found it
    press_front = Utilities::MPI::sum(press_front, MPI_COMM_WORLD);
    press_end   = Utilities::MPI::sum(press_end, MPI_COMM_WORLD);

    return press_front - press_end;
}


template <unsigned int dim>
void NavierStokes<dim>::compute_lift_drag(double &drag_coeff, double &lift_coeff) const
{
  // Compute forces on the cylinder
  // F = Integral_S (sigma * n) dS
  // sigma = -pI + nu * (grad u + grad u^T)

  double force_x = 0.0;
  double force_y = 0.0;

  const QGaussSimplex<dim - 1> face_quadrature_formula(degree_velocity + 1);

  FEFaceValues<dim> fe_face_values(*mapping,
                                   *fe,
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

                      // Stress tensor: sigma = -pI + rho*nu*(grad_u + grad_u^T)
                      // Note: rho=1 in the code, but we use the member variable rho for correctness
                      Tensor<2, dim> stress;
                      for (unsigned int i = 0; i < dim; ++i)
                        stress[i][i] = -p;
                      
                      // Viscous part (symmetric tensor)
                      // stress += rho * nu * (grad_u + transpose(grad_u));
                      // In deal.II:
                      stress += rho * nu * (grad_u + transpose(grad_u));

                      // Local force = stress * normal
                      Tensor<1, dim> force_loc = stress * normal;

                      force_x += force_loc[0] * JxW;
                      force_y += force_loc[1] * JxW; // Lift direction
                    }
                }
            }
        }
    }

  force_x = Utilities::MPI::sum(force_x, MPI_COMM_WORLD);
  force_y = Utilities::MPI::sum(force_y, MPI_COMM_WORLD);

  // Cd = 2 * Fd / (rho * U_mean^2 * D * H) (3D) or *L (2D, L=1)
  
  double U_mean = 0.0;
  if (dim == 2) U_mean = (2.0/3.0) * U_m;
  else          U_mean = (4.0/9.0) * U_m;

  // Reference area. 2D: D. 3D: D*H
  double ref_area = D;
  if (dim == 3) ref_area = D * H;

  const double den = 0.5 * rho * U_mean * U_mean * ref_area;
  
  drag_coeff = force_x / den;
  lift_coeff = force_y / den;
}

template <unsigned int dim>
void NavierStokes<dim>::output(const unsigned int time_step)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  
  // Partition vectors for output
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.push_back("pressure");
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(current_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  
  // Add subdomain ids for MPI debugging
  Vector<float> subdomain(mesh.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = mesh.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(*mapping);

  data_out.write_vtu_with_pvtu_record("./", "solution", time_step, MPI_COMM_WORLD, 4);
}


template <unsigned int dim>
void NavierStokes<dim>::run()
{
  setup();

  // Solution initialization
  // Interpolate BC at initial time (t=0)
  inlet_velocity->set_time(0.0);
  
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           Functions::ZeroFunction<dim>(dim + 1), // Start from rest
                           solution_owned);
  solution = solution_owned;
  solution_old = solution;
  current_solution = solution;

  double time = 0.0;
  unsigned int time_step = 0;

  std::ofstream forces_file;
  if (mpi_rank == 0)
    {
      forces_file.open("forces.txt");
      // Output file format required for analysis:
      // Time | Drag Coeff | Lift Coeff | Delta P
      forces_file << "Time\tCd\tCl\tDeltaP" << std::endl;
    }

  while (time < T)
    {
      time += deltat;
      time_step++;
      
      if (mpi_rank == 0) pcout << "Time step " << time_step << " at t=" << time << std::flush;

      // Update BC
      inlet_velocity->set_time(time);
      
      // ---- LIFT NON-HOMOGENEOUS BCs ONTO current_solution ----
      {
        const FEValuesExtractors::Vector velocities(0);
        const ComponentMask velocity_mask = fe->component_mask(velocities);

        std::map<types::global_dof_index, double> boundary_values;

        // Inlet: non-homogeneous (parabolic profile)
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 inlet_boundary_id,
                                                 *inlet_velocity,
                                                 boundary_values,
                                                 velocity_mask);
        // Walls: no-slip
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 wall_boundary_id,
                                                 Functions::ZeroFunction<dim>(dim + 1),
                                                 boundary_values,
                                                 velocity_mask);
        // Cylinder: no-slip
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 cylinder_boundary_id,
                                                 Functions::ZeroFunction<dim>(dim + 1),
                                                 boundary_values,
                                                 velocity_mask);

        // Apply on the OWNED vector
        solution_owned = current_solution;
        for (const auto &[dof, val] : boundary_values)
          {
            if (locally_owned_dofs.is_element(dof))
              solution_owned(dof) = val;
          }

        // Refresh ghost layer
        current_solution = solution_owned;
      }


      double residual_norm = 1e10;
      double previous_residual = 1e10;
      unsigned int newton_iter = 0;
      double damping = 1.0;

      while (residual_norm > newton_tolerance && newton_iter < newton_max_iterations)
        {
          assemble_newton_system();
          
          residual_norm = system_rhs.l2_norm();
          pcout << " [" << newton_iter << ": " << residual_norm;
          if (damping < 1.0 - 1e-12)
            pcout << " a=" << damping;
          pcout << "]" << std::flush;

          if (residual_norm < newton_tolerance) break;

          // Adaptive damping: reduce step when residual increases
          if (newton_iter > 0 && residual_norm > previous_residual)
            {
              damping = std::max(0.1, damping * 0.5);
            }
          else if (damping < 1.0 - 1e-12)
            {
              damping = std::min(1.0, damping * 2.0);
            }
          previous_residual = residual_norm;

          try
            {
              solve_newton_system();
            }
          catch (const std::exception &e)
            {
              pcout << "(linfail)" << std::flush;
              damping = std::max(0.1, damping * 0.25);
            }

          // Apply Newton update with damping
          solution_owned = current_solution;
          solution_owned.add(damping, newton_update);
          current_solution = solution_owned;

          newton_iter++;
        }
      
      pcout << " Newton: " << newton_iter << " iters, res=" << residual_norm << std::endl;

      if (mpi_rank == 0) pcout << std::endl;

      solution_old = current_solution;

      // Benchmark quantities calculation
      double drag, lift, delta_p;
      compute_lift_drag(drag, lift);
      delta_p = compute_pressure_difference();

      if (mpi_rank == 0)
        {
           forces_file << time << "\t" << drag << "\t" << lift << "\t" << delta_p << std::endl;
           forces_file.flush();
        }

      // Output frequency - Test: output every step
      if (time_step % 1 == 0) 
        output(time_step);
    }
}

// Explicit instantiation
template class NavierStokes<2>;
template class NavierStokes<3>;