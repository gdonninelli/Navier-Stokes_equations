#ifndef TEST_CASES_HPP
#define TEST_CASES_HPP

#include "NavierStokes.hpp"

// Available test cases (circular cylinder only):
//   2D-1   steady,   Re=20,       U_m=0.3
//   2D-2   unsteady, Re=100,      U_m=1.5  (constant inlet)
//   2D-3   unsteady, Re(t) between 0 and 100, U_m=1.5  (sin(pi*t/8) inlet)
//   3D-1Z  steady,   Re=20,       U_m=0.45
//   3D-2Z  unsteady, Re=100,      U_m=2.25 (constant inlet)
//   3D-3Z  unsteady, Re(t) between 0 and 100, U_m=2.25 (sin(pi*t/8) inlet)

template <unsigned int dim>
class BenchmarkInletVelocity : public Function<dim>
{
public:
  BenchmarkInletVelocity(const double H_,
                         const double U_m_,
                         const bool   time_dep,
                         const double T_ramp_ = 0.0)
    : Function<dim>(dim + 1)
    , H(H_)
    , U_m(U_m_)
    , time_dependent(time_dep)
    , T_ramp(T_ramp_)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    // 2D: flow in x-direction (component 0), profile depends on y
    // 3D: flow in z-direction (component 2), profile depends on x,y
    constexpr unsigned int flow_component = (dim == 2) ? 0 : 2;

    if (component == flow_component)
      {
        double profile = 0.0;

        if constexpr (dim == 2)
          {
            const double y = p[1];
            profile = 4.0 * U_m * y * (H - y) / (H * H);
          }
        else if constexpr (dim == 3)
          {
            const double x = p[0];
            const double y = p[1];
            profile =
              16.0 * U_m * x * y * (H - x) * (H - y) / (H * H * H * H);
          }

        if (time_dependent)
          {
            const double t = this->get_time();
            profile *= std::sin(M_PI * t / 8.0);
          }

        // Smooth temporal ramp for impulsive-start cases (e.g. 2D-2).
        // Uses a half-cosine ramp: 0.5*(1 - cos(pi*t/T_ramp)) for t < T_ramp.
        // This avoids the discontinuity u(0)=0 -> u(dt)=U_m that makes
        // the first Newton/GMRES iterations diverge.
        if (T_ramp > 0.0)
          {
            const double t = this->get_time();
            if (t < T_ramp)
              profile *= 0.5 * (1.0 - std::cos(M_PI * t / T_ramp));
            // else: ramp = 1.0 (full velocity)
          }

        return profile;
      }

    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < dim + 1; ++c)
      values[c] = value(p, c);
  }

protected:
  const double H;
  const double U_m;
  const bool   time_dependent;
  const double T_ramp; // Ramp-up duration [s]; 0 = no ramp
};



//* Optional parameters:
//   ts     — time integration scheme   (default depends on test case)
//   nm     — nonlinear solution method  (default depends on test case)
//   deltat — time step (<= 0 automatic selection based on Re)
namespace TestCases
{

//! 2D-1: Steady flow, Re = 20
inline BenchmarkTestCase<2>
make_2D_1(const std::string  &mesh_file,
          TimeScheme          ts     = TimeScheme::BackwardEuler,
          NonlinearMethod     nm     = NonlinearMethod::Newton,
          double              deltat = -1.0,
          double              t_ramp = 1.0)
{
  constexpr double H   = 0.41;
  constexpr double U_m = 0.3;
  constexpr double Re  = 20.0;
  constexpr double T   = 10.0; // run until steady state

  BenchmarkTestCase<2> tc;
  tc.name              = "2D-1";
  tc.description       = "Steady flow around cylinder, Re=20, U_m=0.3";
  tc.mesh_file         = mesh_file;
  tc.degree_velocity   = 2;
  tc.degree_pressure   = 1;
  tc.Re                = Re;
  tc.U_m               = U_m;
  tc.T                 = T;
  tc.deltat            = deltat;
  tc.time_scheme       = ts;
  tc.nonlinear_method  = nm;
  tc.inlet_velocity    =
    std::make_shared<BenchmarkInletVelocity<2>>(H, U_m, /*time_dep=*/false, t_ramp);
  tc.dirichlet_bc      = std::make_shared<ZeroDirichletBC<2>>();
  tc.forcing_term      = std::make_shared<ForcingTerm<2>>();
  tc.initial_condition = std::make_shared<InitialCondition<2>>();
  return tc;
}

//! 2D-2: Unsteady flow, Re = 100
inline BenchmarkTestCase<2>
make_2D_2(const std::string  &mesh_file,
          TimeScheme          ts     = TimeScheme::CrankNicolson,
          NonlinearMethod     nm     = NonlinearMethod::Linearized,
          double              deltat = -1.0)
{
  constexpr double H   = 0.41;
  constexpr double U_m = 1.5;
  constexpr double Re  = 100.0;
  constexpr double T   = 8.0;

  BenchmarkTestCase<2> tc;
  tc.name              = "2D-2";
  tc.description       = "Unsteady flow, Re=100, U_m=1.5, constant inlet";
  tc.mesh_file         = mesh_file;
  tc.degree_velocity   = 2;
  tc.degree_pressure   = 1;
  tc.Re                = Re;
  tc.U_m               = U_m;
  tc.T                 = T;
  tc.deltat            = deltat;
  tc.time_scheme       = ts;
  tc.nonlinear_method  = nm;

  // Smooth ramp over 2 seconds to avoid impulsive start.
  // At Re=100 with constant inlet, jumping from 0 to 1.5 m/s in one dt
  // produces (u_new - 0)/dt -> infinity, killing Newton/GMRES.
  tc.inlet_velocity    =
    std::make_shared<BenchmarkInletVelocity<2>>(H, U_m, /*time_dep=*/false,
                                                /*T_ramp=*/2.0);
  tc.dirichlet_bc      = std::make_shared<ZeroDirichletBC<2>>();
  tc.forcing_term      = std::make_shared<ForcingTerm<2>>();
  tc.initial_condition = std::make_shared<InitialCondition<2>>();
  return tc;
}

//! 2D-3: Unsteady flow, time-varying inlet
inline BenchmarkTestCase<2>
make_2D_3(const std::string  &mesh_file,
          TimeScheme          ts     = TimeScheme::CrankNicolson,
          NonlinearMethod     nm     = NonlinearMethod::Linearized,
          double              deltat = -1.0)
{
  constexpr double H   = 0.41;
  constexpr double U_m = 1.5;
  constexpr double Re  = 100.0;
  constexpr double T   = 8.0;

  BenchmarkTestCase<2> tc;
  tc.name              = "2D-3";
  tc.description       = "Unsteady flow, time-varying inlet sin(pi*t/8), "
                         "U_m=1.5, Re(t) in [0,100]";
  tc.mesh_file         = mesh_file;
  tc.degree_velocity   = 2;
  tc.degree_pressure   = 1;
  tc.Re                = Re;
  tc.U_m               = U_m;
  tc.T                 = T;
  tc.deltat            = deltat;
  tc.time_scheme       = ts;
  tc.nonlinear_method  = nm;
  tc.inlet_velocity    =
    std::make_shared<BenchmarkInletVelocity<2>>(H, U_m, /*time_dep=*/true);
  tc.dirichlet_bc      = std::make_shared<ZeroDirichletBC<2>>();
  tc.forcing_term      = std::make_shared<ForcingTerm<2>>();
  tc.initial_condition = std::make_shared<InitialCondition<2>>();
  return tc;
}

//! 3D-1Z: Steady flow, Re = 20
inline BenchmarkTestCase<3>
make_3D_1Z(const std::string  &mesh_file,
           TimeScheme          ts     = TimeScheme::BackwardEuler,
           NonlinearMethod     nm     = NonlinearMethod::Newton,
           double              deltat = -1.0)
{
  constexpr double H   = 0.41;
  constexpr double U_m = 0.45;
  constexpr double Re  = 20.0;
  constexpr double T   = 10.0;

  BenchmarkTestCase<3> tc;
  tc.name              = "3D-1Z";
  tc.description       = "Steady 3D flow, Re=20, U_m=0.45, circular cylinder";
  tc.mesh_file         = mesh_file;
  tc.degree_velocity   = 2;
  tc.degree_pressure   = 1;
  tc.Re                = Re;
  tc.U_m               = U_m;
  tc.T                 = T;
  tc.deltat            = deltat;
  tc.time_scheme       = ts;
  tc.nonlinear_method  = nm;
  tc.inlet_velocity    =
    std::make_shared<BenchmarkInletVelocity<3>>(H, U_m, /*time_dep=*/false);
  tc.dirichlet_bc      = std::make_shared<ZeroDirichletBC<3>>();
  tc.forcing_term      = std::make_shared<ForcingTerm<3>>();
  tc.initial_condition = std::make_shared<InitialCondition<3>>();
  tc.use_supg          = true;
  return tc;
}

//! 3D-2Z: Unsteady flow, Re = 100
inline BenchmarkTestCase<3>
make_3D_2Z(const std::string  &mesh_file,
           TimeScheme          ts     = TimeScheme::CrankNicolson,
           NonlinearMethod     nm     = NonlinearMethod::Linearized,
           double              deltat = -1.0)
{
  constexpr double H   = 0.41;
  constexpr double U_m = 2.25;
  constexpr double Re  = 100.0;
  constexpr double T   = 8.0;

  BenchmarkTestCase<3> tc;
  tc.name              = "3D-2Z";
  tc.description       = "Unsteady 3D flow, Re=100, U_m=2.25, circular "
                         "cylinder, constant inlet";
  tc.mesh_file         = mesh_file;
  tc.degree_velocity   = 2;
  tc.degree_pressure   = 1;
  tc.Re                = Re;
  tc.U_m               = U_m;
  tc.T                 = T;
  tc.deltat            = (deltat > 0) ? deltat : 0.01;
  tc.time_scheme       = ts;
  tc.nonlinear_method  = nm;
  // Smooth ramp over 2 seconds (same rationale as 2D-2)
  tc.inlet_velocity    =
    std::make_shared<BenchmarkInletVelocity<3>>(H, U_m, /*time_dep=*/false,
                                                /*T_ramp=*/4.0);
  tc.dirichlet_bc      = std::make_shared<ZeroDirichletBC<3>>();
  tc.forcing_term      = std::make_shared<ForcingTerm<3>>();
  tc.initial_condition = std::make_shared<InitialCondition<3>>();
  tc.use_supg          = true;
  return tc;
}

//! 3D-3Z: Unsteady flow, time-varying inlet
inline BenchmarkTestCase<3>
make_3D_3Z(const std::string  &mesh_file,
           TimeScheme          ts     = TimeScheme::CrankNicolson,
           NonlinearMethod     nm     = NonlinearMethod::Linearized,
           double              deltat = -1.0)
{
  constexpr double H   = 0.41;
  constexpr double U_m = 2.25;
  constexpr double Re  = 100.0;
  constexpr double T   = 8.0;

  BenchmarkTestCase<3> tc;
  tc.name              = "3D-3Z";
  tc.description       = "Unsteady 3D flow, time-varying inlet sin(pi*t/8), "
                         "U_m=2.25, Re(t) in [0,100], circular cylinder";
  tc.mesh_file         = mesh_file;
  tc.degree_velocity   = 2;
  tc.degree_pressure   = 1;
  tc.Re                = Re;
  tc.U_m               = U_m;
  tc.T                 = T;

  tc.deltat            = (deltat > 0) ? deltat : 0.01;
  tc.time_scheme       = ts;
  tc.nonlinear_method  = nm;

  tc.inlet_velocity    =
    std::make_shared<BenchmarkInletVelocity<3>>(H, U_m, /*time_dep=*/true);
  tc.dirichlet_bc      = std::make_shared<ZeroDirichletBC<3>>();
  tc.forcing_term      = std::make_shared<ForcingTerm<3>>();
  tc.initial_condition = std::make_shared<InitialCondition<3>>();
  tc.use_supg          = true;
  return tc;
}

}

#endif
