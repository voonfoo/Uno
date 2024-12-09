//
// Created by Voon Foo on 4/12/24.
//

#include <gtest/gtest.h>
#include "optimization/Direction.hpp"
#include "linear_algebra/RectangularMatrix.hpp"
#include "linear_algebra/SparseVector.hpp"
#include "optimization/WarmstartInformation.hpp"
#include "options/Options.hpp"
#include "solvers/OSQP/OSQPSolver.h"
#include "tools/Infinity.hpp"

using namespace uno;

TEST(OsqpSolver, LP) {
    // https://ergo-code.github.io/HiGHS/stable/interfaces/cpp/library/
    // Min    f  =  x_0 +  x_1 + 3
    // s.t.                x_1 <= 7
    //        5 <=  x_0 + 2x_1 <= 15
    //        6 <= 3x_0 + 2x_1
    // 0 <= x_0 <= 4; 1 <= x_1

    const size_t number_variables = 2;
    const size_t number_constraints = 3;
    const size_t number_jacobian_nonzeros = 5;
    const size_t number_hessian_nonzeros = 0;
    Options options(false);

    Logger::set_logger("DEBUG2");
    options["print_subproblem"] = "false";
    OsqpSolver osqp_solver;

    SparseVector<double> linear_objective(number_variables);
    linear_objective.insert(0, 1.);
    linear_objective.insert(1, 1.);

    const std::vector<double> variables_lower_bounds{0., 1.};
    const std::vector<double> variables_upper_bounds{4., OSQP_INFTY};
    const std::vector<double> constraints_lower_bounds{-OSQP_INFTY, 5., 6.};
    const std::vector<double> constraints_upper_bounds{7., 15., OSQP_INFTY};

    RectangularMatrix<double> constraint_jacobian(number_constraints, number_variables);
    constraint_jacobian[0].insert(0, 0.);
    constraint_jacobian[0].insert(1, 1.);
    constraint_jacobian[1].insert(0, 1.);
    constraint_jacobian[1].insert(1, 2.);
    constraint_jacobian[2].insert(0, 3.);
    constraint_jacobian[2].insert(1, 2.);
    constraint_jacobian[3].insert(0, 1.);
    constraint_jacobian[3].insert(1, 0.);
    constraint_jacobian[4].insert(0, 0.);
    constraint_jacobian[4].insert(1, 1.);

    Direction direction(number_variables, number_constraints);
    WarmstartInformation warmstart_information{};
    Vector<double> initial_point{0., 0.};

    osqp_solver.solve_LP(number_variables, number_constraints, variables_lower_bounds, variables_upper_bounds,
                     constraints_lower_bounds,
                     constraints_upper_bounds, linear_objective, constraint_jacobian, initial_point, direction,
                     warmstart_information
    );

    ASSERT_EQ(direction.status, SubproblemStatus::OPTIMAL);

    const double tolerance = 1e-8;
    // check primals
    const std::vector<double> primals_reference{0.5, 2.25};
    for (size_t index: Range(number_variables)) {
        EXPECT_NEAR(direction.primals[index], primals_reference[index], tolerance);
    }
    // check duals
    const std::vector<double> constraint_duals_reference{0., 0.25, 0.25};
    for (size_t index: Range(number_constraints)) {
        EXPECT_NEAR(direction.multipliers.constraints[index], constraint_duals_reference[index], tolerance);
    }
    const std::vector<double> lower_bound_duals_reference{0., 0.};
    const std::vector<double> upper_bound_duals_reference{0., 0.};
    for (size_t index: Range(number_variables)) {
        EXPECT_NEAR(direction.multipliers.lower_bounds[index], lower_bound_duals_reference[index], tolerance);
        EXPECT_NEAR(direction.multipliers.upper_bounds[index], upper_bound_duals_reference[index], tolerance);
    }
}