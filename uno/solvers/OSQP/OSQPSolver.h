//
// Created by Voon Foo on 4/12/24.
//

#ifndef UNO_OSQPSOLVER_H
#define UNO_OSQPSOLVER_H

#include "solvers/QPSolver.hpp"
#include <osqp.h>

namespace uno {
    // forward declaration
    class Options;

    class OsqpSolver : public QPSolver {
    public:
        void
        solve_LP(size_t number_variables, size_t number_constraints, const std::vector<double> &variables_lower_bounds,
                 const std::vector<double> &variables_upper_bounds, const std::vector<double> &constraints_lower_bounds,
                 const std::vector<double> &constraints_upper_bounds, const SparseVector<double> &linear_objective,
                 const RectangularMatrix<double> &constraint_jacobian, const Vector<double> &initial_point,
                 Direction &direction,
                 const WarmstartInformation &warmstart_information) override;

        void
        solve_QP(size_t number_variables, size_t number_constraints, const std::vector<double> &variables_lower_bounds,
                 const std::vector<double> &variables_upper_bounds, const std::vector<double> &constraints_lower_bounds,
                 const std::vector<double> &constraints_upper_bounds, const SparseVector<double> &linear_objective,
                 const RectangularMatrix<double> &constraint_jacobian, const SymmetricMatrix<size_t, double> &hessian,
                 const Vector<double> &initial_point,
                 Direction &direction, const WarmstartInformation &warmstart_information) override;

    private:
        void create_empty_csc(OSQPCscMatrix *P, size_t m, size_t n) ;
        void save_matrix_to_local_format(OSQPCscMatrix *P, size_t m, size_t n, const RectangularMatrix<double> &matrix);
    };

    struct Triplets {
    public:
        int rowIdx;
        int columnIdx;
        double value;
    };
}

#endif //UNO_OSQPSOLVER_H
