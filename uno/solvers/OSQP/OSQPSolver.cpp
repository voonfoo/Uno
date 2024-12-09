//
// Created by Voon Foo on 4/12/24.
//

#include "OSQPSolver.h"
#include "linear_algebra/RectangularMatrix.hpp"
#include "linear_algebra/SymmetricMatrix.hpp"
#include "linear_algebra/SparseVector.hpp"
#include "linear_algebra/Vector.hpp"
#include "optimization/WarmstartInformation.hpp"
#include "optimization/Direction.hpp"

namespace uno {
    // Min    f  =  x_0 +  x_1 + 3
    // s.t.                x_1 <= 7
    //        5 <=  x_0 + 2x_1 <= 15
    //        6 <= 3x_0 + 2x_1
    // 0 <= x_0 <= 4; 1 <= x_1
    void OsqpSolver::solve_LP(size_t number_variables, size_t number_constraints,
                              const std::vector<double> &variables_lower_bounds,
                              const std::vector<double> &variables_upper_bounds,
                              const std::vector<double> &constraints_lower_bounds,
                              const std::vector<double> &constraints_upper_bounds,
                              const SparseVector<double> &linear_objective,
                              const RectangularMatrix<double> &constraint_jacobian, const Vector<double> &initial_point,
                              uno::Direction &direction, const uno::WarmstartInformation &warmstart_information) {
        OSQPSolver *solver = nullptr;
        auto *settings = (OSQPSettings *) malloc(sizeof(OSQPSettings));
        osqp_set_default_settings(settings);
        settings->eps_rel = 1e-9;
        settings->eps_abs = 1e-9;
        auto m = number_constraints + number_variables;
        auto n = number_variables;

        auto pEmpty = RectangularMatrix<double>(number_variables, number_variables);
        for (size_t i = 0; i < number_variables; i++) {
            pEmpty[i] = SparseVector<double>(number_variables);
            for (size_t j = 0; j < number_variables; j++) {
                pEmpty[i].insert(j, 0.0);
            }
        }
        auto *P = (OSQPCscMatrix *) malloc(sizeof(OSQPCscMatrix));
        create_empty_csc(P, n, n);

        auto *A = (OSQPCscMatrix *) malloc(sizeof(OSQPCscMatrix));
        save_matrix_to_local_format(A, m, n, constraint_jacobian);

        std::vector<OSQPFloat> q_vector;
        for (const auto [variable_index, value]: linear_objective) {
            q_vector.push_back(value);
        }

        auto *q = q_vector.data();

        std::vector<OSQPFloat> lb;
        std::vector<OSQPFloat> ub;

        for (size_t constraint_index = 0; constraint_index < number_constraints; constraint_index++) {
            lb.push_back(constraints_lower_bounds[constraint_index]);
            ub.push_back(constraints_upper_bounds[constraint_index]);
        }

        for (size_t variable_index = 0; variable_index < number_variables; variable_index++) {
            lb.push_back(variables_lower_bounds[variable_index]);
            ub.push_back(variables_upper_bounds[variable_index]);
        }

        auto *l = lb.data();
        auto *u = ub.data();

        OSQPInt exitflag = 0;
        exitflag = osqp_setup(&solver, P, q, A, l, u, static_cast<OSQPInt>(m),
                              static_cast<OSQPInt>(n), settings);

        if (exitflag) {
            osqp_cleanup(solver);
            if (A) free(A);
            if (P) free(P);
            if (settings) free(settings);
            direction.status = SubproblemStatus::ERROR;
            return;
        }

        exitflag = osqp_solve(solver);

        auto status = static_cast<osqp_status_type>(solver->info->status_val);
        if (status == OSQP_SOLVED) {
            direction.status = SubproblemStatus::OPTIMAL;
        } else if (status == OSQP_PRIMAL_INFEASIBLE) {
            direction.status = SubproblemStatus::INFEASIBLE;
        } else if (status == OSQP_DUAL_INFEASIBLE) {
            direction.status = SubproblemStatus::OPTIMAL;
        } else {
            direction.status = SubproblemStatus::ERROR;
        }

        auto solution = solver->solution;
        for (size_t variable_index = 0; variable_index < number_variables; variable_index++) {
            direction.primals[variable_index] = solution->x[variable_index];
        }

        for (size_t constraint_index = 0; constraint_index < number_constraints; constraint_index++) {
            direction.multipliers.constraints[constraint_index] = -solution->y[constraint_index];
        }

        direction.subproblem_objective = solver->info->obj_val;

        osqp_cleanup(solver);
        if (A) free(A);
        if (P) free(P);
        if (settings) free(settings);
    }

    void OsqpSolver::create_empty_csc(OSQPCscMatrix *P, size_t m, size_t n) {
        if (!P) return;

        P->m = static_cast<OSQPInt>(m);
        P->n = static_cast<OSQPInt>(n);
        P->nz = -1;
        P->nzmax = 0;
        P->p = (OSQPInt *) malloc((n + 1) * sizeof(OSQPInt));
        P->i = (OSQPInt *) malloc((0) * sizeof(OSQPInt));
        P->x = (OSQPFloat *) malloc((0) * sizeof(OSQPFloat));
    }

    void OsqpSolver::save_matrix_to_local_format(OSQPCscMatrix *P, size_t m, size_t n,
                                                 const RectangularMatrix<double> &matrix) {
        std::vector<OSQPFloat> value_array;
        std::vector<OSQPInt> row_indices;
        std::vector<OSQPInt> column_pointers;

        std::vector<Triplets> triplets;

        for (size_t constraint_index = 0; constraint_index < m; constraint_index++) {
            for (const auto [variable_index, value]: matrix[constraint_index]) {
                if (value == 0.0) {
                    continue;
                }
                triplets.emplace_back(
                        Triplets{static_cast<int>(constraint_index), static_cast<int>(variable_index), value});
            }
        }

        // sort the triplets
        std::sort(triplets.begin(), triplets.end(), [](const Triplets &lhs, const Triplets &rhs) {
            if (lhs.columnIdx == rhs.columnIdx) {
                return lhs.rowIdx < rhs.rowIdx;
            }
            return lhs.columnIdx < rhs.columnIdx;
        });

        size_t nnz = triplets.size();
        value_array.resize(nnz);
        row_indices.resize(nnz);
        column_pointers.resize(n + 1);

        for (size_t j = 0; j < nnz; ++j) {
            value_array[j] = triplets[j].value;
            row_indices[j] = triplets[j].rowIdx;
            column_pointers[triplets[j].columnIdx + 1]++;
        }

        // Step 4: Compute cumulative sums for col_pointers
        for (size_t i = 1; i <= m; ++i) {
            column_pointers[i] += column_pointers[i - 1];
        }

        OSQPFloat *x = new OSQPFloat[nnz];
        std::copy(value_array.begin(), value_array.end(), x);
        OSQPInt *i = new OSQPInt[nnz];
        std::copy(row_indices.begin(), row_indices.end(), i);
        OSQPInt *p = new OSQPInt[n + 1];
        std::copy(column_pointers.begin(), column_pointers.end(), p);

        csc_set_data(P,
                     static_cast<OSQPInt>(m),
                     static_cast<OSQPInt>(n),
                     static_cast<OSQPInt>(nnz),
                     x, i, p);
    }

    void OsqpSolver::solve_QP(size_t number_variables, size_t number_constraints,
                              const std::vector<double> &variables_lower_bounds,
                              const std::vector<double> &variables_upper_bounds,
                              const std::vector<double> &constraints_lower_bounds,
                              const std::vector<double> &constraints_upper_bounds,
                              const SparseVector<double> &linear_objective,
                              const RectangularMatrix<double> &constraint_jacobian,
                              const SymmetricMatrix<size_t, double> &hessian, const Vector<double> &initial_point,
                              uno::Direction &direction, const uno::WarmstartInformation &warmstart_information) {

    }
}
