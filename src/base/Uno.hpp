#ifndef UNO_H
#define UNO_H

#include "Problem.hpp"
#include "GlobalizationMechanism.hpp"

struct Result {
   TerminationStatus status;
   Iterate solution;
   size_t number_variables;
   size_t number_constraints;
   int iteration;
   double cpu_time;
   int objective_evaluations;
   int constraint_evaluations;
   int jacobian_evaluations;
   int hessian_evaluations;
   int number_subproblems_solved;

   void display(bool print_solution);
};

/*! \class Uno
 * \brief Uno
 *
 *  UNO solver
 */
class Uno {
public:
   /*!
    *  Constructor
    *
    * \param globalization_strategy: strategy to promote global convergence
    * \param tolerance: tolerance for termination criteria
    */
   Uno(GlobalizationMechanism& globalization_mechanism, double tolerance, int max_iterations);

   /*!
    *  Solve a given problem with initial primal and dual variables
    *
    * \param problem: optimization problem
    * \param x: primal variables
    * \param multipliers: Lagrange multipliers/dual variables
    */
   Result solve(Problem& problem, std::vector<double>& x, Multipliers& multipliers, bool preprocessing);
   static void preprocessing(Problem& problem, std::vector<double>& x, Multipliers& multipliers);

   GlobalizationMechanism& globalization_mechanism; /*!< Step control strategy (trust region or line-search) */
   double tolerance; /*!< Tolerance of the termination criteria */
   int max_iterations; /*!< Maximum number of iterations */

private:
   static Statistics create_statistics();
   [[nodiscard]] bool termination_criterion_(TerminationStatus is_optimal, int iteration) const;
   TerminationStatus
   compute_termination_status_(Problem& problem, Iterate& current_iterate, double step_norm, double objective_multiplier) const;
};

#endif // UNO_H
