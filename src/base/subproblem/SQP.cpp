#include <cmath>
#include <map>
#include "SQP.hpp"
#include "Constraint.hpp"
#include "Utils.hpp"
#include "Logger.hpp"
#include "QPSolverFactory.hpp"

SQP::SQP(Problem& problem, std::string QP_solver_name, std::string hessian_evaluation_method, bool use_trust_region, bool scale_residuals):
// maximum number of Hessian nonzeros = number nonzeros + possible diagonal inertia correction
ActiveSetMethod(problem, scale_residuals),
solver(QPSolverFactory::create(QP_solver_name, problem.number_variables, problem.number_constraints, problem.hessian_maximum_number_nonzeros + problem.number_variables, true)),
/* if no trust region is used, the problem should be convexified by controlling the inertia of the Hessian */
hessian_evaluation(HessianEvaluationFactory::create(hessian_evaluation_method, problem.number_variables, !use_trust_region)) {
}

Direction SQP::compute_step(Problem& problem, Iterate& current_iterate, double trust_region_radius) {
    /* compute optimality step */
    this->evaluate_optimality_iterate_(problem, current_iterate);
    Direction direction = this->compute_qp_step_(problem, this->solver, current_iterate, trust_region_radius);
    
    if (direction.status == INFEASIBLE) {
        /* infeasible subproblem during optimality phase */
        direction = this->restore_feasibility(problem, current_iterate, direction, trust_region_radius);
    }
    // the solution is now feasible
    
    direction.objective_multiplier = problem.objective_sign;
    direction.predicted_reduction = [&](double step_length) {
        return this->compute_qp_predicted_reduction_(current_iterate, direction, step_length);
    };
    return direction;
}

Direction SQP::restore_feasibility(Problem& problem, Iterate& current_iterate, Direction& phase_II_direction, double trust_region_radius) {
    this->evaluate_feasibility_iterate_(problem, current_iterate, phase_II_direction.constraint_partition);
   return this->compute_l1qp_step_(problem, this->solver, current_iterate, phase_II_direction.constraint_partition, phase_II_direction.x, trust_region_radius); 
}

/* private methods */

void SQP::evaluate_optimality_iterate_(Problem& problem, Iterate& current_iterate) {
    /* compute first- and second-order information */
    current_iterate.compute_objective_gradient(problem);
    current_iterate.compute_constraints_jacobian(problem);
    this->hessian_evaluation->compute(problem, current_iterate, problem.objective_sign, current_iterate.multipliers.constraints);
    return;
}

void SQP::evaluate_feasibility_iterate_(Problem& problem, Iterate& current_iterate, ConstraintPartition& constraint_partition) {
    /* update the multipliers of the general constraints */
    std::vector<double> constraint_multipliers = this->generate_l1_multipliers_(problem, current_iterate.multipliers.constraints, constraint_partition);
    /* compute first- and second-order information */
    current_iterate.compute_constraints_jacobian(problem);
    current_iterate.is_hessian_computed = false;
    double objective_multiplier = 0.;
    this->hessian_evaluation->compute(problem, current_iterate, objective_multiplier, constraint_multipliers);
    return;
}