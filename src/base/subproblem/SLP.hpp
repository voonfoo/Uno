#ifndef SLP_H
#define SLP_H

#include "ActiveSetMethod.hpp"

class SLP : public ActiveSetMethod {
public:
    SLP(Problem& problem, std::string QP_solver_name, bool use_trust_region, bool scale_residuals);

    Direction compute_step(Problem& problem, Iterate& current_iterate, double trust_region_radius = INFINITY) override;
    Direction restore_feasibility(Problem& problem, Iterate& current_iterate, Direction& phase_2_direction, double trust_region_radius = INFINITY) override;
    
    /* use references to allow polymorphism */
    std::shared_ptr<QPSolver> solver; /*!< Solver that solves the subproblem */
    
private:
    void evaluate_optimality_iterate_(Problem& problem, Iterate& current_iterate);
    void evaluate_feasibility_iterate_(Problem& problem, Iterate& current_iterate, Direction& phase_2_direction);
};

#endif // SLP_H
