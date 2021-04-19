#ifndef SUBPROBLEM_H
#define SUBPROBLEM_H

#include <cmath>
#include <vector>
#include <memory>
#include "Problem.hpp"
#include "Iterate.hpp"
#include "Phase.hpp"
#include "Direction.hpp"
#include "Constraint.hpp"
#include "MA57Solver.hpp"

/*! \class Subproblem
 * \brief Subproblem
 *
 *  Local approximation of a nonlinear optimization problem (virtual class) 
 */
class Subproblem {
public:
    /*!
     *  Constructor
     * 
     * \param solver: solver that solves the subproblem
     * \param name: name of the strategy
     */
    Subproblem(std::string residual_norm, std::vector<Range>& subproblem_variables_bounds, bool scale_residuals);
    virtual ~Subproblem();

    virtual Iterate evaluate_initial_point(const Problem& problem, const std::vector<double>& x, const Multipliers& multipliers) = 0;
    
    virtual std::vector<Direction> compute_directions(Problem& problem, Iterate& current_iterate, double trust_region_radius=INFINITY) = 0;
    virtual std::vector<Direction> restore_feasibility(Problem& problem, Iterate& current_iterate, Direction& phase_2_direction, double trust_region_radius=INFINITY) = 0;
    
    virtual void compute_optimality_measures(const Problem& problem, Iterate& iterate) = 0;
    virtual void compute_infeasibility_measures(const Problem& problem, Iterate& iterate, const Direction& direction) = 0;
    
    static void project_point_in_bounds(std::vector<double>& x, const std::vector<Range>& variables_bounds);
    static double project_strictly_variable_in_bounds(double variable_value, const Range& variable_bounds);
    static std::vector<Range> generate_constraints_bounds(const Problem& problem, const std::vector<double>& current_constraints);
    static std::vector<double> compute_least_square_multipliers(const Problem& problem, Iterate& current_iterate, const std::vector<double>& default_multipliers, LinearSolver& solver, double multipliers_max_size=1e3);
    static std::vector<double> compute_least_square_multipliers(const Problem& problem, Iterate& current_iterate, const std::vector<double>& default_multipliers, double multipliers_max_size=1e3);
    
    double compute_KKT_error(const Problem& problem, Iterate& iterate, double objective_mutiplier) const;
    void compute_residuals(const Problem& problem, Iterate& iterate, const Multipliers& multipliers, double objective_multiplier) const;
    
    std::string residual_norm;
    // when the subproblem is reformulated (e.g. when slacks are introduced), the bounds may be altered as well
    std::vector<Range> subproblem_variables_bounds;
    int number_subproblems_solved;
    // when the parameterization of the subproblem (e.g. penalty or barrier parameter) is updated, signal it
    bool subproblem_definition_changed;
    bool scale_residuals;
    
protected:
    virtual double compute_complementarity_error_(const Problem& problem, Iterate& iterate, const Multipliers& multipliers) const;
};

#endif // SUBPROBLEM_H
