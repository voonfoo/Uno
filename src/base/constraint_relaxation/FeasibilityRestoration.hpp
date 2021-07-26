#ifndef FEASIBILITYRESTORATION_H
#define FEASIBILITYRESTORATION_H

#include "ConstraintRelaxationStrategy.hpp"
#include "GlobalizationStrategy.hpp"

enum Phase {FEASIBILITY_RESTORATION = 1, OPTIMALITY = 2};

class FeasibilityRestoration : public ConstraintRelaxationStrategy {
public:
   FeasibilityRestoration(const Problem& problem, const std::map<std::string, std::string>& options, bool use_trust_region);
   Iterate initialize(Statistics& statistics, const Problem& problem, std::vector<double>& x, Multipliers& multipliers) override;

   // direction computation
   void generate_subproblem(const Problem& problem, Iterate& current_iterate, double trust_region_radius) override;
   Direction compute_feasible_direction(Statistics& statistics, const Problem& problem, Iterate& current_iterate) override;
   Direction solve_feasibility_problem(Statistics& statistics, const Problem& problem, Iterate& current_iterate, Direction& direction) override;

   bool is_acceptable(Statistics& statistics, const Problem& problem, Iterate& current_iterate, Iterate& trial_iterate, const Direction& direction,
         double step_length) override;
   double compute_predicted_reduction(const Problem& problem, Iterate& current_iterate, const Direction& direction, double step_length) override;

private:
   const std::unique_ptr<GlobalizationStrategy> phase_1_strategy;
   const std::unique_ptr<GlobalizationStrategy> phase_2_strategy;
   Phase current_phase;

   void form_feasibility_problem(const Problem& problem, const Iterate& current_iterate, const std::vector<double>& phase_2_direction, const
   ConstraintPartition& constraint_partition);
   static void set_restoration_multipliers(std::vector<double>& constraints_multipliers, const ConstraintPartition& constraint_partition);
   void compute_infeasibility_measures(const Problem& problem, Iterate& iterate, const ConstraintPartition& constraint_partition);
};

#endif //FEASIBILITYRESTORATION_H