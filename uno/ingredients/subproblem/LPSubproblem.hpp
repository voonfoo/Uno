#ifndef UNO_LPSUBPROBLEM_H
#define UNO_LPSUBPROBLEM_H

#include "ActiveSetSubproblem.hpp"
#include "solvers/QP/LPSolver.hpp"
#include "tools/Options.hpp"

class LPSubproblem : public ActiveSetSubproblem {
public:
   LPSubproblem(const ReformulatedProblem& problem, const Options& options);

   [[nodiscard]] Direction solve(Statistics& statistics, const ReformulatedProblem& problem, Iterate& current_iterate) override;
   [[nodiscard]] PredictedReductionModel generate_predicted_reduction_model(const ReformulatedProblem& problem, const Direction& direction) const override;
   [[nodiscard]] size_t get_hessian_evaluation_count() const override;
   [[nodiscard]] double get_proximal_coefficient() const override;

private:
   // use pointers to allow polymorphism
   const std::unique_ptr<LPSolver> solver; /*!< Solver that solves the subproblem */

   void evaluate_problem(const ReformulatedProblem& problem, Iterate& current_iterate);
};

#endif // UNO_LPSUBPROBLEM_H
