// Copyright (c) 2018-2023 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#ifndef UNO_TRUSTREGIONSTRATEGY_H
#define UNO_TRUSTREGIONSTRATEGY_H

#include "GlobalizationMechanism.hpp"

/*! \class TrustRegionStrategy
 * \brief Trust region strategy
 *
 *  Trust region strategy
 */
class TrustRegionStrategy : public GlobalizationMechanism {
public:
   TrustRegionStrategy(ConstraintRelaxationStrategy& constraint_relaxation_strategy, const Options& options);

   void initialize(Statistics& statistics, Iterate& first_iterate) override;
   std::tuple<Iterate, double> compute_acceptable_iterate(Statistics& statistics, const Model& model, Iterate& current_iterate) override;

private:
   double radius; /*!< Current trust region radius */
   const double increase_factor;
   const double decrease_factor;
   const double activity_tolerance;
   const double min_radius;
   const double radius_reset_threshold;
   const bool use_second_order_correction;
   // statistics table
   int statistics_SOC_column_order;
   int statistics_TR_radius_column_order;

   void increase_radius(double step_norm);
   void decrease_radius(double step_norm);
   void decrease_radius();
   void reset_active_trust_region_multipliers(const Model& model, const Direction& direction, Iterate& trial_iterate) const;
   void set_statistics(Statistics& statistics, const Direction& direction);
   [[nodiscard]] bool termination() const;
   void print_iteration();
};

#endif // UNO_TRUSTREGIONSTRATEGY_H
