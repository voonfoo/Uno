// Copyright (c) 2018-2023 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#ifndef UNO_LEYFFERFILTERSTRATEGY_H
#define UNO_LEYFFERFILTERSTRATEGY_H

#include "FilterStrategy.hpp"

class LeyfferFilterStrategy : public FilterStrategy {
public:
   explicit LeyfferFilterStrategy(const Options& options);

   [[nodiscard]] bool is_iterate_acceptable(const ProgressMeasures& current_progress_measures, const ProgressMeasures& trial_progress_measures,
         const PredictedReduction& predicted_reduction, double objective_multiplier) override;
};

#endif // UNO_LEYFFERFILTERSTRATEGY_H