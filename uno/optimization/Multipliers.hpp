// Copyright (c) 2018-2023 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#ifndef UNO_MULTIPLIERS_H
#define UNO_MULTIPLIERS_H

#include "linear_algebra/Vector.hpp"

struct Multipliers {
   std::vector<double> lower_bounds{}; /*!< Multipliers of the lower bound constraints */
   std::vector<double> upper_bounds{}; /*!< Multipliers of the lower bound constraints */
   std::vector<double> constraints{}; /*!< Multipliers of the general constraints */
   double objective{1.};

   Multipliers(size_t number_variables, size_t number_constraints);
   [[nodiscard]] double norm_1() const;
};

inline Multipliers::Multipliers(size_t number_variables, size_t number_constraints) : lower_bounds(number_variables),
      upper_bounds(number_variables), constraints(number_constraints) {
}

inline double Multipliers::norm_1() const {
   return ::norm_1(this->constraints) + ::norm_1(this->lower_bounds) + ::norm_1(this->upper_bounds);
}

#endif // UNO_MULTIPLIERS_H
