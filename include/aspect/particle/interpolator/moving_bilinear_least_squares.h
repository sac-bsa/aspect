/*
 Copyright (C) 2017 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file LICENSE.  If not see
 <http://www.gnu.org/licenses/>.
 */


#ifndef _aspect_particle_interpolator_moving_least_squares_h
#define _aspect_particle_interpolator_moving_least_squares_h

#include <aspect/particle/interpolator/interface.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace Particle {
    namespace Interpolator {
      /**
       * Return the interpolated properties of all particles of the given cell using bilinear least squares method.
       * Currently, only the two dimensional model is supported.
       *
       * @ingroup ParticleInterpolators
       */
      template<int dim>
      class MovingBilinearLeastSquares : public Interface<dim>, public aspect::SimulatorAccess<dim> {
      public:
        /**
         * Return the cell-wise evaluated properties of the bilinear least squares function at the positions.
         */
        std::vector<std::vector<double> >
        properties_at_points(const ParticleHandler<dim> &particle_handler,
                             const std::vector<Point<dim> > &positions,
                             const ComponentMask &selected_properties,
                             const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const override;

        // avoid -Woverloaded-virtual:
        using Interface<dim>::properties_at_points;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters(ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters(ParameterHandler &prm);

      private:
        enum NeighboringCellChoice {
          only_current_cell = 0,
          first_neighbors = 1,
          second_neighbors = 2
        };
        NeighboringCellChoice neighbor_usage = second_neighbors;
        double r = 1;
        double phi(double difference)  const;
        double dirac_delta_h(const Tensor<1, dim, double>& difference, double cell_diameter) const;

        static std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> get_lots_of_neighbors(
                typename parallel::distributed::Triangulation<dim>::active_cell_iterator origin_cell);

        template<typename T>
        static void no_duplicate_insert(std::vector<T>& vector, const T& item);

        double moving_bilinear_least_squares(std::size_t property_index, const ParticleHandler<dim> &particle_handler, const Point<dim>& support_point, std::size_t n_particles,
                const std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>& relevant_cells) const;
      }; // class MovingLeastSquares
    } // namespace Interpolator

  } // namespace Particle
} // namespace aspect
#endif //_aspect_particle_interpolator_moving_least_squares_h
