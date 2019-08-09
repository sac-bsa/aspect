/*
  Copyright (C) 2016 - 2017 by the authors of the ASPECT code.

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

#include <aspect/particle/interpolator/moving_least_sqaures.h>
#include <aspect/simulator_access.h>
#include <deal.II/grid/grid_tools.h>
#include <cmath>
#include <limits>


namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      template<int dim>
      std::vector<std::vector<double> >
      MovingLeastSquares<dim>::properties_at_points(const ParticleHandler<dim> &particle_handler,
                                                 const std::vector<Point<dim> > &positions,
                                                 const ComponentMask &selected_properties,
                                                 const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const {
        // What I know:
        // particle_handler
        //   Knows what properties are carried "particle_handler.n_properties_per_particle()" --Bilinear
        // positions
        //   We are given several points to generate values for through positions, we return the values through cell_properties
        // selected_properties
        //   We are given a specific property to interpolate "property_index"
        // cell
        //   We can iterate over the particles of a cell with "particle_handler.particles_in_cell(cell)"
        //   Neighbors can be retrieved through
        //   "std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> neighbors;
        //   GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(found_cell,neighbors);" --Cell Average
        // we know the dimension
        // it is possible to do one property at a time or multiple
        // We know where the center of our cell is with approximated_cell_midpoint which is the average of the locations of support points
        const unsigned int n_particle_properties = particle_handler.n_properties_per_particle();

        AssertThrow(n_particle_properties == selected_properties.size(),
                    ExcMessage("Internal Error - the component mask did not have "
                               "the same length as the particle handler properties"));

        AssertThrow(!positions.empty(),
                    ExcMessage("The particle property interpolator was not given any "
                               "positions to evaluate the particle cell_properties at."));
        AssertThrow(dim == 2, ExcNotImplemented("Three dimensions has not been implemented yet"));


        typename parallel::distributed::Triangulation<dim>::active_cell_iterator found_cell;
        if (cell == typename parallel::distributed::Triangulation<dim>::active_cell_iterator()) {
          // We can not simply use one of the points as input for find_active_cell_around_point
          // because for vertices of mesh cells we might end up getting ghost_cells as return value
          // instead of the local active cell. So make sure we are well in the inside of a cell.
          const Point<dim> approximated_cell_midpoint =
                  std::accumulate(positions.begin(), positions.end(), Point<dim>())
                  / static_cast<double> (positions.size());
          found_cell = (GridTools::find_active_cell_around_point<>(this->get_mapping(),
                                                                   this->get_triangulation(),
                                                                   approximated_cell_midpoint)).first;
        } // if
        else
          found_cell = cell;

        double cell_diameter = found_cell->diameter();

        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
                relevant_cells = get_lots_of_neighbors(found_cell);
        unsigned int n_particles = 0;
        for (const auto& current_cell : relevant_cells)
        {
          const typename ParticleHandler<dim>::particle_iterator_range particle_range =
                  particle_handler.particles_in_cell(current_cell);
          n_particles += std::distance(particle_range.begin(), particle_range.end());
        }
        AssertThrow(n_particles != 0, ExcMessage("No particles were found in any of the cells"));

        // Initialize cell_properties to have nan's
        std::vector<std::vector<double>> cell_properties(positions.size(), std::vector<double>(n_particle_properties,

                                                                                               std::numeric_limits<double>::signaling_NaN()));
        for (std::size_t property_index = 0; property_index < n_particle_properties; ++property_index)
        {
          if (!selected_properties[property_index])
            continue;
          std::size_t support_point_index = 0;
          for (auto support_point = positions.begin(); support_point != positions.end(); ++support_point, ++support_point_index)
          {
            // Ac = r
            // WAc = Wr
            // (A^T)WAc = (A^T)Wr
            // inv((A^T)WA)(A^T)WAc = inv((A^T)WA)(A^T)Wr
            // c = inv((A^T)WA)(A^T)Wr

            constexpr unsigned int matrix_dimension = 6 ; // 1, x, y, x^2, y^2, xy
            dealii::LAPACKFullMatrix<double> A(n_particles, matrix_dimension);
            dealii::LAPACKFullMatrix<double> W_matrix(n_particles);
            dealii::Vector<double> Wr(n_particles);


            std::size_t particle_index = 0;

            for (const auto& current_cell : relevant_cells)
            {
              const typename ParticleHandler<dim>::particle_iterator_range particle_range =
                      particle_handler.particles_in_cell(current_cell);
              for (typename ParticleHandler<dim>::particle_iterator particle = particle_range.begin();
                   particle != particle_range.end(); ++particle, ++particle_index)
              {
                AssertThrow(particle_index < n_particles, ExcMessage("We should not have this many particles"));
                const double particle_property_value = particle->get_properties()[property_index];
                const dealii::Tensor<1, dim, double> difference =  (*support_point) - particle->get_location();
                const double weighting = dirac_delta_h(difference, cell_diameter);
                Wr[particle_index] = weighting * particle_property_value;

                A(particle_index, 0) = 1;
                A(particle_index, 1) = difference[0]/cell_diameter;
                A(particle_index, 2) = difference[1]/cell_diameter;
                A(particle_index, 3) = std::pow(difference[0]/cell_diameter, 2);
                A(particle_index, 4) = std::pow(difference[1]/cell_diameter, 2);
                A(particle_index, 5) = difference[0] * difference[1] / std::pow(cell_diameter, 2);

                for (std::size_t index = 0; index < n_particles; ++index)
                  W_matrix(particle_index,index) = (particle_index == index)? weighting : 0;
                // diagonal matrix didn't have a matrix multiplication method

              } // for particle
            } // for cell

            dealii::LAPACKFullMatrix<double> ATW(matrix_dimension, n_particles);
            dealii::LAPACKFullMatrix<double>ATWA(matrix_dimension, matrix_dimension);
            A.Tmmult(ATW, W_matrix);
            ATW.mmult(ATWA, A);
            dealii::LAPACKFullMatrix<double>ATWA_inverse(ATWA);
            ATWA_inverse.compute_inverse_svd(1E-15);
            dealii::Vector<double> ATWr(matrix_dimension);
            dealii::Vector<double> c(matrix_dimension);
            A.Tvmult(ATWr, Wr);
            ATWA_inverse.vmult(c, ATWr);

            cell_properties[support_point_index][property_index] = c[0];

          } // for support point

        } // for property

        return cell_properties;
      } // properties_at_points




      template<int dim>
      void
      MovingLeastSquares<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.declare_entry("Allow cells without particles", "false",
                              Patterns::Bool(),
                              "By default, every cell needs to contain particles to use this interpolator "
                              "plugin. If this parameter is set to true, cells are allowed to have no particles, "
                              "in which case the interpolator will return 0 for the cell's properties.");
          }  // Particles
          prm.leave_subsection();
        } // Postprocess
        prm.leave_subsection();
      } // declare parameters

      template<int dim>
      void
      MovingLeastSquares<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            allow_cells_without_particles = prm.get_bool("Allow cells without particles");
          } // Particles
          prm.leave_subsection();
        } // Postprocess
        prm.leave_subsection();
      } // parse parameters

      template<int dim>
      double MovingLeastSquares<dim>::phi(double r)  const {
        // This function implements equation 6.27 from C.S. Peskin's 2002
        // Immersed Boundary method, published in Volume 11 of Acta Numerica
        r *= 2;
        if (r >= 2 || r <= -2)
          return 0;
        else if (r <= -1)
          return (5 + 2 * r - std::sqrt(-7 - 12 * r - 4 * std::pow(r, 2))) / 8;
        else if (r <= 0)
          return (3 + 2 * r + std::sqrt(1 - 4 * r - 4 * std::pow(r, 2))) / 8;
        else if (r <= 1)
          return (3 - 2 * r + std::sqrt(1 + 4 * r - 4 * std::pow(r, 2))) / 8;
        else // (r <= 2)
          return (5 - 2 * r - std::sqrt(- 7 + 12 * r - 4 * std::pow(r, 2))) / 8;
      } // phi

      template<int dim>
      double MovingLeastSquares<dim>::dirac_delta_h(const Tensor<1, dim, double>& x, double cell_diameter) const {
        double value = 1 / std::pow(cell_diameter, dim);
        for (int coordinate = 0; coordinate < dim; ++coordinate)
          value *= phi(x[coordinate]/cell_diameter);
        return value;
      } // dirac_delta


      template<int dim>
      std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
      MovingLeastSquares<dim>::get_lots_of_neighbors(
              typename parallel::distributed::Triangulation<dim>::active_cell_iterator origin_cell)
      {
        /*  O = Original
         *  N = Neighbor
         *  X = second_neighbor
         *  _ = not included
         * ___________
         * |_|_|X|_|_|
         * |_|X|N|X|_|
         * |X|N|O|N|X|
         * |_|X|N|X|_|
         * |_|_|X|_|_|
         *
         */
        AssertThrow(origin_cell != typename parallel::distributed::Triangulation<dim>::active_cell_iterator(),
                    ExcMessage("A function tried to get the neighbors of a ghost cell"));
        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> all_cells;
        all_cells.emplace_back(origin_cell);
        // We only want to have the cells, its neighbors, and its second neighbors
        // Its second neighbors can contain its first neighbors so we must ensure there is no duplication

        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> original_neighbors;

        GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(origin_cell, original_neighbors);
        // Get active neighbors clears the vector for us, so we can reuse one vector for all second neighbors
        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> neighbors;
        for (const auto& cell : original_neighbors) {
          no_duplicate_insert(all_cells, cell);
          GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(cell, neighbors);
          for (const auto& neighboring_cell : neighbors)
            no_duplicate_insert(all_cells, neighboring_cell);
        } // for
        return all_cells;
      } // get_lots_of_neighbors

      template<int dim>
      template<typename T>
      void MovingLeastSquares<dim>::no_duplicate_insert(std::vector<T>& vector, const T& item)
      {
        for (const auto& vector_item : vector)
          if (vector_item == item)
            return;
        vector.emplace_back(item);
      }




    } // namespace interpolator
  } // namespace particle
} // namespace aspect


// explicit instantiations
namespace aspect {
  namespace Particle {
    namespace Interpolator {

      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(MovingLeastSquares,
      "moving least squares",
      "Interpolates particle properties using the idea of moving least squares regression")

    }
  }
}


