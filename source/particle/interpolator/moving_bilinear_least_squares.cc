/*
  Copyright (C) 2019 by the authors of the ASPECT code.
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

#include <aspect/particle/interpolator/moving_bilinear_least_squares.h>
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
      MovingBilinearLeastSquares<dim>::properties_at_points(const ParticleHandler<dim> &particle_handler,
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
        }
        else
          found_cell = cell;


        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
                relevant_cells(get_lots_of_neighbors(found_cell));
        /*if (neighbor_usage == second_neighbors)
        {
          relevant_cells = get_lots_of_neighbors(found_cell);
        } else if (neighbor_usage == first_neighbors)
        {
          GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(found_cell, relevant_cells);
          relevant_cells.emplace_back(found_cell);
        } else if (neighbor_usage == only_current_cell)
        {
          relevant_cells.emplace_back(found_cell);
        }*/

        unsigned int n_particles = 0;
        for (const auto& current_cell : relevant_cells)
        {
          const typename ParticleHandler<dim>::particle_iterator_range particle_range =
                  particle_handler.particles_in_cell(current_cell);
          n_particles += std::distance(particle_range.begin(), particle_range.end());
        }
        AssertThrow(n_particles != 0, ExcMessage("No particles were found in any of the cells"));

        std::vector<std::vector<double>> cell_properties(positions.size(), std::vector<double>(n_particle_properties,
                std::numeric_limits<double>::signaling_NaN()));

        for (std::size_t property_index = 0; property_index < n_particle_properties; ++property_index)
        {
          if (!selected_properties[property_index])
            continue;
          std::size_t support_point_index = 0;
          for (auto support_point_location = positions.begin(); support_point_location != positions.end(); ++support_point_location, ++support_point_index)
          {
            cell_properties[support_point_index][property_index] = moving_bilinear_least_squares(property_index, particle_handler, (*support_point_location), n_particles, relevant_cells);
          }
        }

        return cell_properties;
      }

      template<>
      double
      MovingBilinearLeastSquares<2>::moving_bilinear_least_squares(
              std::size_t property_index,
              const ParticleHandler<2> &particle_handler,
              const Point<2>& support_point,
              std::size_t n_particles,
              const std::vector<typename parallel::distributed::Triangulation<2>::active_cell_iterator>& relevant_cells) const
      {
        const double cell_diameter = relevant_cells.front()->diameter();

        // Ac = f
        // WAc = Wf
        // (A^T)WAc = (A^T)Wf
        // inv((A^T)WA)(A^T)WAc = inv((A^T)WA)(A^T)Wf
        // c = inv((A^T)WA)(A^T)Wf

        constexpr unsigned int matrix_dimension = 6; // 1, x, y // x^2, y^2, xy
        dealii::LAPACKFullMatrix<double> WA(n_particles, matrix_dimension);
        dealii::LAPACKFullMatrix<double> A(n_particles, matrix_dimension);
        dealii::Vector<double> Wb(n_particles);


        std::size_t particle_index = 0;

        for (const auto& current_cell : relevant_cells)
        {
          const typename ParticleHandler<2>::particle_iterator_range particle_range =
                  particle_handler.particles_in_cell(current_cell);
          for (typename ParticleHandler<2>::particle_iterator particle = particle_range.begin();
               particle != particle_range.end(); ++particle, ++particle_index)
          {
            const double particle_property_value = particle->get_properties()[property_index];
            const dealii::Tensor<1, 2, double> difference = (support_point - particle->get_location()) / cell_diameter;
            const double weighting =dirac_delta_h(difference, cell_diameter);
                    //dirac_delta_h(difference, cell_diameter); // phi(difference.norm());
            Wb[particle_index] = weighting * particle_property_value;

            A(particle_index, 0) = 1;
            A(particle_index, 1) = difference[0];
            A(particle_index, 2) = difference[1];

            A(particle_index, 3) = std::pow(difference[0], 2);
            A(particle_index, 4) = std::pow(difference[1], 2);
            A(particle_index, 5) = difference[0] * difference[1];

//            A(particle_index, 6) = std::pow(difference[0], 3);
//            A(particle_index, 7) = std::pow(difference[0], 2) * difference[1];
//            A(particle_index, 8) = difference[0] * std::pow(difference[1], 2);
//            A(particle_index, 9) = std::pow(difference[1], 3);
//
//            A(particle_index, 10) = std::pow(difference[0], 4);
//            A(particle_index, 11) = std::pow(difference[0], 3) * difference[1];
//            A(particle_index, 12) = std::pow(difference[0] * difference[1], 2);
//            A(particle_index, 13) = difference[0] * std::pow(difference[1], 3);
//            A(particle_index, 14) = std::pow(difference[1], 4);


            WA(particle_index, 0) = weighting; // A(particle_index, 0) is always 1
            WA(particle_index, 1) = weighting * A(particle_index, 1);
            WA(particle_index, 2) = weighting * A(particle_index, 2);

            WA(particle_index, 3) = weighting * A(particle_index, 3);
            WA(particle_index, 4) = weighting * A(particle_index, 4);
            WA(particle_index, 5) = weighting * A(particle_index, 5);
//
//            WA(particle_index, 6) = weighting * std::pow(difference[0], 3);
//            WA(particle_index, 7) = weighting * std::pow(difference[0], 2) * difference[1];
//            WA(particle_index, 8) = weighting * difference[0] * std::pow(difference[1], 2);
//            WA(particle_index, 9) = weighting * std::pow(difference[1], 3);
//            WA(particle_index, 10) = weighting * std::pow(difference[0], 4);
//            WA(particle_index, 11) = weighting * std::pow(difference[0], 3) * difference[1];
//            WA(particle_index, 12) = weighting * std::pow(difference[0] * difference[1], 2);
//            WA(particle_index, 13) = weighting * difference[0] * std::pow(difference[1], 3);
//            WA(particle_index, 14) = weighting * std::pow(difference[1], 4);
          }
        }

        dealii::LAPACKFullMatrix<double>ATWA(matrix_dimension, matrix_dimension);
        A.Tmmult(ATWA, WA);
        constexpr double svd_threshold = 1E-15;
        dealii::LAPACKFullMatrix<double>ATWA_inverse(ATWA);
        ATWA_inverse.compute_inverse_svd(svd_threshold);
        dealii::Vector<double> ATWb(matrix_dimension);
        dealii::Vector<double> c(matrix_dimension);
        A.Tvmult(ATWb, Wb);
        ATWA_inverse.vmult(c, ATWb);

        return c[0];
      }
template<>
      double
      MovingBilinearLeastSquares<3>::moving_bilinear_least_squares(
              std::size_t property_index,
              const ParticleHandler<3> &particle_handler,
              const Point<3>& support_point,
              std::size_t n_particles,
              const std::vector<typename parallel::distributed::Triangulation<3>::active_cell_iterator>& relevant_cells) const
      {
        const double cell_diameter = relevant_cells.front()->diameter();
        // Ac = f
        // WAc = Wf
        // (A^T)WAc = (A^T)Wf
        // inv((A^T)WA)(A^T)WAc = inv((A^T)WA)(A^T)Wf
        // c = inv((A^T)WA)(A^T)Wf

        constexpr unsigned int matrix_dimension = 10; // 1, x, y , z // x^2, y^2, z^2, xy, xz, yz
        dealii::LAPACKFullMatrix<double> A(n_particles, matrix_dimension);
        dealii::LAPACKFullMatrix<double> W(n_particles);
        dealii::Vector<double> Wf(n_particles);


        std::size_t particle_index = 0;

        for (const auto& current_cell : relevant_cells)
        {
          const typename ParticleHandler<3>::particle_iterator_range particle_range =
                  particle_handler.particles_in_cell(current_cell);
          for (typename ParticleHandler<3>::particle_iterator particle = particle_range.begin();
               particle != particle_range.end(); ++particle, ++particle_index)
          {
            const double particle_property_value = particle->get_properties()[property_index];
            const dealii::Tensor<1, 3, double> difference = (support_point - particle->get_location())/cell_diameter;
            const double weighting = dirac_delta_h(difference, cell_diameter);
            Wf[particle_index] = weighting * particle_property_value;

            A(particle_index, 0) = 1;
            A(particle_index, 1) = difference[0];
            A(particle_index, 2) = difference[1];
            A(particle_index, 3) = difference[2];

            A(particle_index, 4) = std::pow(difference[0], 2);
            A(particle_index, 5) = std::pow(difference[1], 2);
            A(particle_index, 6) = std::pow(difference[2], 2);
            A(particle_index, 7) = difference[0] * difference[1];
            A(particle_index, 8) = difference[0] * difference[2];
            A(particle_index, 9) = difference[1] * difference[2];


            for (std::size_t index = 0; index < n_particles; ++index)
              W(particle_index,index) = (particle_index == index)? weighting : 0;

          }
        }

        dealii::LAPACKFullMatrix<double> ATW(matrix_dimension, n_particles);
        dealii::LAPACKFullMatrix<double>ATWA(matrix_dimension, matrix_dimension);
        A.Tmmult(ATW, W);
        ATW.mmult(ATWA, A);
        constexpr double svd_threshold = 1E-15;
        dealii::LAPACKFullMatrix<double>ATWA_inverse(ATWA);
        ATWA_inverse.compute_inverse_svd(svd_threshold);
        dealii::Vector<double> ATWr(matrix_dimension);
        dealii::Vector<double> c(matrix_dimension);
        A.Tvmult(ATWr, Wf);
        ATWA_inverse.vmult(c, ATWr);

        return c[0];
      }

      template<int dim>
      void
      MovingBilinearLeastSquares<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.declare_entry("Moving least squares radius", "1",
                              Patterns::Double(1E-15, 2),
                              "Moving least squares uses a function with compact support to weight particles. This parameter chooses where the support becomes 0. This is measured in cell_radius");
           /* prm.declare_entry("Use neighboring cells for particle interpolation",
                    "2",
                    Patterns::Integer(0, 2),
                    "Moving Least squares can operate using no neighbors, "
                    "each cell's first face neighbors, or each cells second face neighbors. Valid values are 0, 1 or 2.");*/

          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }

      template<int dim>
      void
      MovingBilinearLeastSquares<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            phi_scaling = prm.get_double("Moving least squares radius");
            //neighbor_usage = static_cast<NeighboringCellChoice>(prm.get_integer("Use neighboring cells for particle interpolation"));
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }

      template<int dim>
      double MovingBilinearLeastSquares<dim>::phi(double r) const
      {
        // This function implements equation 6.27 from C.S. Peskin's 2002
        // Immersed Boundary method, published in Volume 11 of Acta Numerica
        r *= 2 / phi_scaling;
        if (r >= 2 || r <= -2)
          return 0;
        else if (r <= -1)
          return (5 + 2 * r - std::sqrt(-7 - 12 * r - 4 * std::pow(r, 2))) / 4;
        else if (r <= 0)
          return (3 + 2 * r + std::sqrt(1 - 4 * r - 4 * std::pow(r, 2))) / 4;
        else if (r <= 1)
          return (3 - 2 * r + std::sqrt(1 + 4 * r - 4 * std::pow(r, 2))) / 4;
        else // (r <= 2)
          return (5 - 2 * r - std::sqrt(- 7 + 12 * r - 4 * std::pow(r, 2))) / 4;
      }

      template<int dim>
      double MovingBilinearLeastSquares<dim>::dirac_delta_h(const Tensor<1, dim, double>& x, double cell_diameter) const
      {
        double value = 1 /std::pow(cell_diameter, dim);
        for (int coordinate = 0; coordinate < dim; ++coordinate)
          value *= phi(x[coordinate]);
        return value;
      }


      template<int dim>
      std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
      MovingBilinearLeastSquares<dim>::get_lots_of_neighbors(
              typename parallel::distributed::Triangulation<dim>::active_cell_iterator origin_cell)
      {
        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> all_cells;
        all_cells.emplace_back(origin_cell);
        // We only want to have the cells, its neighbors, and its second neighbors
        // Its second neighbors can contain its first neighbors, as well as duplicates of it's second neighbors so we must check that there is no duplication

        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> original_neighbors;

        GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(origin_cell, original_neighbors);
        for (const typename parallel::distributed::Triangulation<dim>::active_cell_iterator & current_cell : original_neighbors) {
          no_duplicate_insert(all_cells, current_cell);
          std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> neighbors;
          GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(current_cell, neighbors);
          for (const typename parallel::distributed::Triangulation<dim>::active_cell_iterator& neighboring_cell : neighbors)
            no_duplicate_insert(all_cells, neighboring_cell);
        }
        return all_cells;
      }

      template<int dim>
      template<typename T>
      void MovingBilinearLeastSquares<dim>::no_duplicate_insert(std::vector<T>& vector, const T& item)
      {
        for (const auto& vector_item : vector)
          if (vector_item == item)
            return;
        vector.emplace_back(item);
      }
    }
  }
}


// explicit instantiations
namespace aspect {
  namespace Particle {
    namespace Interpolator {

      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(MovingBilinearLeastSquares,
      "moving bilinear least squares",
      "Interpolates particle properties using "
      "moving least squares regression")

    }
  }
}


