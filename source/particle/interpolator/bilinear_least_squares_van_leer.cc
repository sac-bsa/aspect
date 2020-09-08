
/*
  Copyright (C) 2020 by the authors of the ASPECT code.

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

#include <aspect/particle/interpolator/bilinear_least_squares_van_leer.h>
#include <aspect/postprocess/particles.h>
#include <aspect/simulator.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/lac/full_matrix.templates.h>

#include <boost/lexical_cast.hpp>

namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      template <int dim>
      std::vector<std::vector<double> >
      BilinearLeastSquaresVanLeer<dim>::properties_at_points(const ParticleHandler<dim> &particle_handler,
                                                             const std::vector<Point<dim> > &positions,
                                                             const ComponentMask &selected_properties,
                                                             const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const
      {
        const unsigned int n_particle_properties = particle_handler.n_properties_per_particle();

        const unsigned int property_index = selected_properties.first_selected_component(selected_properties.size());

        AssertThrow(property_index != numbers::invalid_unsigned_int,
                    ExcMessage("Internal error: the particle property interpolator was "
                               "called without a specified component to interpolate."));

        const Point<dim> approximated_cell_midpoint = std::accumulate (positions.begin(), positions.end(), Point<dim>())
                                                      / static_cast<double> (positions.size());

        typename parallel::distributed::Triangulation<dim>::active_cell_iterator found_cell;

        if (cell == typename parallel::distributed::Triangulation<dim>::active_cell_iterator())
          {
            // We can not simply use one of the points as input for find_active_cell_around_point
            // because for vertices of mesh cells we might end up getting ghost_cells as return value
            // instead of the local active cell. So make sure we are well in the inside of a cell.
            Assert(positions.size() > 0,
                   ExcMessage("The particle property interpolator was not given any "
                              "positions to evaluate the particle cell_properties at."));


            found_cell =
              (GridTools::find_active_cell_around_point<> (this->get_mapping(),
                                                           this->get_triangulation(),
                                                           approximated_cell_midpoint)).first;
          }
        else
          found_cell = cell;

        const typename ParticleHandler<dim>::particle_iterator_range particle_range =
          particle_handler.particles_in_cell(found_cell);


        std::vector<std::vector<double> > cell_properties(positions.size(),
                                                          std::vector<double>(n_particle_properties,
                                                                              numbers::signaling_nan<double>()));

        const unsigned int n_particles = std::distance(particle_range.begin(), particle_range.end());

        AssertThrow(n_particles != 0,
                    ExcMessage("At least one cell contained no particles. The 'bilinear'"
                               "interpolation scheme does not support this case. "));


        // Noticed that the size of matrix A is n_particles x matrix_dimension
        // which usually is not a square matrix. Therefore, we find the
        // least squares solution of Ax=r by solving the "normal" equations
        // (A^TA) x = A^Tr.
        const unsigned int matrix_dimension = (dim == 2) ? 3 : 4;
        dealii::LAPACKFullMatrix<double> A(n_particles, matrix_dimension);
        std::vector<Vector<double>> r(n_particle_properties, Vector<double>(n_particles));
        for (unsigned int property_index = 0; property_index < n_particle_properties; ++property_index)
          if (selected_properties[property_index])
            r[property_index] = 0;

        unsigned int positions_index = 0;
        const double cell_diameter = found_cell->diameter();
        for (typename ParticleHandler<dim>::particle_iterator particle = particle_range.begin();
             particle != particle_range.end(); ++particle, ++positions_index)
          {
            const auto particle_property_value = particle->get_properties();
            for (unsigned int property_index = 0; property_index < n_particle_properties; ++property_index)
              if (selected_properties[property_index])
                r[property_index][positions_index] = particle_property_value[property_index];

            const Tensor<1, dim, double> relative_particle_position = (particle->get_location() - approximated_cell_midpoint) / cell_diameter;
            A(positions_index, 0) = 1;
            A(positions_index, 1) = relative_particle_position[0];
            A(positions_index, 2) = relative_particle_position[1];
            if (dim == 3)
              {
                A(positions_index, 3) = relative_particle_position[2];
                AssertThrow(false, ExcNotImplemented("The van leer limiter currently is only being tested on 2D models"));
              }
          }

        dealii::LAPACKFullMatrix<double> B(matrix_dimension, matrix_dimension);

        std::vector<Vector<double>> c_ATr(n_particle_properties, Vector<double>(matrix_dimension));
        std::vector<Vector<double>> c(n_particle_properties, Vector<double>(matrix_dimension));

        constexpr double threshold = 1e-15;
        unsigned int index_positions = 0;

        // Form the matrix B=A^TA and right hand side A^Tr of the normal equation.
        A.Tmmult(B, A, false);
        dealii::LAPACKFullMatrix<double> B_inverse(B);
        B_inverse.compute_inverse_svd(threshold);

        for (unsigned int property_index = 0; property_index < n_particle_properties; ++property_index)
          {
            if (selected_properties[property_index])
              {
                A.Tvmult(c_ATr[property_index],r[property_index]);
                B_inverse.vmult(c[property_index], c_ATr[property_index]);
                //TODO actually limit using Van Leer
              }
          }

        for (typename std::vector<Point<dim>>::const_iterator itr = positions.begin(); itr != positions.end(); ++itr, ++index_positions)
          {
            const Tensor<1, dim, double> relative_support_point_location = (*itr - approximated_cell_midpoint) / cell_diameter;
            for (unsigned int property_index = 0; property_index < n_particle_properties; ++property_index)
              {
                double interpolated_value = c[property_index][0] +
                                            c[property_index][1] * relative_support_point_location[0] +
                                            c[property_index][2] * relative_support_point_location[1];
                if (dim == 3)
                  {
                    interpolated_value += c[property_index][3] * relative_support_point_location[2];
                  }

                cell_properties[index_positions][property_index] = interpolated_value;
              }
          }
        return cell_properties;
      }



      template <int dim>
      void
      BilinearLeastSquaresVanLeer<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        prm.leave_subsection();
      }



      template <int dim>
      void
      BilinearLeastSquaresVanLeer<dim>::parse_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        prm.leave_subsection();
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(BilinearLeastSquaresVanLeer,
                                            "bilinear least squares van leer limiter",
                                            "Interpolates particle properties onto a vector of points using a "
                                            "bilinear least squares method. "
                                            "Note that deal.II must be configured with BLAS and LAPACK to "
                                            "support this operation.")
    }
  }
}
