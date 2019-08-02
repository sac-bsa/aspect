/*
  Copyright (C) 2017 - 2018 by the authors of the ASPECT code.

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

#include <aspect/particle/interpolator/peskin_weighting.h>
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
      PeskinWeighting<dim>::properties_at_points(const ParticleHandler<dim> &particle_handler,
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
        } // if
        else
          found_cell = cell;


        std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
                relevant_cells = get_lots_of_neighbors(found_cell);


        // Initialize cell_properties to have nan's
        std::vector<std::vector<double>> cell_properties(positions.size(), std::vector<double>(n_particle_properties,
                                                                                               std::numeric_limits<double>::signaling_NaN()));
        // fix it so that selected properties have 0s instead of nans
          for (std::size_t particle_property = 0; particle_property < n_particle_properties; ++particle_property)
            if (selected_properties[particle_property])
              for (std::size_t support_point = 0; support_point < positions.size(); ++support_point)
                cell_properties[support_point][particle_property] = 0;


        // Outermost for loop should go by support point
        // Next loop should go through the cells
        // Next loop should go through the particles
        // Next loop should go through each properties

        for (std::size_t support_point = 0; support_point < cell_properties.size(); ++support_point) {
          double accumulated_dirac_delta_h = 0;
          for (const auto &current_cell : relevant_cells) {
            const typename ParticleHandler<dim>::particle_iterator_range particle_range =
                    particle_handler.particles_in_cell(current_cell);
            for (auto particle = particle_range.begin(); particle != particle_range.end(); ++particle) {
              const double coefficient = dirac_delta_h(positions[support_point] - particle->get_location(),
                                                       found_cell->diameter());
              accumulated_dirac_delta_h += coefficient;
              const auto particle_properties = particle->get_properties();
              for (std::size_t property_index = 0; property_index < n_particle_properties; ++property_index)
                if (selected_properties[property_index])
                  cell_properties[support_point][property_index] += coefficient * particle_properties[property_index];
            } // for particle
          } // for cell
          AssertThrow(accumulated_dirac_delta_h != 0,
                      ExcMessage(
                              "Peskin Weighting was unable to find any particles within peskin scaling cell diameters of a support point"));
          for (std::size_t property_index = 0; property_index < n_particle_properties; ++property_index)
            if (selected_properties[property_index])
              cell_properties[support_point][property_index] /= accumulated_dirac_delta_h;
        }
        return cell_properties;
      } // properties_at_points




      template<int dim>
      void
      PeskinWeighting<dim>::declare_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.declare_entry ("Allow cells without particles", "false",
                               Patterns::Bool (),
                               "By default, every cell needs to contain particles to use this interpolator "
                               "plugin. If this parameter is set to true, cells are allowed to have no particles, "
                               "in which case the interpolator will return 0 for the cell's properties.");
            prm.declare_entry ("peskin scaling", "2", Patterns::Double(0),
                               "The scaling changes the range that particles "
                               "are included in. It defaults to 2, meaning that"
                               " particles up to two cell diameters from a "
                               "support point are used. It cannot be 0 or less than 0.");
          } // Particles
          prm.leave_subsection();
        } // Postprocess
        prm.leave_subsection();
      } // declare parameters

      template<int dim>
      void
      PeskinWeighting<dim>::parse_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            allow_cells_without_particles = prm.get_bool("Allow cells without particles");
            peskin_scaling = prm.get_double("peskin scaling");
            AssertThrow(peskin_scaling > 0, ExcMessage("peskin scaling must be greater than 0"));
          } // Particles
          prm.leave_subsection();
        } // Postprocess
        prm.leave_subsection();
      } // parse parameters

      template<int dim>
      double PeskinWeighting<dim>::phi(double r)  const {
        // This function implements equation 6.27 from C.S. Peskin's 2002
        // Immersed Boundary method, published in Volume 11 of Acta Numerica
        r *= 2 / peskin_scaling;
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
      double PeskinWeighting<dim>::dirac_delta_h(const Tensor<1, dim, double>& x, double cell_diameter) const {
        double value = 1 / std::pow(cell_diameter, dim);
        for (int coordinate = 0; coordinate < dim; ++coordinate)
          value *= phi(x[coordinate]/cell_diameter);
        return value;
      } // dirac_delta

      template<int dim>
      std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
              PeskinWeighting<dim>::get_lots_of_neighbors(
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
      void PeskinWeighting<dim>::no_duplicate_insert(std::vector<T>& vector, const T& item)
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

      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(PeskinWeighting,
                                            "peskin weighting",
                                            "Interpolates particle properties using a weighting formula from a paper by Peskin")

    }
  }
}
