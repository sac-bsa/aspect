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
// This code is currently configured to take the average of the current cell
// and all of its neighbors and use that for all of the point's inside the cell
#include <aspect/particle/interpolator/wide_average.h>
#include <aspect/simulator_access.h>
#include <deal.II/grid/grid_tools.h>


namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      template<int dim>
      std::vector<std::vector<double> >
      WideAverage<dim>::properties_at_points(const ParticleHandler<dim> &particle_handler,
                                             const std::vector<Point<dim> > &positions,
                                             const ComponentMask &selected_properties,
                                             const CellType &cell) const

      {
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
        // We know where the center of our cell is with approximated_cell_midpoint
        const unsigned int n_particle_properties = particle_handler.n_properties_per_particle();

        AssertThrow(!positions.empty(),
                    ExcMessage("The particle property interpolator was not given any "
                               "positions to evaluate the particle cell_properties at."));


        CellType found_cell;

        if (cell == CellType())
          {
            // We can not simply use one of the points as input for find_active_cell_around_point
            // because for vertices of mesh cells we might end up getting ghost_cells as return value
            // instead of the local active cell. So make sure we are well in the inside of a cell.


            const Point<dim> approximated_cell_midpoint =
              std::accumulate(positions.begin(), positions.end(), Point<dim>())
              / static_cast<double> (positions.size());
            found_cell =
              (GridTools::find_active_cell_around_point<>(this->get_mapping(),
                                                          this->get_triangulation(),
                                                          approximated_cell_midpoint)).first;
          }
        else
          {
            found_cell = cell;
          }

        unsigned int n_particles = 0;
        std::vector<double> cell_properties(n_particle_properties, numbers::signaling_nan<double>());

        // Get neighbor cells
        std::vector<CellType> neighbors;
        GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(found_cell, neighbors);

        switch (type_of_averaging)
          {
            case Arithmetic :
              for (unsigned int i = 0; i < n_particle_properties; ++i)
                if (selected_properties[i])
                  cell_properties[i] = 0.0;
              // Sum particles in actual cell,
              arithmetic_adding_backend(particle_handler, found_cell, selected_properties, n_particles,
                                        cell_properties);

              for (CellType neighbor : neighbors)
                arithmetic_adding_backend(particle_handler, neighbor, selected_properties, n_particles,
                                          cell_properties);
              // Divide by the number of cells counted
              for (unsigned int i = 0; i < n_particle_properties; ++i)
                if (selected_properties[i])
                  cell_properties[i] /= n_particles;
              break;

            case Geometric :
              for (unsigned int i = 0; i < n_particle_properties; ++i)
                if (selected_properties[i])
                  cell_properties[i] = 1.0;
              // Sum the inverses of particles in the actual cell
              geometric_multiplication_backend(particle_handler, found_cell,
                                               selected_properties, n_particles,
                                               cell_properties);

              for (CellType neighbor : neighbors)
                geometric_multiplication_backend(particle_handler, neighbor,
                                                 selected_properties, n_particles,
                                                 cell_properties);
              // Divide by the number of cells counted
              for (unsigned int i = 0; i < n_particle_properties; ++i)
                if (selected_properties[i])
                  cell_properties[i] = std::pow(cell_properties[i], 1 / n_particles);
              break;

            case Harmonic :
              for (unsigned int i = 0; i < n_particle_properties; ++i)
                if (selected_properties[i])
                  cell_properties[i] = 0.0;
              // Sum the inverses of particles in the actual cell
              harmonic_adding_backend(particle_handler, found_cell,
                                      selected_properties, n_particles,
                                      cell_properties);

              for (CellType neighbor : neighbors)
                harmonic_adding_backend(particle_handler, neighbor,
                                        selected_properties, n_particles,
                                        cell_properties);
              // Divide by the number of cells counted
              for (unsigned int i = 0; i < n_particle_properties; ++i)
                if (selected_properties[i])
                  cell_properties[i] = n_particles / cell_properties[i];
              break;

          } // switch


        return std::vector<std::vector<double>>(positions.size(), cell_properties);
      } // properties_at_points



      template<int dim>
      void
      WideAverage<dim>::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.declare_entry("Averaging Method",
                              "1", // Uses Arithmetic Averaging by default
                              Patterns::List(Patterns::Integer(1, 3)),
                              "This chooses the averaging type used. "
                              "1 = Arithmetic, 2 = Geometric, 3 = Harmonic");
          } // Particles
          prm.leave_subsection();
        } // Postprocess
        prm.leave_subsection();
      } // declare parameters

      template<int dim>
      void
      WideAverage<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            long int averaging_type = prm.get_integer("Averaging Method");
            AssertThrow(averaging_type > 0 && averaging_type < 4,
                        ExcMessage("The specified method for averaging was invalid,"
                                   " please use a number in the range[1, 3] for Arithmetic, Geometric, or Harmonic"));
            type_of_averaging = static_cast<AverageType>(averaging_type);


          } // Particles
          prm.leave_subsection();
        } // Postprocess
        prm.leave_subsection();
      } // parse parameters


      template<int dim>
      void WideAverage<dim>::arithmetic_adding_backend(const ParticleHandler<dim> &particle_handler,
                                                       const CellType &cell,
                                                       const ComponentMask &selected_properties,
                                                       unsigned int &n_particles,
                                                       std::vector<double> &cell_properties)
      {
        const typename ParticleHandler<dim>::particle_iterator_range particle_range =
          particle_handler.particles_in_cell(cell);
        for (typename ParticleHandler<dim>::particle_iterator particle = particle_range.begin();
             particle != particle_range.end(); ++particle, ++n_particles)
          {
            const ArrayView<const double> &particle_properties = particle->get_properties();
            for (unsigned long i = 0; i < particle_properties.size(); ++i)
              if (selected_properties[i])
                cell_properties[i] += particle_properties[i];
          } // for loop
      } // arithmetic adding backend

      template<int dim>
      void WideAverage<dim>::harmonic_adding_backend(const ParticleHandler<dim> &particle_handler,
                                                     const CellType &cell,
                                                     const ComponentMask &selected_properties,
                                                     unsigned int &n_particles,
                                                     std::vector<double> &cell_properties)
      {
        const typename ParticleHandler<dim>::particle_iterator_range particle_range =
          particle_handler.particles_in_cell(cell);

        for (typename ParticleHandler<dim>::particle_iterator particle = particle_range.begin();
             particle != particle_range.end(); ++particle, ++n_particles)
          {
            const ArrayView<const double> &particle_properties = particle->get_properties();
            for (unsigned long i = 0; i < particle_properties.size(); ++i)
              if (selected_properties[i])
                cell_properties[i] += 1 / particle_properties[i];
          } // for loop

      } // Harmonic adding backend

      template<int dim>
      void WideAverage<dim>::geometric_multiplication_backend(const ParticleHandler<dim> &particle_handler,
                                                              const CellType &cell,
                                                              const ComponentMask &selected_properties,
                                                              unsigned int &n_particles,
                                                              std::vector<double> &cell_properties)
      {
        const typename ParticleHandler<dim>::particle_iterator_range particle_range =
          particle_handler.particles_in_cell(cell);

        for (typename ParticleHandler<dim>::particle_iterator particle = particle_range.begin();
             particle != particle_range.end(); ++particle, ++n_particles)
          {

            const ArrayView<const double> &particle_properties = particle->get_properties();
            for (unsigned long i = 0; i < particle_properties.size(); ++i)
              if (selected_properties[i])
                cell_properties[i] *= particle_properties[i];
          } // for loop
      } // geometric_multiplication_backend


    } // namespace interpolator
  } // namespace particle
} // namespace aspect


// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {

      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(WideAverage,
                                            "wide average",
                                            "Interpolates particle properties using a "
                                            "chosen average of points")

    }
  }
}
