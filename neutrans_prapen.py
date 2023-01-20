import os
import numpy as np
import openmc
import openmoc
import openmc.mgxs as mgxs
from openmc import openmoc_compatible
from matplotlib import pyplot as plt

def calculation(pitch_to_diameter, fuel_pin_radius, fp_grid, enrichment, intrinsic):
    # Delete Summary and Statepoint Files
    try:
        os.remove("summary.h5")
        os.remove("statepoint.50.h5")
    except OSError:
        pass
    
    pitch_size = pitch_to_diameter * fuel_pin_radius * 2
    lower_left_coordinate = -pitch_size * fp_grid / 2.0
    
    ###############################################################################
    # MATERIALS
    ###############################################################################
    
    # Instantiate Nuclides
    u235 = openmc.Nuclide('U235')
    u238 = openmc.Nuclide('U238')
    si28 = openmc.Nuclide('Si28')
    fe56 = openmc.Nuclide('Fe56')  
    cr52 = openmc.Nuclide('Cr52')
    al27 = openmc.Nuclide('Al27')
    h1 = openmc.Nuclide('H1')
    o16 = openmc.Nuclide('O16')
    
    # Instantiate Materials and register the Nuclides
    #- Fuel (U3Si2)
    fuel = openmc.Material(name='U3Si2')
    fuel.set_density('g/cm3', 11.468)
    fuel.add_nuclide(u235, enrichment * 92.7, percent_type='wo')
    fuel.add_nuclide(u238, (1 - enrichment) * 92.7, percent_type='wo')
    fuel.add_nuclide(si28, 7.3, percent_type='wo')
    
    #- Moderator (Water)
    moderator = openmc.Material(name='Water')
    moderator.set_density('g/cm3', 0.72555)
    moderator.add_nuclide(h1, 11.11, percent_type='wo')
    moderator.add_nuclide(o16, 88.89, percent_type='wo')
    
    #- Cladding (FeCrAl)
    cladding = openmc.Material(name='FeCrAl')
    cladding.set_density('g/cm3', 7.25)
    cladding.add_nuclide(fe56, 74., percent_type='wo')
    cladding.add_nuclide(cr52, 21., percent_type='wo')
    cladding.add_nuclide(al27, 5., percent_type='wo')
    
    # Instantiate a Materials object
    materials_file = openmc.Materials([fuel, moderator, cladding])
    materials_file.cross_sections = "/home/nazvix/.config/spyder-py3/tugas_akhir/cross_sections/cross_sections.xml"
    
    # Export to "materials.xml"
    materials_file.export_to_xml()
    
    ###############################################################################
    # GEOMETRY
    ###############################################################################
    
    # Create cylinders for the fuel and clad
    fuel_outer_radius = openmc.ZCylinder(x0=0.0, y0=0.0, r=fuel_pin_radius)
    clad_outer_radius = openmc.ZCylinder(x0=0.0, y0=0.0, r=fuel_pin_radius+0.06)
    
    # Create boundary planes to surround the geometry
    min_x = openmc.XPlane(x0=lower_left_coordinate, boundary_type='reflective')
    max_x = openmc.XPlane(x0=-lower_left_coordinate, boundary_type='reflective')
    min_y = openmc.YPlane(y0=lower_left_coordinate, boundary_type='reflective')
    max_y = openmc.YPlane(y0=-lower_left_coordinate, boundary_type='reflective')
    min_z = openmc.ZPlane(z0=-100., boundary_type='vacuum')
    max_z = openmc.ZPlane(z0=+100., boundary_type='vacuum')
    
    # Create a Universe to encapsulate a fuel pin
    fuel_pin_universe = openmc.Universe(name='Fuel Pin')
    
    # Create a fuel Cell
    fuel_cell = openmc.Cell(name='Fuel')
    fuel_cell.fill = fuel
    fuel_cell.region = -fuel_outer_radius
    fuel_pin_universe.add_cell(fuel_cell)
    
    # Create a clad Cell
    clad_cell = openmc.Cell(name='Clad')
    clad_cell.fill = cladding
    clad_cell.region = +fuel_outer_radius & -clad_outer_radius
    fuel_pin_universe.add_cell(clad_cell)
    
    # Create a moderator Cell
    moderator_cell = openmc.Cell(name='Moderator')
    moderator_cell.fill = moderator
    moderator_cell.region = +clad_outer_radius
    fuel_pin_universe.add_cell(moderator_cell)
    
    # Create fuel assembly Lattice
    assembly = openmc.RectLattice(name='Fuel Assembly')
    assembly.pitch = (pitch_size, pitch_size)
    assembly.lower_left = [lower_left_coordinate] * 2
    
    # Initialize an empty 17x17 array of the lattice universes
    universes = np.empty((fp_grid, fp_grid), dtype=openmc.Universe)
    
    # Fill the array with the fuel pin and guide tube universes
    universes[:,:] = fuel_pin_universe
    
    # Store the array of universes in the lattice
    assembly.universes = universes
    
    # Create root Cell
    root_cell = openmc.Cell(name='root cell')
    root_cell.fill = assembly
    
    # Add boundary planes
    root_cell.region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z
    
    # Create root Universe
    root_universe = openmc.Universe(universe_id=0, name='root universe')
    root_universe.add_cell(root_cell)
    
    # Create Geometry and set root Universe
    geometry = openmc.Geometry(root_universe)
    # Export to "geometry.xml"
    geometry.export_to_xml()
    
    ###############################################################################
    # Simulation Parameters
    ###############################################################################
    # OpenMC simulation parameters
    batches = 50
    inactive = 10
    particles = 1000
    
    # Instantiate a Settings object
    settings_file = openmc.Settings()
    settings_file.batches = batches
    settings_file.inactive = inactive
    settings_file.particles = particles
    settings_file.output = {'tallies': False}
    
    # Create an initial uniform spatial source distribution over fissionable zones
    bounds = [lower_left_coordinate, lower_left_coordinate, -10, -lower_left_coordinate, -lower_left_coordinate, 10.]
    uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
    settings_file.source = openmc.Source(space=uniform_dist)
    
    # Export to "settings.xml"
    settings_file.export_to_xml()
    
    ###############################################################################
    # MGXS
    ###############################################################################
    
    # Instantiate a 2-group EnergyGroups object
    groups = mgxs.EnergyGroups()
    groups.group_edges = np.array([0., 0.625, 20.0e6])
    
    # Initialize a 2-group MGXS Library for OpenMOC
    mgxs_lib = openmc.mgxs.Library(geometry)
    mgxs_lib.energy_groups = groups
    
    # Specify multi-group cross section types to compute
    mgxs_lib.mgxs_types = ['nu-transport', 'nu-fission', 'fission', 'nu-scatter matrix', 'chi']
    
    # Specify a "cell" domain type for the cross section tally filters
    mgxs_lib.domain_type = 'cell'
    
    # Specify the cell domains over which to compute multi-group cross sections
    mgxs_lib.domains = geometry.get_all_material_cells().values()
    
    # Compute cross sections on a nuclide-by-nuclide basis
    mgxs_lib.by_nuclide = True
    
    # Construct all tallies needed for the multi-group cross section library
    mgxs_lib.build_library()
    
    # Create a "tallies.xml" file for the MGXS Library
    tallies_file = openmc.Tallies()
    mgxs_lib.add_to_tallies_file(tallies_file, merge=True)
    
    ###############################################################################
    # Tally
    ###############################################################################
    
    # Instantiate a tally Mesh
    global mesh
    mesh = openmc.RegularMesh(mesh_id=1)
    mesh.dimension = [fp_grid, fp_grid]
    mesh.lower_left = [lower_left_coordinate, lower_left_coordinate]
    mesh.upper_right = [-lower_left_coordinate, -lower_left_coordinate]
    
    # Instantiate tally Filter
    mesh_filter = openmc.MeshFilter(mesh)
    
    # Instantiate the Tally
    tally = openmc.Tally(name='mesh tally')
    tally.filters = [mesh_filter]
    tally.scores = ['fission', 'nu-fission']
    
    # Add tally to collection
    tallies_file.append(tally)
    
    # Export all tallies to a "tallies.xml" file
    tallies_file.export_to_xml()
    
    openmc.run()
    
    ###############################################################################
    # Storing MGXS Data
    ###############################################################################
    
    # Load the last statepoint file
    global sp
    with openmc.StatePoint('statepoint.50.h5') as sp: 
        sp = openmc.StatePoint('statepoint.50.h5')
        
        # Initialize MGXS Library with OpenMC statepoint data
        mgxs_lib.load_from_statepoint(sp)
        
        # Retrieve OpenMC's k-effective value
        openmc_keff = sp.k_combined
        
        # Retrieve the NuFissionXS object for the fuel cell from the library
        fuel_mgxs = mgxs_lib.get_mgxs(fuel_cell, 'nu-fission')
        
        fuel_mgxs.print_xs()
        
        # Store the cross section data in an "mgxs/mgxs.h5" HDF5 binary file
        mgxs_lib.build_hdf5_store(filename='atf-mgxs.h5', directory='mgxs')
    
    ###############################################################################
    # Verification with OpenMOC
    ###############################################################################

    # Create an OpenMOC Geometry from the OpenMC Geometry
    openmoc_geometry = openmc.openmoc_compatible.get_openmoc_geometry(mgxs_lib.geometry)
    # Load the library into the OpenMOC geometry
    global materials
    materials = openmoc.materialize.load_openmc_mgxs_lib(mgxs_lib, openmoc_geometry)
    
    # Generate tracks for OpenMOC
    track_generator = openmoc.TrackGenerator(openmoc_geometry, num_azim=intrinsic, azim_spacing=0.1)
    #track_generator = openmoc.TrackGenerator3D(openmoc_geometry, num_azim=intrinsic, azim_spacing=0.001, num_polar=32, z_spacing=0.001)
    track_generator.generateTracks()
    
    # Run OpenMOC
    global solver
    solver = openmoc.CPUSolver(track_generator)
    solver.computeEigenvalue()
    
    # Print report of keff and bias with OpenMC
    openmoc_keff = solver.getKeff()
    bias = (openmoc_keff - openmc_keff) * 1e5
    
    print('openmc keff = {0:1.6f}'.format(openmc_keff))
    print('openmoc keff = {0:1.6f}'.format(openmoc_keff))
    print('bias [pcm]: {0:1.1f}'.format(bias))

    #return openmc_keff
    return [openmc_keff, openmoc_keff, bias]

def visualization(fp_grid):
    ###############################################################################
    # Plot
    ###############################################################################
    
    # Get the OpenMC fission rate mesh tally data
    mesh_tally = sp.get_tally(name='mesh tally')
    openmc_fission_rates = mesh_tally.get_values(scores=['nu-fission'])
    
    # Close the statepoint file now that we're done getting information from it
    sp.close()
    
    # Reshape array to 2D for plotting
    openmc_fission_rates.shape = (fp_grid, fp_grid)
    
    # Normalize to the average pin power
    openmc_fission_rates /= np.mean(openmc_fission_rates[openmc_fission_rates > 0.])
    
    # Create OpenMOC Mesh on which to tally fission rates
    openmoc_mesh = openmoc.process.Mesh()
    openmoc_mesh.dimension = np.array(mesh.dimension)
    openmoc_mesh.lower_left = np.array(mesh.lower_left)
    openmoc_mesh.upper_right = np.array(mesh.upper_right)
    openmoc_mesh.width = openmoc_mesh.upper_right - openmoc_mesh.lower_left
    openmoc_mesh.width /= openmoc_mesh.dimension
    
    # Tally OpenMOC fission rates on the Mesh
    openmoc_fission_rates = openmoc_mesh.tally_fission_rates(solver)
    openmoc_fission_rates = np.squeeze(openmoc_fission_rates)
    openmoc_fission_rates = np.fliplr(openmoc_fission_rates)
    
    # Normalize to the average pin fission rate
    openmoc_fission_rates /= np.mean(openmoc_fission_rates[openmoc_fission_rates > 0.])
    
    # Ignore zero fission rates in guide tubes with Matplotlib color scheme
    openmc_fission_rates[openmc_fission_rates == 0] = np.nan
    openmoc_fission_rates[openmoc_fission_rates == 0] = np.nan
    
    # Plot OpenMOC's fission rates in the right subplot
    fig2 = plt.subplot(122)
    plt.imshow(openmoc_fission_rates, interpolation='none', cmap='jet')
    plt.title('Laju Fisi')
    plt.colorbar()
    
    return plt.show()