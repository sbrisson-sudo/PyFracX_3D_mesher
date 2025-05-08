#!/usr/bin/env python3

#  Author : Sylvain BRISSON   
#  Created On : Wed Apr 24 2024

# Imports 
import os 
import pygmsh
import gmsh
import meshio
import argparse
import numpy as np 

def mesh_DFN_gmsh(input_geom, res):
    """Mesh a set of intesecting planes.

    :param input_geom: input geometry to mesh
    :type input_geom: meshio object
    :param res: mesh resolution
    :type res: float
    """

    with pygmsh.occ.Geometry() as geom:

        # adding points
        points = [geom.add_point(point) for point in input_geom.points]

        # adding lines
        loops = []
        # loop over cell type
        for cells in input_geom.cells:
            if not cells.dim == 2 : continue 
            # loop over cell 
            for cell in cells.data:
                nb_vertices = len(cell)
                lines = [geom.add_line(points[cell[i]],points[cell[(i+1)%nb_vertices]]) for i in range(nb_vertices)]
                loops.append(geom.add_curve_loop(lines))

        # adding surfaces
        surfaces = [geom.add_plane_surface(loop) for loop in loops]

        # surfaces union
        new_surfaces = geom.boolean_union([*surfaces])
        new_surfaces_dim_tag = [gmsh_obj.dim_tag for gmsh_obj in new_surfaces]

        geom.synchronize()

        # Forcing mesh size on the boundaries of these new surfaces
        gmsh.model.mesh.setSize(gmsh.model.getBoundary(new_surfaces_dim_tag, False, False, True),res)

        geom.synchronize()
        g_mesh = geom.generate_mesh(order=1,algorithm=2)

    # only keep elements of dim 2
    g_mesh.cells = [cells for cells in g_mesh.cells if cells.dim == 2 ]

    return g_mesh

def mesh_DFN_gmsh_injection_point(input_geom, res, inj_point_coor):
    """Mesh a set of intesecting planes.

    :param input_geom: input geometry to mesh
    :type input_geom: meshio object
    :param res: mesh resolution
    :type res: float
    :param inj_point_coor: 3D coordinates of the injection point to be enforced
    :type inj_point_coor: np.array(float)
    :param inj_fracture_id: index of the fracture in the input geometry to which the injection point belong
    :type inj_fracture_id: int
    :param res_inj_point: mesh resolution at injection point, default to res
    :type res_inj_point: float
    """

    with pygmsh.occ.Geometry() as geom:

        # adding points
        points = [geom.add_point(point) for point in input_geom.points]

        # adding lines
        loops = []
        # loop over cell type
        for cells in input_geom.cells:
            if not cells.dim == 2 : continue 
            # loop over cell 
            for cell in cells.data:
                nb_vertices = len(cell)
                lines = [geom.add_line(points[cell[i]],points[cell[(i+1)%nb_vertices]]) for i in range(nb_vertices)]
                loops.append(geom.add_curve_loop(lines))

        # adding surfaces
        surfaces = [geom.add_plane_surface(loop) for loop in loops]

        # Injection point
        inj_point = geom.add_point(inj_point_coor, res)

        geom.synchronize()

        # Computing fragments between surfaces and injection point
        fragments = geom.boolean_fragments(surfaces, inj_point)

        geom.synchronize()

        # Getting the resulting surfaces and imposing mesh resolution at their boundaries
        new_surfaces = [elmt for elmt in fragments if elmt.dim_tag[0] == 2]
        new_surfaces_dim_tag = [gmsh_obj.dim_tag for gmsh_obj in new_surfaces]
        gmsh.model.mesh.setSize(gmsh.model.getBoundary(new_surfaces_dim_tag, False, False, True),res)

        g_mesh = geom.generate_mesh(order=1,algorithm=2)

    # only keep elements of dim 2
    g_mesh.cells = [cells for cells in g_mesh.cells if cells.dim == 2 ]

    return g_mesh

def mesh_DFN_gmsh_injection_point_with_refinement(input_geom, inj_point_coor, res_fine, res_coarse, d_min, d_max, ):
    """Mesh a set of intesecting planes.

    :param input_geom: input geometry to mesh
    :type input_geom: meshio object
    :param res: mesh resolution
    :type res: float
    :param inj_point_coor: 3D coordinates of the injection point to be enforced
    :type inj_point_coor: np.array(float)
    :param inj_fracture_id: index of the fracture in the input geometry to which the injection point belong
    :type inj_fracture_id: int
    :param res_inj_point: mesh resolution at injection point, default to res
    :type res_inj_point: float
    :param res_fine: mesh resolution at injection point
    :type res_fine: float
    :param res:_coarse mesh resolution elsewhere
    :type res_coarse: float
    :param d_min: distance to injection point at which mesh size starts increasing
    :type d_min: float
    :param d_max: distance to injection point at which mesh size reaches res_coarse
    :type d_max: float
    """

    with pygmsh.occ.Geometry() as geom:

        # adding points
        points = [geom.add_point(point) for point in input_geom.points]

        # adding lines
        loops = []
        # loop over cell type
        for cells in input_geom.cells:
            if not cells.dim == 2 : continue 
            # loop over cell 
            for cell in cells.data:
                nb_vertices = len(cell)
                lines = [geom.add_line(points[cell[i]],points[cell[(i+1)%nb_vertices]]) for i in range(nb_vertices)]
                loops.append(geom.add_curve_loop(lines))

        # adding surfaces
        surfaces = [geom.add_plane_surface(loop) for loop in loops]

        # Injection point
        inj_point = geom.add_point(inj_point_coor)

        geom.synchronize()

        # Computing fragments between surfaces and injection point
        fragments = geom.boolean_fragments(surfaces, inj_point)

        geom.synchronize()

        # Imposing the mesh size field as a boundary layer (around the point of injection)
        size_field = geom.add_boundary_layer(
            lcmin = res_fine,
            lcmax = res_coarse,
            distmin = d_min,
            distmax = d_max,
            nodes_list = [inj_point]
            )

        geom.set_background_mesh([size_field], operator="Min")
        
        geom.synchronize()

        g_mesh = geom.generate_mesh(order=1,algorithm=2)

    # only keep elements of dim 2
    g_mesh.cells = [cells for cells in g_mesh.cells if cells.dim == 2 ]

    return g_mesh

class FakeGmshObject:
    def __init__(self, tag):
        self._id = tag


def mesh_DFN_gmsh_injection_point_with_refinement_at_intersections(input_geom, inj_point_coor, res_inj, res_intersec, res_coarse, d_min_inj, d_max_inj, d_min_intersec, d_max_intersec, out_file):
    """Mesh a set of intesecting planes.

    :param input_geom: input geometry to mesh
    :type input_geom: meshio object
    :param res: mesh resolution
    :type res: float
    :param inj_point_coor: 3D coordinates of the injection point to be enforced
    :type inj_point_coor: np.array(float)
    :param inj_fracture_id: index of the fracture in the input geometry to which the injection point belong
    :type inj_fracture_id: int
    :param res_inj_point: mesh resolution at injection point, default to res
    :type res_inj_point: float
    :param res_fine: mesh resolution at injection point
    :type res_fine: float
    :param res:_coarse mesh resolution elsewhere
    :type res_coarse: float
    :param d_min: distance to injection point at which mesh size starts increasing
    :type d_min: float
    :param d_max: distance to injection point at which mesh size reaches res_coarse
    :type d_max: float
    """

    with pygmsh.occ.Geometry() as geom:

        # adding points
        points = [geom.add_point(point) for point in input_geom.points]

        # adding lines
        loops = []
        # loop over cell type
        for cells in input_geom.cells:
            if not cells.dim == 2 : continue 
            # loop over cell 
            for cell in cells.data:
                nb_vertices = len(cell)
                lines = [geom.add_line(points[cell[i]],points[cell[(i+1)%nb_vertices]]) for i in range(nb_vertices)]
                loops.append(geom.add_curve_loop(lines))

        # adding surfaces
        surfaces = [geom.add_plane_surface(loop) for loop in loops]
        nb_fractures = len(surfaces)
        
        print(f"{len(surfaces)} surfaces to mesh")

        # Injection point
        inj_point = geom.add_point(inj_point_coor)

        geom.synchronize()
        
        gmsh.model.occ.synchronize()  # Synchronize the OpenCASCADE geometry
        
        surfaces_before_fragments = gmsh.model.getEntities(dim=2)
        curves_before_fragments = gmsh.model.getEntities(dim=1)
    
        # We compute their director vector
        coor_points_lines_before_fragments = []
        for curve in curves_before_fragments:
            curve_tag = curve[1]
            line_points = gmsh.model.getAdjacencies(1, curve_tag)[1]  # Get points of the line
            if len(line_points) != 2:
                raise ValueError("A line should have exactly 2 points.")
            p1 = np.array(gmsh.model.getValue(0, line_points[0], []))
            p2 = np.array(gmsh.model.getValue(0, line_points[1], []))
            coor_points_lines_before_fragments.extend([p1,p2])
            
        coor_points_surfaces_before_fragments = []
        for surface in surfaces_before_fragments:
            surface_tag = surface[1]
            surface_lines = gmsh.model.getAdjacencies(2, surface_tag)[1]
            # print(f"Surface {surface_tag} : lines = {surface_lines}")
            coor_points_surface = []
            for line_tag in surface_lines:
                surface_line_points = gmsh.model.getAdjacencies(1, line_tag)[1]
                # print(f"Line {line_tag} : points = {surface_line_points}")
                coor_points_surface.extend([gmsh.model.getValue(0, point, []) for point in surface_line_points])
            coor_points_surfaces_before_fragments.append(coor_points_surface)
            
        sub_frac_id = [0 for _ in range(nb_fractures)]
        sub_frac_tags = [[] for _ in range(nb_fractures)]
        # Computing fragments between surfaces and injection point
        fragments = geom.boolean_fragments(surfaces, inj_point)
        
        new_surfaces = [elmt for elmt in fragments if elmt.dim_tag[0] == 2]
        print(f"{len(new_surfaces)} surfaces to mesh after boolean_fragments")

        geom.synchronize()
        
        # Assign physical id to the new surfaces
        sub_frac_count = 1 # gmsh is 1-based
        for elmt in fragments:
            if elmt.dim == 2:
                # We look for common points with the unfragmented surfaces
                frac_id_found = False
                surface_tag = elmt.dim_tag[1]
                surface_lines = gmsh.model.getAdjacencies(2, surface_tag)[1]
                for line_tag in surface_lines:
                    surface_line_points = gmsh.model.getAdjacencies(1, line_tag)[1]
                    coor_surface_points = [gmsh.model.getValue(0, point, []) for point in surface_line_points]
                    for frac_id,coor_unfragmented_surface_points in enumerate(coor_points_surfaces_before_fragments):
                        for ptA in coor_surface_points:
                            for ptB in coor_unfragmented_surface_points:
                                if np.all(np.isclose(ptA, ptB)):
                                    # Common point found
                                    frac_id_found = True 
                                    # We put the correspondance in the map 
                                    frag_surf_id = f"{frac_id}_{sub_frac_id[frac_id]}"
                                    
                                    # This goes though pygmsh, on the gmsh side 
                                    # the physical group is only associated to a tag
                                    geom.add_physical(elmt, label=frag_surf_id)
                                    
                                    sub_frac_tags[frac_id].append(sub_frac_count)
                                    sub_frac_count += 1
                                    sub_frac_id[frac_id]+=1
                                    break
                            if frac_id_found: break
                        if frac_id_found: break
                    if frac_id_found: break
        
        curves_after_fragments = gmsh.model.getEntities(dim=1)
        
        # Identify the intersection curve
        intersection_curves = []
        for curve in curves_after_fragments:
            curve_tag = curve[1]
            line_points = gmsh.model.getAdjacencies(1, curve_tag)[1]  # Get points of the line
            if len(line_points) != 2:
                raise ValueError("A line should have exactly 2 points.")
            p1 = np.array(gmsh.model.getValue(0, line_points[0], []))
            p2 = np.array(gmsh.model.getValue(0, line_points[1], []))
            
            # If one of the two ending points was among the prior lines -> not an intersection
            for p in coor_points_lines_before_fragments:
                is_in_prior_list = False
                if np.all(np.isclose(p1, p)) or np.all(np.isclose(p2, p)):
                    is_in_prior_list = True
                    break
            
            if not(is_in_prior_list):
                # print(f"Intersecting curve coordinates : {p1}, {p2}")
                intersection_curves.append(curve)
                
        print(f"{len(intersection_curves)} intersecting curves")

        # Imposing the mesh size field as a boundary layer (around the point of injection)
        size_field_inj = geom.add_boundary_layer(
            lcmin = res_inj,
            lcmax = res_coarse,
            distmin = d_min_inj,
            distmax = d_max_inj,
            nodes_list = [inj_point]
            )

        if intersection_curves:
            
            edges_list = [FakeGmshObject(curve_tag) for _,curve_tag in intersection_curves]
            
            size_field_intersections = geom.add_boundary_layer(
                lcmin=res_intersec,
                lcmax=res_coarse,
                distmin=d_min_intersec,
                distmax=d_max_intersec,
                edges_list=edges_list,
                num_points_per_curve = 100
            )        
            
            # Combine size fields
            geom.set_background_mesh([size_field_inj, size_field_intersections], operator="Min")
            # geom.set_background_mesh([size_field_intersections], operator="Min")
        else:
            geom.set_background_mesh([size_field_inj], operator="Min")
        
        geom.synchronize()
        
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        g_mesh = geom.generate_mesh(order=1,algorithm=2)
        
        # We save the mesh in a tmp file
        # But we want to change the fracture ids to the correct ones 
        gmsh.write("tmp.vtk")
        
    # Now we open it again using meshio and apply the fragment to fracture mapping
    mesh = meshio.read("tmp.vtk")
    
    mapping_dict = {fragment_id: frac_id for frac_id, list_fragment_id in enumerate(sub_frac_tags) for fragment_id in list_fragment_id}

    if "CellEntityIds" in mesh.cell_data:
        data = mesh.cell_data[f"CellEntityIds"][0]
        mapped_data = np.vectorize(lambda x: mapping_dict.get(x, x))(data)
        mesh.cell_data["CellEntityIds" ][0] = mapped_data
        mesh.cell_data["FractureID"] = mesh.cell_data.pop("CellEntityIds")
        
    else :
        print("Error : no CellEntityIds in the output gmsh file")
    
    os.remove("tmp.vtk")
    meshio.write(out_file, mesh)
        
    return g_mesh

if __name__ == "__main__":
        
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Tool to mesh 3D discrete fracture networkd using gmsh.")
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('-res', type=float, required=True, help='Resolution as a float value')
    args = parser.parse_args()

    # Input geometry file describing polygons (here : obj format)
    assert os.path.exists(args.input_file), f"Input file {args.input_file} not found"
    geom_obj = meshio.read(args.input_file)
    nb_surfaces = 0
    for cells in geom_obj.cells:
        if cells.dim == 2:
            nb_surfaces += len(cells.data)
    print(f"Input geometry file read : {nb_surfaces} planes to mesh.")

    # Generating mesh 
    mesh = mesh_DFN_gmsh(geom_obj, args.res)

    print("Mesh generated :")
    print(mesh)
    mesh.write(args.output_file)
    print(f"Wrote {args.output_file}")
