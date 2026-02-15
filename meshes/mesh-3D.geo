SetFactory("OpenCASCADE");

// --- Parameters ---
L = 2.5;       // Length (z)
H = 0.41;      // Height (y)
W = 0.41;      // Width (x)
D = 0.1;       // Cylinder Diameter
Cyl_z = 0.45;  // Cylinder center Z position
Cyl_y = 0.2;   // Cylinder center Y position (0.15 + 0.05)

// Mesh Sizes
lc_global = 0.1;    // Coarser mesh far away
lc_cyl = 0.02;      // Fine mesh near cylinder

// --- Geometry Construction ---

// 1. Create the main Channel (Box)
//    Box(Tag) = {x, y, z, dx, dy, dz}
Box(1) = {0, 0, 0, W, H, L};

// 2. Create the Cylinder
//    We extend it slightly outside the walls (x = -0.1 to W + 0.1)
//    to ensure the Boolean cut is clean and robust.
//    Cylinder(Tag) = {center_x, center_y, center_z, dx, dy, dz, radius}
Cylinder(2) = {-0.1, Cyl_y, Cyl_z, W + 0.2, 0, 0, D/2};

// 3. Subtract the Cylinder from the Channel
//    BooleanDifference returns the volume of object 1 minus object 2.
BooleanDifference(3) = { Volume{1}; Delete; } { Volume{2}; Delete; };

// --- Mesh Refinement ---
// We use a "Distance Field" to refine the mesh only near the cylinder surface
Field[1] = Distance;
Field[1].CurvesList = {7, 8}; // IDs of the circle curves (Gmsh usually assigns these)
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_cyl;
Field[2].LcMax = lc_global;
Field[2].DistMin = 0.05; // Fine mesh within 5cm of cylinder
Field[2].DistMax = 0.5;  // Transitions to coarse mesh over 0.5m

Background Field = 2;

// --- Physical Groups (Crucial for C++) ---
// We use "BoundingBox" to automatically find the correct faces
// based on their coordinates.

eps = 1e-3; // Tolerance

// Inlet: Face at z = 0
inlet_surf[] = Surface In BoundingBox{-eps, -eps, -eps, W+eps, H+eps, eps};
Physical Surface("inlet", 101) = {inlet_surf[]};

// Outlet: Face at z = 2.5
outlet_surf[] = Surface In BoundingBox{-eps, -eps, L-eps, W+eps, H+eps, L+eps};
Physical Surface("outlet", 102) = {outlet_surf[]};

// Cylinder: The curved surface inside the volume
// We find it by excluding the outer box limits
cyl_surf[] = Surface In BoundingBox{0+eps, 0+eps, 0+eps, W-eps, H-eps, L-eps};
Physical Surface("cylinder", 103) = {cyl_surf[]};

// Walls: Top, Bottom, and Sides (Everything else)
// We select all boundaries of the volume...
all_surfs[] = Boundary{ Volume{3}; };
// ...and subtract the ones we already identified.
Physical Surface("walls", 104) = {all_surfs[], -inlet_surf[], -outlet_surf[], -cyl_surf[]};

// The Fluid Volume
Physical Volume("fluid", 201) = {3};