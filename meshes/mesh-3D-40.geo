SetFactory("OpenCASCADE");

// FORZA IL FORMATO COMPATIBILE CON DEAL.II
Mesh.MshFileVersion = 2.2; 

// --- Parameters ---
L = 2.5;       // Length (z)
H = 0.41;      // Height (y)
W = 0.41;      // Width (x)
D = 0.1;       // Cylinder Diameter
Cyl_z = 0.45;  // Cylinder center Z position
Cyl_y = 0.2;   // Cylinder center Y position

// Mesh Sizes (Refined for Level 40 - nodes along diameter)
// Target: 40 nodes along D = 0.1 -> lc_cyl = 0.1 / 40 = 0.0025
// Global scaling: lc_global = 10 * lc_cyl = 0.025
lc_global = 0.025;  
lc_cyl = 0.0025;      

// --- Geometry Construction ---
Box(1) = {0, 0, 0, W, H, L};
Cylinder(2) = {-0.1, Cyl_y, Cyl_z, W + 0.2, 0, 0, D/2};
BooleanDifference(3) = { Volume{1}; Delete; } { Volume{2}; Delete; };

// --- Mesh Refinement (Box locale invece di Distance/Threshold) ---
Field[1] = Box;
Field[1].VIn = lc_cyl;
Field[1].VOut = lc_global;
// Il box racchiude il cilindro e si estende lungo z per coprire la scia
Field[1].XMin = 0.0; Field[1].XMax = W;
Field[1].YMin = 0.1; Field[1].YMax = 0.3;
Field[1].ZMin = Cyl_z - 0.1; Field[1].ZMax = Cyl_z + 0.6; 

Background Field = 1;

// --- Physical Groups ---
eps = 1e-3; 
inlet_surf[] = Surface In BoundingBox{-eps, -eps, -eps, W+eps, H+eps, eps};
Physical Surface("inlet", 101) = {inlet_surf[]};

outlet_surf[] = Surface In BoundingBox{-eps, -eps, L-eps, W+eps, H+eps, L+eps};
Physical Surface("outlet", 102) = {outlet_surf[]};

cyl_surf[] = Surface In BoundingBox{0+eps, 0+eps, 0+eps, W-eps, H-eps, L-eps};
Physical Surface("cylinder", 103) = {cyl_surf[]};

all_surfs[] = Boundary{ Volume{3}; };
Physical Surface("walls", 104) = {all_surfs[], -inlet_surf[], -outlet_surf[], -cyl_surf[]};
Physical Volume("fluid", 201) = {3};