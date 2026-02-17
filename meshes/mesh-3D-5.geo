SetFactory("OpenCASCADE");

// --- Compatibilità e Performance ---
Mesh.MshFileVersion = 2.2; 
General.NumThreads = 0; 
Mesh.Algorithm3D = 10; // HXT

// --- Parametri ---
L = 2.5;       
H = 0.41;      
W = 0.41;      
D = 0.1;       
Cyl_z = 0.45;  
Cyl_y = 0.2;   

// --- Mesh Sizes (Livello 5 - ULTRA COARSE) ---
// Target: 5 nodi sul diametro D=0.1
// lc_cyl = 0.1 / 5 = 0.02
lc_cyl = 0.02;      
// Fuori dal cilindro celle enormi (15cm)
lc_global = 0.15;    

// --- Geometria ---
Box(1) = {0, 0, 0, W, H, L};
Cylinder(2) = {-0.1, Cyl_y, Cyl_z, W + 0.2, 0, 0, D/2};
BooleanDifference(3) = { Volume{1}; Delete; } { Volume{2}; Delete; };

// --- Refinement (Box locale) ---
Field[1] = Box;
Field[1].VIn = lc_cyl;
Field[1].VOut = lc_global;

// Box stretto attorno al cilindro
Field[1].XMin = 0.0; Field[1].XMax = W;
Field[1].YMin = 0.1; Field[1].YMax = 0.3;
Field[1].ZMin = Cyl_z - 0.1; Field[1].ZMax = Cyl_z + 0.6; 

Background Field = 1;

// --- Physical Groups (ROBUSTI) ---
eps = 1e-3; 

// Inlet
inlet_surf[] = Surface In BoundingBox{-eps, -eps, -eps, W+eps, H+eps, eps};
Physical Surface("inlet", 101) = {inlet_surf[]};

// Outlet
outlet_surf[] = Surface In BoundingBox{-eps, -eps, L-eps, W+eps, H+eps, L+eps};
Physical Surface("outlet", 102) = {outlet_surf[]};

// Cylinder (Selezione precisa)
cyl_surf[] = Surface In BoundingBox{ -eps, Cyl_y - D/2 - eps, Cyl_z - D/2 - eps, 
                                     W+eps, Cyl_y + D/2 + eps, Cyl_z + D/2 + eps};
Physical Surface("cylinder", 103) = {cyl_surf[]};

// Walls
all_surfs[] = Boundary{ Volume{3}; };
Physical Surface("walls", 104) = {all_surfs[], -inlet_surf[], -outlet_surf[], -cyl_surf[]};

Physical Volume("fluid", 201) = {3};