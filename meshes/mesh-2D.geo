// Parameters
H = 0.41;        // Total height (0.15 + 0.1 + 0.16)
L = 2.2;         // Total length
D = 0.1;         // Cylinder diameter
cx = 0.2;        // Cylinder center X (0.15 + radius)
cy = 0.2;        // Cylinder center Y (0.15 + radius)
lc = 0.05;       // Global mesh size
lc_cyl = 0.005;  // Refined mesh size near cylinder

// Points for the outer rectangle
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};

// Lines for the outer rectangle
Line(1) = {1, 2}; // Bottom Wall
Line(2) = {2, 3}; // Outlet
Line(3) = {3, 4}; // Top Wall
Line(4) = {4, 1}; // Inlet

// Create the cylinder using a Circle Loop
Point(5) = {cx, cy, 0, lc_cyl};
Point(6) = {cx + D/2, cy, 0, lc_cyl};
Point(7) = {cx, cy + D/2, 0, lc_cyl};
Point(8) = {cx - D/2, cy, 0, lc_cyl};
Point(9) = {cx, cy - D/2, 0, lc_cyl};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Define Surfaces
Curve Loop(1) = {1, 2, 3, 4};       // Outer boundary
Curve Loop(2) = {5, 6, 7, 8};       // Inner hole (cylinder)
Plane Surface(1) = {1, 2};          // Surface with a hole

// Physical Groups for C++ Identification
Physical Curve("inlet", 101) = {4};
Physical Curve("outlet", 102) = {2};
Physical Curve("walls", 103) = {1, 3};
Physical Curve("cylinder", 104) = {5, 6, 7, 8};
Physical Surface("fluid", 201) = {1};
