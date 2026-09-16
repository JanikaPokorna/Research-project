SetFactory("OpenCASCADE");

// Characteristic length
lc_min = 0.1;
lc_max = 0.5;

// Points
Point(1) = {0.0, 0.5, 0, lc_max};
Point(2) = {0.2, 0.1, 0, lc_max};
Point(3) = {0.7, 0.0, 0, lc_max};
Point(4) = {1.0, 0.4, 0, lc_max};
Point(5) = {0.8, 0.9, 0, lc_max};
Point(6) = {0.5, 1.0, 0, lc_max};
Point(7) = {0.2, 0.8, 0, lc_max};

// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6}; 
Line(6) = {6, 7}; 
Line(7) = {7, 1}; 

// Curve loop - closed chain of lines forming the boundary
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};

// Plane surface
Plane Surface(1) = {1};

// Physical groups
Physical Surface("Omega") = {1};
Physical Curve("Gamma_1") = {1};
Physical Curve("Gamma_2") = {2};
Physical Curve("Gamma_3") = {3};
Physical Curve("Gamma_4") = {4};
Physical Curve("Gamma_5") = {5};
Physical Curve("Gamma_6") = {6};
Physical Curve("Gamma_7") = {7};