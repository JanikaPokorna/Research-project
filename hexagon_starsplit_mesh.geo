//base structure of the complex hexagonal mesh

SetFactory("OpenCASCADE");

// ---------------------------
// Mesh size parameters
// ---------------------------
lc_min = 0.05;
lc_max = 0.20;

// ---------------------------
// Outer HEXAGON (6 points)
// (roughly centered, easy geometry)
// ---------------------------
Point(1) = {0.0, 0.5, 0, lc_max};   // left
Point(2) = {0.2, 0.1, 0, lc_max};   // bottom-left
Point(3) = {0.7, 0.1, 0, lc_max};   // bottom-right
Point(4) = {1.0, 0.5, 0, lc_max};   // right
Point(5) = {0.7, 0.9, 0, lc_max};   // top-right
Point(6) = {0.2, 0.9, 0, lc_max};   // top-left
Point(101) = {0.5, 0.5, 0, lc_min};  // center

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6}; //creates boundary from lines
Plane Surface(1) = {1};


// Star from center to outer corners (splits into 6 "pie slices")
Line(101) = {101, 1};
Line(102) = {101, 2};
Line(103) = {101, 3};
Line(104) = {101, 4};
Line(105) = {101, 5};
Line(106) = {101, 6};


// ---------------------------
// Fragment surface with star lines
// This splits the hexagon into 6 triangles
// ---------------------------
BooleanFragments{ Surface{1}; Delete;}{
  Curve{101}; Curve{102}; Curve{103};
  Curve{104}; Curve{105}; Curve{106}; 
}

// ---------------------------
// Physical groups
// ---------------------------
Physical Surface("Domain") = Surface{:};

Physical Curve("Outer_1") = {1};
Physical Curve("Outer_2") = {2};
Physical Curve("Outer_3") = {3};
Physical Curve("Outer_4") = {4};
Physical Curve("Outer_5") = {5};
Physical Curve("Outer_6") = {6};