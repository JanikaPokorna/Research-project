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

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1}; // main "city" surface


// ---------------------------
// Interior points (district vertices)
// Pick points inside; adjust as you like
// ---------------------------
Point(101) = {0.50, 0.50, 0, lc_min};  // center
Point(102) = {0.35, 0.35, 0, lc_min};
Point(103) = {0.65, 0.35, 0, lc_min};
Point(104) = {0.70, 0.60, 0, lc_min};
Point(105) = {0.50, 0.75, 0, lc_min};
Point(106) = {0.30, 0.60, 0, lc_min};


// ---------------------------
// Internal splitting lines (district borders)
// IMPORTANT: these must be inside the surface
// These lines will be "imprinted" into the surface by BooleanFragments.
// ---------------------------

// Star from center to outer corners (splits into 6 "pie slices")
Line(101) = {101, 1};
Line(102) = {101, 2};
Line(103) = {101, 3};
Line(104) = {101, 4};
Line(105) = {101, 5};
Line(106) = {101, 6};

// Additional internal borders to create non-triangular districts
Line(107) = {102, 103}; // bottom chord
Line(108) = {103, 104}; // diagonal right
Line(109) = {104, 105}; // upper-right chord
Line(110) = {105, 106}; // upper-left chord
Line(111) = {106, 102}; // diagonal left

// A couple of cross-cuts (optional, makes more districts)
// Line(112) = {102, 101};
// Line(113) = {103, 101};
// Line(114) = {105, 101};
// Line(115) = {106, 101};


// ---------------------------
// Fragment the main surface by the internal curves
// This creates multiple surfaces that SHARE the internal boundaries.
// That is the "conforming interface" guarantee.
// ---------------------------
BooleanFragments{ Surface{1}; Delete; }{ Curve{101:111}; Delete; }
Physical Surface("Domain") = Surface{:};
// ---------------------------
// Physical groups
// Outer boundary curves: keep for BCs
// After fragmentation, the original outer lines usually survive with same IDs,
// but OCC can renumber things in some cases.
// We'll still define physical groups for the original outer edges.
// ---------------------------
Physical Curve("Outer_1") = {1};
Physical Curve("Outer_2") = {2};
Physical Curve("Outer_3") = {3};
Physical Curve("Outer_4") = {4};
Physical Curve("Outer_5") = {5};
Physical Curve("Outer_6") = {6};