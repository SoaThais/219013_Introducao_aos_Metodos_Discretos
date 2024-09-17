Merge "derme_.stl"; // Derme
Merge "gordura_.stl"; // Gordura
Merge "tumor_.stl"; // Tumor
Merge "musculo_.stl"; // Musculo

Coherence Mesh;

Surface Loop(1) = {1, 2, 3, 4};

Physical Surface("derme", 1) = {1};
Physical Surface("gordura", 2) = {2};
Physical Surface("tumor", 3) = {3};
Physical Surface("musculo", 4) = {4};

Volume(1) = {1};

