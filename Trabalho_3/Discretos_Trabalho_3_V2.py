from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import petsc4py.PETSc as PETSc
import ufl
import pyvista
import numpy as np

def boundary(x):
    return np.isclose(x[0], 0.0)

def w_muscle_ufl(T):
    condition = ufl.le(T, 45.0)
    return ufl.conditional(condition, 0.45 + 3.55 * ufl.exp(-((T - 45.0) ** 2) / 12.0), 4.0)

def w_fat_ufl(T):
    condition = ufl.le(T, 45.0)
    return ufl.conditional(condition, 0.36 + 0.36 * ufl.exp(-((T - 45.0) ** 2) / 12.0), 0.72)

def w_tumor_ufl(T):
    condition1 = ufl.lt(T, 37.0)
    condition2 = ufl.le(T, 42.0)
    return ufl.conditional(condition1, 0.833, ufl.conditional(condition2, 0.833 - ((T - 37.0) ** 4.8) / (5.438 * (10 ** 3)), 0.416))

def w_dermis_ufl(T):
    return 0.5

def Qr_ufl(x):
    # Injections Points and Parameters
    x1, y1, z1 = 0.020, 0.055, 0.055
    A1, r01 = 0.8e6, 0.6e-3
    x2, y2, z2 = 0.025, 0.045, 0.045
    A2, r02 = 0.7e6, 0.6e-3
    x3, y3, z3 = 0.015, 0.040, 0.040
    A3, r03 = 0.7e6, 0.6e-3

    r1 = ufl.sqrt((x[0] - x1)**2 + (x[1] - z1)**2)
    r2 = ufl.sqrt((x[0] - x2)**2 + (x[1] - z2)**2)
    r3 = ufl.sqrt((x[0] - x3)**2 + (x[1] - z3)**2)

    Qr_value = A1 * ufl.exp(-r1**2 / r01**2) + A2 * ufl.exp(-r2**2 / r02**2) + A3 * ufl.exp(-r3**2 / r03**2)
    return Qr_value

msh_file = "./Tecido.msh"

mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD)

# Espaço de Funções
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# Condição de Contorno de Dirichlet
T_D0 = PETSc.ScalarType(37)

facets = dolfinx.mesh.locate_entities_boundary(
    mesh,
    dim = (mesh.topology.dim - 1),
    marker = lambda x: np.isclose(x[0], 0.0),
)

dofs = dolfinx.fem.locate_dofs_topological(V = V, entity_dim = mesh.topology.dim - 1, entities = facets)

bc = dolfinx.fem.dirichletbc(value = T_D0, dofs = dofs, V = V)

bcs = [bc]

T = ufl.TrialFunction(V)
T_old = dolfinx.fem.Function(V)

Qm = dolfinx.fem.Function(V)
c = dolfinx.fem.Function(V)
cb = dolfinx.fem.Function(V)
rho = dolfinx.fem.Function(V)
wb_barra = dolfinx.fem.Function(V)
k = dolfinx.fem.Function(V)

dx = ufl.Measure("dx", domain = mesh, subdomain_data = cell_tags)

v = ufl.TestFunction(V)

Ta = PETSc.ScalarType(37.0)

# derme   1 
# gordura 2
# tumor   3
# musculo 4
def parameters(tag):
  if tag == 3:
    Qm       = PETSc.ScalarType(4200.0)
    c        = PETSc.ScalarType(4200.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1000.0)
    wb_barra = PETSc.ScalarType(0.757981)
    k        = PETSc.ScalarType(0.55)
  elif tag == 4:
    Qm       = PETSc.ScalarType(420.0)
    c        = PETSc.ScalarType(3800.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1000.0)
    wb_barra = PETSc.ScalarType(1.87895)
    k        = PETSc.ScalarType(0.45)
  elif tag == 2:
    Qm       = PETSc.ScalarType(420.0)
    c        = PETSc.ScalarType(2500.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1000.0)
    wb_barra = PETSc.ScalarType(0.504908)
    k        = PETSc.ScalarType(0.21)
  elif tag == 1:
    Qm       = PETSc.ScalarType(420.0)
    c        = PETSc.ScalarType(3600.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1200.0)
    wb_barra = PETSc.ScalarType(0.5)
    k        = PETSc.ScalarType(0.40)
  return Qm, c, cb, rho, wb_barra, k

qm_array = Qm.x.array
c_array = c.x.array
cb_array = cb.x.array
rho_array = rho.x.array
wb_barra_array = wb_barra.x.array
k_array = k.x.array

cell_tags_values = cell_tags.values
cell_tags_indices = cell_tags.indices

for cell_id in range(mesh.topology.index_map(mesh.topology.dim).size_local):
    tag = cell_tags_values[cell_tags_indices[cell_id]]
    Qm_, c_, cb_, rho_, wb_barra_, k_ = parameters(tag)
    
    dofs = V.dofmap.cell_dofs(cell_id)
    for dof in dofs:
        qm_array[dof] = Qm_
        c_array[dof] = c_
        cb_array[dof] = cb_
        rho_array[dof] = rho_
        wb_barra_array[dof] = wb_barra_
        k_array[dof] = k_

Qm.x.array[:] = qm_array
c.x.array[:] = c_array
cb.x.array[:] = cb_array
rho.x.array[:] = rho_array
wb_barra.x.array[:] = wb_barra_array
k.x.array[:] = k_array

a = k * ufl.inner(ufl.grad(T), ufl.grad(v)) * dx + wb_barra * cb * T * v * dx
L = Qm * v * dx + wb_barra * cb * Ta * v * dx

problem = LinearProblem(a, L, bcs = bcs)
T_sol = problem.solve()

cells, types, x = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["T"] = T_sol.x.array.real
grid.set_active_scalars("T")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)
    plotter.screenshot("Discretos_Trabalho_3.png")
else:
    plotter.show()

T_old.x.array[:] = T_sol.x.array[:]

dt = PETSc.ScalarType(0.1)
num_steps = 30000

wb = dolfinx.fem.Function(V)
wb_array = wb.x.array

for s in range(num_steps):
    print(f'Step {s} de {num_steps}')
    cell_tags_values = cell_tags.values
    cell_tags_indices = cell_tags.indices

    for cell_id in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        tag = cell_tags_values[cell_tags_indices[cell_id]]
        T_value = np.mean(T_old.x.array[V.dofmap.cell_dofs(cell_id)])

        if (tag == 1):
            wb_ = w_dermis_ufl(T_value)
        elif (tag == 2):
            wb_ = w_fat_ufl(T_value)
        elif (tag == 3):
            wb_ = w_tumor_ufl(T_value)
        elif (tag == 4):
            wb_ = w_muscle_ufl(T_value)
        dofs = V.dofmap.cell_dofs(cell_id)
        for dof in dofs:
            wb_array[dof] = wb_

    wb.x.array[:] = wb_array
    Qr = Qr_ufl(ufl.SpatialCoordinate(mesh))

    a = k * ufl.inner(ufl.grad(T), ufl.grad(v)) * dx + wb * cb * T * v * dx
    L = Qm * v * dx + wb * cb * Ta * v * dx + Qr * v * dx
    problem = LinearProblem(a, L, bcs = bcs)
    T_sol = problem.solve()

    # cells, types, x = dolfinx.plot.vtk_mesh(V)
    # grid = pyvista.UnstructuredGrid(cells, types, x)
    # grid.point_data["T"] = T_sol.x.array.real
    # grid.set_active_scalars("T")
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    # if pyvista.OFF_SCREEN:
    #     pyvista.start_xvfb(wait=0.1)
    #     plotter.screenshot("Discretos_Trabalho_3.png")
    # else:
    #     plotter.show()

    T_old.x.array[:] = T_sol.x.array[:]

cells, types, x = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["T"] = T_sol.x.array.real
grid.set_active_scalars("T")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait = 0.1)
    plotter.screenshot("Discretos_Trabalho_3.png")
else:
    plotter.show()