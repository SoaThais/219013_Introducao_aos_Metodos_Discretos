# Bibliotecas
from   mpi4py import MPI
from   dolfinx.fem.petsc import LinearProblem

import dolfinx
import petsc4py.PETSc as PETSc
import ufl
import pyvista
import numpy as np

# Funções e parâmetros para os diferentes tipos de tecido
def w_muscle(T):
    condition = ufl.le(T, 45.0)
    return ufl.conditional(condition, 0.45 + 3.55 * ufl.exp(-((T - 45.0) ** 2) / 12.0), 4.0)

def w_fat(T):
    condition = ufl.le(T, 45.0)
    return ufl.conditional(condition, 0.36 + 0.36 * ufl.exp(-((T - 45.0) ** 2) / 12.0), 0.72)

def w_tumor(T):
    condition1 = ufl.lt(T, 37.0)
    condition2 = ufl.le(T, 42.0)
    return ufl.conditional(condition1, 0.833, ufl.conditional(condition2, 0.833 - ((T - 37.0) ** 4.8) / (5.438 * (10 ** 3)), 0.416))

def w_dermis():
    return 0.5

def parameters(tag):
  # Tumor 
  if tag == 3:
    Qm       = PETSc.ScalarType(4200.0)
    c        = PETSc.ScalarType(4200.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1000.0)
    wb_barra = PETSc.ScalarType(0.757981)
    k        = PETSc.ScalarType(0.55)
  # Musculo
  elif tag == 4:
    Qm       = PETSc.ScalarType(420.0)
    c        = PETSc.ScalarType(3800.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1000.0)
    wb_barra = PETSc.ScalarType(1.87895)
    k        = PETSc.ScalarType(0.45)
  # Gordura
  elif tag == 2:
    Qm       = PETSc.ScalarType(420.0)
    c        = PETSc.ScalarType(2500.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1000.0)
    wb_barra = PETSc.ScalarType(0.504908)
    k        = PETSc.ScalarType(0.21)
  # Derme
  elif tag == 1:
    Qm       = PETSc.ScalarType(420.0)
    c        = PETSc.ScalarType(3600.0)
    cb       = PETSc.ScalarType(4200.0)
    rho      = PETSc.ScalarType(1200.0)
    wb_barra = PETSc.ScalarType(0.5)
    k        = PETSc.ScalarType(0.40)
  return Qm, c, cb, rho, wb_barra, k

def Qr_calculation(x):
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

# Importa o .msh
msh_file = "./Tecido.msh"

# Obtém a malha, os rótulos de superfície (1 - Derme, 2 - Gordura, 3 - Tumor, 4 - Musculo) e de contorno (null)
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD)

# Espaço de Funções
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# Valor para condição de contorno de Dirichlet em x = 0
T_D0 = PETSc.ScalarType(37)

# Contorno em x = 0
facets = dolfinx.mesh.locate_entities_boundary(
    mesh,
    dim = (mesh.topology.dim - 1),
    marker = lambda x: np.isclose(x[0], 0.0),
)

# Graus de liberdade em x = 0
dofs = dolfinx.fem.locate_dofs_topological(V = V, entity_dim = mesh.topology.dim - 1, entities = facets)

# Criação da condição de contorno em x = 0
bc = dolfinx.fem.dirichletbc(value = T_D0, dofs = dofs, V = V)

bcs = [bc]

# Funções de temperatura atual (T) e anterior (T_old)
T     = ufl.TrialFunction(V)
T_old = dolfinx.fem.Function(V)

# Discretização espacial
dx = ufl.Measure("dx", domain = mesh, subdomain_data = cell_tags)

# Função teste
v = ufl.TestFunction(V)

# Temperatura arterial
Ta = PETSc.ScalarType(37.0)

# Geração de calor metabólico
Qm       = dolfinx.fem.Function(V)
# Calor especifico do tecido
c        = dolfinx.fem.Function(V)
# Calor específico do sangue
cb       = dolfinx.fem.Function(V)
# Densidade do tecido
rho      = dolfinx.fem.Function(V)
# Taxa de perfusão sanguínea média
wb_barra = dolfinx.fem.Function(V)
# Condutividade térmica
k        = dolfinx.fem.Function(V)

# Criação de arrays a partir das funções anteriores
qm_array = Qm.x.array
c_array = c.x.array
cb_array = cb.x.array
rho_array = rho.x.array
wb_barra_array = wb_barra.x.array
k_array = k.x.array

# Obtenção dos arrays com os valores e indices de cada célula
cell_tags_values = cell_tags.values
cell_tags_indices = cell_tags.indices

for cell_id in range(mesh.topology.index_map(mesh.topology.dim).size_local):
    # Obtenção do rótulo de superfície da célula cell_id
    tag = cell_tags_values[cell_tags_indices[cell_id]]
    # Obtenção dos parâmetros da célula a partir do seu rótulo
    Qm_, c_, cb_, rho_, wb_barra_, k_ = parameters(tag)
    
    # Graus de liberdade da célula
    dofs = V.dofmap.cell_dofs(cell_id)

    # Atribuição dos parâmetros encontrados para cada grau de liberdade da célula
    for dof in dofs:
        qm_array[dof] = Qm_
        c_array[dof] = c_
        cb_array[dof] = cb_
        rho_array[dof] = rho_
        wb_barra_array[dof] = wb_barra_
        k_array[dof] = k_

# Atualização dos valores das funções
Qm.x.array[:] = qm_array
c.x.array[:] = c_array
cb.x.array[:] = cb_array
rho.x.array[:] = rho_array
wb_barra.x.array[:] = wb_barra_array
k.x.array[:] = k_array

# Montagem do sistema em estado estacionário = condição inicial do problema
a = k * ufl.inner(ufl.grad(T), ufl.grad(v)) * dx + wb_barra * cb * T * v * dx
L = Qm * v * dx + wb_barra * cb * Ta * v * dx

# Resolução do sistema
problem = LinearProblem(a, L, bcs = bcs)
T_sol = problem.solve()
T_old.x.array[:] = T_sol.x.array[:]

# Plotagem da condição inicial
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

# Passo de tempo
dt = PETSc.ScalarType(0.1)
# Número de passos de tempo
num_steps = 30000

# Taxa de perfusão sanguinea
wb = dolfinx.fem.Function(V)
wb_array = wb.x.array

for s in range(num_steps):
    print(f'Step {s} de {num_steps}')
    
    # Cálculo da taxa de perfusão sanguínea (wb) em função de T_old
    for cell_id in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        tag = cell_tags_values[cell_tags_indices[cell_id]]

        # Recupera temperatura da célula em T_old
        T_value = np.mean(T_old.x.array[V.dofmap.cell_dofs(cell_id)])

        # Cálculo de wb para a célula
        if (tag == 1):
            wb_ = w_dermis()
        elif (tag == 2):
            wb_ = w_fat(T_value)
        elif (tag == 3):
            wb_ = w_tumor(T_value)
        elif (tag == 4):
            wb_ = w_muscle(T_value)
        
        # Atribuição de wb para todos os graus de liberdade da célula
        dofs = V.dofmap.cell_dofs(cell_id)
        for dof in dofs:
            wb_array[dof] = wb_

    # Atualização da função wb
    wb.x.array[:] = wb_array

    # Cálculo de Qr 
    Qr = Qr_calculation(ufl.SpatialCoordinate(mesh))

    # Montagem do sistema 
    a = k * ufl.inner(ufl.grad(T), ufl.grad(v)) * dx + wb * cb * T * v * dx
    L = Qm * v * dx + wb * cb * Ta * v * dx + Qr * v * dx

    # Resolução do sistema
    problem = LinearProblem(a, L, bcs = bcs)
    T_sol = problem.solve()

    # Atualização de T_old
    T_old.x.array[:] = T_sol.x.array[:]

# Plotagem da solução final
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