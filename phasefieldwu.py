# phase-field crack propagation in L-shaped panel using FEniCS
from subprocess import *
from dolfin import *
import numpy as np
import sympy as sp
from ufl import replace

# set_log_level(PROGRESS)
parameters["form_compiler"]["quadrature_degree"]=1
parameters["form_compiler"]["cpp_optimize"] = True

snes_solver_parameters_bounds = {"nonlinear_solver": "snes",
#                          "reset_jacobian":True,
                          "symmetric": True,
                          "snes_solver": {"maximum_iterations": 100,
                                          "linear_solver": "umfpack",
                                          "preconditioner": "bjacobi",
                                          "report": False,
                                          "line_search": "basic",
                                          "method":"vinewtonrsls",
                                          "absolute_tolerance":1e-8,
                                          "relative_tolerance":1e-8,
                                          "krylov_solver": {"absolute_tolerance":1e-10,
                                                            "relative_tolerance":1e-10}}}


# create results folder
call(['rm','-rf','./results'])
call(['mkdir','results'])

# read mesh from file
mesh=Mesh('LSP.xml')

#plot(mesh,title='finite element mesh',window_width=800,window_height=500).write_png('./results/mesh')

dx = dx(metadata={'quadrature_degree': 1}) # quadrature order


# material
xe=2.585e4
xnu=.18
xft=2.7
xgf=.09
xl=5.

# cornelissen
p=2.
k0=-1.3546*xft**2./xgf
xwc=5.1361*xgf/xft


lch=xe*xgf/xft**2.
a1=4./pi*lch/xl
a2=2.*(-2.*k0*xgf/xft**2.)**(2./3.)-(p+.5)
a3=0.
if p==2.: a3=1./a2*(.125*(xwc*xft/xgf)**2.-(1.+a2))

xlambda=xe*xnu/(1.+xnu)/(1.-2.*xnu)
xmu=xe/2./(1.+xnu)


# Define constitutive functions
# strain tensor
def eps(u):
    return sym(grad(u))

# stress tensor
def sig0(u):
    return xlambda*tr(eps(u))*Identity(2) + 2*xmu*eps(u)


# driving function
# plane strain
def y(u):
    s1=(sig0(u)[0,0]+sig0(u)[1,1])/2.\
        + sqrt(((sig0(u)[0,0]-sig0(u)[1,1])/2.)**2+sig0(u)[0,1]**2)
    s2=(sig0(u)[0,0]+sig0(u)[1,1])/2.\
        - sqrt(((sig0(u)[0,0]-sig0(u)[1,1])/2.)**2+sig0(u)[0,1]**2)
    s3=xlambda/2./(xlambda+xmu)*(s1+s2)
    smax=conditional(gt(s1,s2),s1,s2)
    betac=9.    
    J2bar=.5*(s1**2+s2**2+s3**2-(s1+s2+s3)**2/3.)
    sigeq=(betac*(smax+abs(smax))/2. \
           + sqrt(3.*J2bar))/(1.+betac)
    return sigeq**2/2./xe


# degradation function
def w(d):
    R=(1.0-d)**p
    Q=a1*d+a1*a2*d**2.0+a1*a2*a3*d**3.0
    return R/(R+Q)

def wp(d):
    R=(1.-d)**p
    Rp=-p*(1.-d)**(p-1.)
    Q=a1*d+a1*a2*d**2.+a1*a2*a3*d**3.
    Qp=a1+2.*a1*a2*d+3.*a1*a2*a3*d**2.
    return (Q*Rp-R*Qp)/(R+Q)**2.

# max principal stress
def prstr(u,d):
    s1=(sig0(u)[0,0]+sig0(u)[1,1])/2.\
        + sqrt(((sig0(u)[0,0]-sig0(u)[1,1])/2.)**2+sig0(u)[0,1]**2)
    s2=(sig0(u)[0,0]+sig0(u)[1,1])/2.\
        - sqrt(((sig0(u)[0,0]-sig0(u)[1,1])/2.)**2+sig0(u)[0,1]**2)
    smax=conditional(gt(s1,s2),s1,s2)
    return w(d)*smax


# Define function space for displacemnet
U = VectorFunctionSpace(mesh, 'CG', 1)
# Define function space for damage
D = FunctionSpace(mesh,'CG',1)

# boundary conditions
def support(x,on_boundary):
    return near(x[1],-250., 1.e-10)
bc1 = DirichletBC(U, Constant((0.,0.)), support)


def load(x):
    return near(x[0],220.,1.e-10) and near(x[1],0.,1.e-10)
uy = Expression('uy',degree=1, uy=0.)
bc2 = DirichletBC(U.sub(1), uy, load,method="pointwise")

bc_u=[bc1,bc2]

# Define test and trial functions
du= TestFunction(U)
utr=TrialFunction(U)
dd= TestFunction(D)
dtr=TrialFunction(D)

# Define functions 
u = Function(U)
d = Function(D)
E_u=inner(w(d)*sig0(u),eps(du))*dx
E_u_u = derivative(E_u,u,utr)
E_d = (2.*xgf/pi/xl*((1.-d)*dd+xl**2*dot(grad(d), grad(dd)))+wp(d)*y(u)*dd)*dx
E_d_d = derivative(E_d,d,dtr)

E_utr=replace(E_u,{u:utr})
E_dtr=replace(E_d,{d:dtr})

problem_u     = LinearVariationalProblem(lhs(E_utr), rhs(E_utr), u, bc_u)
problem_d_nl = NonlinearVariationalProblem(E_d, d, [], E_d_d)

# lower and upper bounds
lb = interpolate(Constant(0.), D)
ub = interpolate(Constant(1.), D)
#problem_d_nl.set_bounds(lb,ub)
solver_u  = LinearVariationalSolver(problem_u)
solver_d  = NonlinearVariationalSolver(problem_d_nl)     
solver_d.parameters.update(snes_solver_parameters_bounds)

# Create VTK files for visualization output
vtkfile_d = File('./results/damage.pvd')
vtkfile_s = File('./results/stress.pvd')

X=U.tabulate_dof_coordinates()
X.resize((U.dim(), 2))    

disps=[]
loads=[]


d0=interpolate(Constant(0.),D)
# stepping
for u1 in np.arange(0.,1.,1.e-2):

    # increment top edge displacement
    uy.uy=u1

    iter=1; err_d=1.

    while err_d>1.e-8 and iter<200:

        # solve elasticity problem
        solver_u.solve()
        # solve damage evolution
        problem_d_nl.set_bounds(lb.vector(),ub.vector())
        solver_d  = NonlinearVariationalSolver(problem_d_nl)
        solver_d.parameters.update(snes_solver_parameters_bounds)
        solver_d.solve()    

        d_diff = d.vector().get_local() - d0.vector().get_local()
        err_d= np.linalg.norm(d_diff, ord=np.Inf)

        d0.assign(d)
        iter+=1
        
    # update lower bounds
    lb.assign(d)

    # Save solution to file (VTK)
    vtkfile_d << (d, abs(u1))
    


    stress=project(prstr(u,d),D)
    vtkfile_s << (stress, abs(u1))

    f=assemble(E_u_u)*u.vector()
    
    # load-displacement 
    load,disp = 0., 0.
    for i, (xi,ui,fi) in enumerate(zip(X,u.vector(), f)):
        #if near(xi[1],100.,1.e-10) and (near(xi[0],-2.5,1.e-10) or near(xi[0],2.5,1.e-10)):
        if near(xi[1],0.,1.e-10) and near(xi[0],220.,1.e-10):
            if ui!=0.:
                disp=abs(ui)
                load+=fi*.1 # note: *100 for thickness /1000 in kN
    o=open('./results/output','a')
    o.write(str(disp)+' '+str(abs(load))+'\n')
    o.close()

