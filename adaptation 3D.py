import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

nx, ny, nz = 50, 50, 50
dx = 1.0
x = np.linspace(0, nx * dx, nx)
y = np.linspace(0, ny * dx, ny)
z = np.linspace(0, nz * dx, nz)
X, Y, Z = np.meshgrid(x, y, z)

# Sample primitive variables
rho = np.sin(X) + np.cos(Y) + np.exp(-Z)
vx = np.cos(X) * np.sin(Y)
vy = np.sin(X) * np.cos(Y)
vz = np.exp(-Z)
P = np.sin(X) * np.exp(-Y) + np.cos(Z)
def Conserved(rho, vx, vy, vz, P, gamma, V):
    M   = rho * V
    px  = rho * vx * V
    py  = rho * vy * V
    pz  = rho * vz * V
    E   = ( (P / (gamma - 1)) + (0.5 * rho * (vx**2 + vy**2 +vz**2)) ) * V
    return M , px, py, pz, E

def Primitive(M, px, py, pz, E, gamma, V):
    rho = M / V
    vx  = px / (rho * V)
    vy  = py / (rho * V)
    vz  = pz / (rho * V)
    P   = ( (E/V) - (0.5 * rho * (vx**2 + vy**2 + vz**2)) ) * (gamma - 1)
    return rho, vx, vy, vz, P

def Gradient(f, dx):
    R = -1
    L = 1

    fdx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2 * dx)
    fdy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2 * dx)
    fdz = (np.roll(f, R, axis=2) - np.roll(f, L, axis=2)) / (2 * dx)

    return fdx, fdy, fdz


def SlopeLimit(f, dx, fdx, fdy, fdz):
    R = -1
    L = 1

    fdx = np.maximum(0., np.minimum(1., ((f - np.roll(f, L, axis=0)) / dx) / (fdx + 1.08e-8 * (fdx == 0)))) * fdx
    fdx = np.maximum(0., np.minimum(1., (-(f - np.roll(f, R, axis=0)) / dx) / (fdx + 1.08e-8 * (fdx == 0)))) * fdx
    fdy = np.maximum(0., np.minimum(1., ((f - np.roll(f, L, axis=1)) / dx) / (fdy + 1.08e-8 * (fdy == 0)))) * fdy
    fdy = np.maximum(0., np.minimum(1., (-(f - np.roll(f, R, axis=1)) / dx) / (fdy + 1.08e-8 * (fdy == 0)))) * fdy
    fdz = np.maximum(0., np.minimum(1., ((f - np.roll(f, L, axis=2)) / dx) / (fdz + 1.08e-8 * (fdz == 0)))) * fdz
    fdz = np.maximum(0., np.minimum(1., (-(f - np.roll(f, R, axis=2)) / dx) / (fdz + 1.08e-8 * (fdz == 0)))) * fdz

    return fdx, fdy, fdz


def Extrapolation(f, dx, fdx, fdy, fdz):
	# directions for np.roll() 
    R = -1   # right
    L = 1    # left
    
    fXL = f - (fdx * (dx / 2))
    fXL = np.roll(fXL, R, axis = 0)
    fXR = f + (fdx * (dx / 2))
    
    fYL = f - (fdy * (dx / 2))
    fYL = np.roll(fYL, R, axis = 1)
    fYR = f + (fdy * (dx / 2))
    
    fZL = f - (fdz * (dx / 2))
    fZL = np.roll(fZL, R, axis = 2)
    fZR = f + (fdz * (dx / 2))   
    return fXL, fXR, fYL, fYR, fZL, fZR 

def Flux(rhoL, rhoR, vxL, vxR, vyL, vyR, vzL, vzR, PLs, PRs, gamma):

    # left and right energies    
    EnL = (PLs / (gamma - 1) ) + ( 0.5 * rhoL * (vxL**2 + vyL**2 + vzL **2) )
    EnR = (PRs / (gamma - 1) ) + ( 0.5 * rhoR * (vxR**2 + vyR**2 + vzR **2) )

    # compute star (averaged) states
    # Check for negative densities and pressures
    
    rhoStar = 0.5 * (rhoL + rhoR)
    pxStar  = 0.5 * ( (rhoL *vxL) + (rhoR * vxR) )
    pyStar  = 0.5 * ( (rhoL *vyL) + (rhoR * vyR) )
    pzStar  = 0.5 * ( (rhoL *vzL) + (rhoR * vzR) )

    EStar   = 0.5 * (EnL + EnR)
    PStar   = (gamma - 1) * (EStar - (0.5 * (pxStar**2 + pyStar**2 + pzStar**2) / rhoStar))
    # Check for negative densities and pressures
    if np.any(np.isnan([rhoL, rhoR, PLs, PRs, vxL, vxR, vyL, vyR, vzL, vzR, EnL, EnR])):
        return np.zeros_like(rhoL), np.zeros_like(rhoL), np.zeros_like(rhoL), np.zeros_like(rhoL), np.zeros_like(rhoL)


    # compute fluxes (local Lax-Friedrichs/Rusanov)
    fluxM   = pxStar
    fluxpx  = (pxStar**2 / rhoStar) + PStar
    fluxpy  = (pxStar * pyStar) /rhoStar
    fluxpz  = (pxStar * pzStar) /rhoStar
    fluxE   = (EStar + PStar) * pxStar / rhoStar

    # find wavespeeds
    CLs = np.sqrt(np.maximum(0, gamma * np.maximum(0, PLs) / np.maximum(1e-10, rhoL))) + np.abs(vxL)
    CRs = np.sqrt(np.maximum(0, gamma * np.maximum(0, PRs) / np.maximum(1e-10, rhoR))) + np.abs(vxR)

    C           = np.maximum(CLs, CRs)

    # add stabilizing diffusive term
    fluxM   -= C * 0.5 * (rhoL - rhoR)
    fluxpx  -= C * 0.5 * (rhoL * vxL - rhoR * vxR)  
    fluxpy  -= C * 0.5 * (rhoL * vyL - rhoR * vyR)
    fluxpz  -= C * 0.5 * (rhoL * vzL - rhoR * vzR)
    fluxE   -= C * 0.5 * (EnL - EnR)

    return fluxM, fluxpx, fluxpy, fluxpz, fluxE


def ApplyFlux(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
    # directions for np.roll()
    R = -1   # right
    L = 1    # left

    # update solution
    F += - dt * dx * flux_F_X
    F +=   dt * dx * np.roll(flux_F_X,L,axis=0)
    F += - dt * dx * flux_F_Y
    F +=   dt * dx * np.roll(flux_F_Y,L,axis=1)
    F += - dt * dx * flux_F_Z
    F +=   dt * dx * np.roll(flux_F_Z,L,axis=2)

    return F
def main_3d():
    # Simulation parameters
    N = 32  # resolution
    boxsize = 1.0
    gamma = 5/3
    courant_fac = 0.4
    t = 0
    tEnd = 2
    tOut = 0.02
    outputCount=1
    useSlopeLimiting = True

    # Mesh
    dx = boxsize / N
    vol = dx**3
    xlin = np.linspace(0.5*dx, boxsize-0.5*dx, N)
    Y, X, Z = np.meshgrid(xlin, xlin, xlin)

    # Generate Initial Conditions
    w0 = 0.1
    sigma = 0.05/np.sqrt(2.)
    rho = 1. + (np.abs(Y-0.5) < 0.25)
    vx = -0.5 + (np.abs(Y-0.5) < 0.25)
    vy = w0 * np.sin(4*np.pi*X) * (np.exp(-(Y-0.25)**2/(2 * sigma**2)) + np.exp(-(Y-0.75)**2/(2*sigma**2)))
    vz = w0 * np.sin(4*np.pi*X) * (np.exp(-(Z-0.25)**2/(2 * sigma**2)) + np.exp(-(Z-0.75)**2/(2*sigma**2)))
    P = 2.5 * np.ones(X.shape)



    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy = Conserved(rho, vx, vy, vz, P, gamma, vol )
    # Prep figure for 3D visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D plot
    surface=ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=rho.flatten(), cmap='viridis')


    # Simulation Main Loop
    while t <= tEnd:
        rho, vx, vy, vz, P = Primitive(Mass, Momx, Momy, Momz, Energy, gamma, vol )
		
		# get time step (CFL) = dx / max signal speed
        speed_squared = np.maximum(0, vx**2 + vy**2 + vz**2)
        dt = 0.001
        plotThisTurn = False
        if t + dt > outputCount*tOut:
            dt = outputCount*tOut - t
            plotThisTurn = True
        # Convert back to primitive variables
        rho_dx, rho_dy, rho_dz = Gradient(rho, dx)
        vx_dx,  vx_dy, vx_dz  = Gradient(vx,  dx)
        vy_dx,  vy_dy, vy_dz  = Gradient(vy,  dx)
        vz_dx,  vz_dy, vz_dz  = Gradient(vz,  dx)
        P_dx,   P_dy, P_dz   = Gradient(P,   dx)
        if useSlopeLimiting:
            rho_dx, rho_dy, rho_dz = SlopeLimit(rho, dx, rho_dx, rho_dy, rho_dz)
            vx_dx,  vx_dy, vx_dz  = SlopeLimit(vx , dx, vx_dx,  vx_dy, vx_dz )
            vy_dx,  vy_dy, vy_dz  = SlopeLimit(vy , dx, vy_dx,  vy_dy, vy_dz )
            vz_dx,  vz_dy, vz_dz  = SlopeLimit(vz , dx, vz_dx,  vz_dy, vz_dz )   
            P_dx,   P_dy, P_dz   = SlopeLimit(P  , dx, P_dx,   P_dy, P_dz  )
        # extrapolate half-step in time
        rho_prime = rho - 0.5*dt * ( vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy+vz * rho_dz + rho * vz_dz)
        vx_prime  = vx  - 0.5*dt * ( vx * vx_dx + vy * vx_dy+ vz * vx_dz + (1/rho) * P_dx )
        vy_prime  = vy  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + vz*vy_dz +(1/rho) * P_dy )
        vz_prime  = vz  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + vz*vz_dz +(1/rho) * P_dz )
        P_prime   = P   - 0.5*dt * ( gamma*P * (vx_dx + vy_dy+vz_dz)  + vx * P_dx + vy * P_dy+ vz * P_dz )
		
		# extrapolate in space to face centers
        rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = Extrapolation(rho_prime,dx, rho_dx, rho_dy, rho_dz )
        vx_XL,  vx_XR,  vx_YL,  vx_YR, vx_ZL, vx_ZR  = Extrapolation(vx_prime,dx,  vx_dx,  vx_dy,  vx_dz)
        vy_XL,  vy_XR,  vy_YL,  vy_YR, vy_ZL, vy_ZR  = Extrapolation(vy_prime,dx,  vy_dx,  vy_dy,  vy_dz)
        vz_XL,  vz_XR,  vz_YL,  vz_YR, vz_ZL, vz_ZR  = Extrapolation(vz_prime,dx,  vz_dx,  vz_dy,  vz_dz)
        P_XL,   P_XR,   P_YL,   P_YR, P_ZL, P_ZR   = Extrapolation(P_prime,dx,   P_dx,   P_dy,   P_dz)
		
		# compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X, flux_Energy_X = Flux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, vz_XL, vz_XR, P_XL, P_XR, gamma)
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Momz_Y, flux_Energy_Y = Flux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, vz_YL, vz_YR, P_YL, P_YR, gamma)
        flux_Mass_Z, flux_Momy_Z, flux_Momx_Z, flux_Momz_Z, flux_Energy_Z = Flux(rho_ZL, rho_ZR, vy_ZL, vy_ZR, vx_ZL, vx_ZR, vz_ZL, vz_ZR, P_ZL, P_ZR, gamma)
        # update solution
        Mass   = ApplyFlux(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
        Momx   = ApplyFlux(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
        Momy   = ApplyFlux(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
        Momz   = ApplyFlux(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)
        Energy = ApplyFlux(Energy, flux_Energy_X, flux_Energy_Y, flux_Energy_Z, dx, dt)
        
        t+=dt


        # Update the facecolors for the density field
        surface.set_array(rho.flatten())
        surface.autoscale()

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'Time: {t:.2f}')

        # Display the plot in real-time
        plt.pause(0.0000001)
        print(f"Time: {t:.4f}, dt: {dt:.6e}")
        if plotThisTurn:
            outputCount += 1
    plt.show()
if __name__ == "__main__":
    main_3d()