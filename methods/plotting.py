import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime
import os
from pathlib import Path

CURR_DIR = os.getcwd()

def plot_errors(data, loc, params, vmin_vmax=None, save=False, fname='err'):
    cmap = mpl.cm.plasma
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=10)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ticks = [0,1,2,3,4]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=10)
    ax2.set_title('XZ plane')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=10)
    ax3.set_title('YZ plane')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=(ax2, ax3), orientation='vertical', label='Mean relative error')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_lognorm_sigma(data, loc, params, vmin_vmax=None, save=False, fname='lognorm_err'):
    cmap = mpl.cm.plasma
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=10)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ticks = [0,1,2,3,4]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=10)
    ax2.set_title('XZ plane')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=10)
    ax3.set_title('YZ plane')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=(ax2, ax3), orientation='vertical', label=r'$\sigma_{log},\ [log(mD)]$')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_perm(data, loc, params, vmin_vmax=None, save=False, fname='permeability'):

    data = data/1000 # Perm in Darcy

    cmap = mpl.cm.Set2_r
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0]/1000, vmax=vmin_vmax[1]/1000) # Darcy => .../1000 

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=100)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')
    
    ticks = [0,1,2,3,4]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax2.set_title('XZ plane')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax3.set_title('YZ plane')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              ax=(ax2, ax3), anchor=(0, 0.5), shrink=1, orientation='vertical', label='Permeability, D')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()

def plot_press_plan(data, loc, params, vmin_vmax=None, save=False, fname='Pore pressure'):
    # horizontal plan only 

    pore_press = data[:, :, loc[2]]

    cmap = mpl.cm.viridis
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(pore_press), vmax=np.max(pore_press))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig, ax1 = plt.subplots(figsize=(7, 3), layout="constrained")

    ax1.imshow(pore_press.transpose(), extent=[x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]], origin='lower', cmap=cmap, norm=norm) 
    # ax1.contourf(x_ax, y_ax, pore_press.transpose(), cmap=cmap, norm=norm, levels=20)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ticks = [0,1,2,3,4]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax1, orientation='vertical', label='Pore pressure, MPa')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_ev_dens_slice(data, loc, params, vmin_vmax=None, save=False, fname='seism_dens'):

    data = data/np.prod(params.dx_dy_dz) # events per m3

    cmap = mpl.cm.terrain
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=100)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ticks = [0,1,2,3,4]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax2.set_title('XZ plane')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax3.set_title('YZ plane')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=(ax2, ax3), orientation='vertical', label=r'Seismic events in $m^3$')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()
   

def plot_event_list(ev_list, params, vmin_vmax=None, save=False, fname='events in space time and mag', title=None):

    fig = plt.figure(figsize=(11, 4), layout="constrained") #, dpi=300)

    ax = fig.add_subplot(projection='3d')

    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(ev_list[:,0]), vmax=np.max(ev_list[:,0]))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
    
    cmap = mpl.cm.viridis_r

    t, x, y, d, M = [ev_list[:, ii] for ii in range(ev_list.shape[-1])]
    x_m, y_m, d_m = [dxdydz * xyz for dxdydz, xyz in zip(params.dx_dy_dz, [x, y, d])]
    x_km, y_km, d_km = [(bound[0] + xyz)/1000 for bound, xyz in zip(params.sides, [x_m, y_m, d_m])]

    scatter = ax.scatter(x_km, y_km, - d_km, marker='o', c=t, cmap=cmap, norm=norm, s=(5*M)**2, label='event')
    # ax.set_aspect('equal')
    ax.set_xlabel('x, km')
    ax.set_ylabel('y, km')
    ax.set_zlabel('Depth, km')
    ax.set_xlim(1e-3*np.array(params.sides[0]))
    ax.set_ylim(1e-3*np.array(params.sides[1]))
    ax.set_zlim(-1e-3*np.array(params.sides[2]))
    ax.invert_zaxis()

    ax.set_xticks([0,1,2,3,4])
    ax.set_yticks([0,1,2,3,4])
    ax.set_zticks([-1, -1.5, -2])
    if title:
        ax.set_title(title)

    # produce a legend with a cross-section of sizes from the scatter
    m_handles = [
        mpl.lines.Line2D([], [], color=cmap(norm(60)), marker='o', linestyle='None', markersize=5*1, alpha=0.6, label='1'), # markersize=sqrt(s) from scatter
        mpl.lines.Line2D([], [], color=cmap(norm(60)), marker='o', linestyle='None', markersize=5*2, alpha=0.6, label='2'),
        mpl.lines.Line2D([], [], color=cmap(norm(60)), marker='o', linestyle='None', markersize=5*3, alpha=0.6, label='3'),
        ]

    m_legend = fig.legend(handles = m_handles, loc=(0.63, 0.6), title= "M")
    fig.colorbar(scatter, location='bottom', label='Time, h',  shrink=0.2)

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_masked_perm(data, mask, loc, params, vmin_vmax=None, save=False, fname='masked permeability'):

    data = data/1000 # Perm in Darcy
    data[mask==0] = np.nan # masking

    cmap = mpl.cm.Set2_r
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0]/1000, vmax=vmin_vmax[1]/1000) # Darcy => .../1000 
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=100, corner_mask=True)
    ax1.contour(x_ax, y_ax, mask[:, :, loc[2]].transpose(), colors='k', levels=[0.999,], linewidths=0.5) # contour for mask
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')
    ax1.set_xlim([1.2, 2.8])
    ax1.set_ylim([1.2, 2.8])

    ticks = [1.5, 2, 2.5]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    
    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=100, corner_mask=True)
    ax2.contour(x_ax, -z_ax, mask[:, loc[1], :].transpose(), colors='k', levels=[0.999,], linewidths=0.5)
    ax2.set_title('XZ plane')
    # ax2.set_aspect('equal')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')
    ax2.set_xlim([1, 3])
    ax2.set_ylim([-1.8, -1.15])

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=100, corner_mask=True)
    ax3.contour(y_ax, -z_ax, mask[loc[0], :, :].transpose(), colors='k', levels=[0.999,], linewidths=0.5)
    ax3.set_title('YZ plane')
    # ax3.set_aspect('equal')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')
    ax3.set_xlim([1, 3])
    ax3.set_ylim([-1.8, -1.15])

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              ax=(ax2, ax3), anchor=(0, 0.5), shrink=1, orientation='vertical', label='Permeability, D')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_perm_with_contour(data, mask, loc, params, vmin_vmax=None, save=False, fname='masked permeability'):

    data = data/1000 # Perm in Darcy

    cmap = mpl.cm.Set2_r
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0]/1000, vmax=vmin_vmax[1]/1000) # Darcy => .../1000 

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=100)
    ax1.contour(x_ax, y_ax, mask[:, :, loc[2]].transpose(), colors='k', levels=[0.999,]) # contour for mask
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ticks = [0,1,2,3,4]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax2.contour(x_ax, -z_ax, mask[:, loc[1], :].transpose(), colors='k', levels=[0.999,])
    ax2.set_title('XZ plane')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax3.contour(y_ax, -z_ax, mask[loc[0], :, :].transpose(), colors='k', levels=[0.999,])
    ax3.set_title('YZ plane')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              ax=(ax2, ax3), anchor=(0, 0.5), shrink=1, orientation='vertical', label='Permeability, D')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()

def plot_loss_mask(data, loc, params, vmin_vmax=None, save=False, fname='loss mask'):
    cmap = mpl.cm.cool
    if vmin_vmax is None:
        # norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
        norm = mpl.colors.LogNorm(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=[0.001, 0.01, 0.1, 0.5, 1])
    # ax1.imshow(data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ticks = [0,1,2,3,4]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=[0.001, 0.01, 0.1, 0.5, 1])
    ax2.set_title('XZ plane')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=[0.001, 0.01, 0.1, 0.5, 1])
    ax3.set_title('YZ plane')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=(ax2, ax3), orientation='vertical', label='Loss mutiplicator')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()

def plot_error_vs_event_num(evs, errs):
    x = evs.flatten()
    y = errs.flatten()

    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(x, y, marker='+')
    ax.set_xscale('log')
    ax.set_ylim([0, 0.5])
    ax.set_xlim([1e-1, 100])

    coef = np.polyfit(np.log(x[x>2e-1]), y[x>2e-1], 1)
    poly1d_fn = np.poly1d(coef) 
    xx = np.array([0.1, 100])
    plt.plot(xx, poly1d_fn(np.log(xx)), c='r', linestyle=':', linewidth=2)

    ax.set_xlabel('Number of events in a cell')
    ax.set_ylabel(r'$\sigma_{log},\ [log(mD)]$')


def event_pics_for_gif(ev, params):
    np.random.seed(42)
    Path(f'{CURR_DIR}/pics/').mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    
    max_time = int(np.max(ev[:,0])) # last event
            
    fig = plt.figure(figsize=(11, 4), layout="constrained") #, dpi=300)
    ax = fig.add_subplot(projection='3d')
    norm = mpl.colors.Normalize(0, max_time)
    cmap = mpl.cm.viridis_r

    ax.set_xlabel('x, km')
    ax.set_ylabel('y, km')
    ax.set_zlabel('Depth, km')
    ax.set_xlim(1e-3*np.array(params.sides[0]))
    ax.set_ylim(1e-3*np.array(params.sides[1]))
    ax.set_zlim(-1e-3*np.array(params.sides[2]))
    ax.invert_zaxis()

    ax.set_xticks([0,1,2,3,4])
    ax.set_yticks([0,1,2,3,4])
    ax.set_zticks([-1, -1.5, -2])
    
    # produce a legend with a cross-section of sizes from the scatter
    m_handles = [
        mpl.lines.Line2D([], [], color=cmap(norm(60)), marker='o', linestyle='None', markersize=5*1, alpha=0.6, label='1'), # markersize=sqrt(s) from scatter
        mpl.lines.Line2D([], [], color=cmap(norm(60)), marker='o', linestyle='None', markersize=5*2, alpha=0.6, label='2'),
        mpl.lines.Line2D([], [], color=cmap(norm(60)), marker='o', linestyle='None', markersize=5*3, alpha=0.6, label='3'),
        ]
    fig.legend(handles = m_handles, loc=(0.63, 0.6), title= "M")

    sc = ax.scatter([0],[0],[0], c=max_time, cmap=cmap, norm=norm, s=0,)
    plt.colorbar(sc, location='bottom', label='Time, h',  shrink=0.2)

    t, x, y, d, M = [ev[:, ii] for ii in range(ev.shape[-1])]
    x_m, y_m, d_m = [dxdydz * xyz for dxdydz, xyz in zip(params.dx_dy_dz, [x, y, d])]
    x_km, y_km, d_km = [(bound[0] + xyz)/1000 for bound, xyz in zip(params.sides, [x_m, y_m, d_m])]
    
    
    for hh in range(max_time):
        title = f'Time = {hh} h'
        step = t.astype('int') == hh
        if True in step:
            ax.scatter(x_km[step], y_km[step], - d_km[step], marker='o', c=t[step], cmap=cmap, norm=norm, s=(5*M[step])**2, label='event')
        
        ax.set_title(title)
        
        fname = f'{CURR_DIR}/pics/h_{hh}_{timestamp}'        
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)