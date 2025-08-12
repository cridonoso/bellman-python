import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os


def plot_deterministic(solution, fig=None, save=None):
    if fig is None: 
        fig = plt.figure(figsize=(7, 6), dpi=300) # Ajustamos un poco el tamaño para mejor visualización
    
    
    w_grid = solution['states']['W']
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25)
    fig.suptitle('Análisis del Problema de Consumo de la Torta', fontsize=16)

    # 2. Asignar cada gráfico a una celda (o celdas) de la grilla
    ax_cpol = fig.add_subplot(gs[0, 0])
    ax_wpol = fig.add_subplot(gs[0, 1])
    ax_vfi = fig.add_subplot(gs[1, :])

    # === Gráfico en (0, 0): Política de Consumo (cpol) ===
    ax_cpol.plot(w_grid, solution['cpol'], color='darkblue', label="c(W)")
    ax_cpol.plot(w_grid, (1-0.98**2)*w_grid, linestyle=':', color='darkred', linewidth=3, label='Teórico')
    ax_cpol.set_title('Función de Política de Consumo')
    ax_cpol.set_xlabel('Nivel de Riqueza (W)')
    ax_cpol.set_ylabel('Consumo Óptimo (c)')
    ax_cpol.legend()
    ax_cpol.grid(True, linestyle='--', alpha=0.6)

    ax_wpol.plot(w_grid, solution['wpol'], color='darkgreen', label="W'")
    ax_wpol.plot(w_grid, 0.98**2*w_grid, linestyle=':', color='darkred', linewidth=3, label='Teórico')
    ax_wpol.plot(w_grid, w_grid, color='k', linestyle='--', linewidth=1, label="45°")

    ax_wpol.set_title("Función de Política de Riqueza")
    ax_wpol.set_xlabel('Nivel de Riqueza (W)')
    ax_wpol.set_ylabel("Riqueza Siguiente Período (W')")
    ax_wpol.legend()
    ax_wpol.grid(True, linestyle='--', alpha=0.6)

    values_v = solution['history']['V']
    n_val = len(values_v)
    cmap = plt.get_cmap('viridis_r')
    norm = colors.PowerNorm(gamma=0.5, vmin=0., vmax=n_val - 1)

    for k, v in enumerate(values_v):
        color = cmap(norm(k))
        ax_vfi.plot(w_grid, v[0], color=color, linewidth=1)

    ax_vfi.plot(w_grid, 1/np.sqrt(1 - 0.98**2)*np.sqrt(w_grid), linestyle=':', color='darkred', linewidth=3, label='Teórico')
    ax_vfi.set_title('Convergencia de la Función de Valor')
    ax_vfi.set_xlabel('Nivel de Riqueza (W)')
    ax_vfi.set_ylabel('Función de Valor V(W)')
    ax_vfi.legend()
    ax_vfi.grid(True, linestyle='--', alpha=0.6)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax_vfi.inset_axes([1.02, 0.05, 0.03, 0.9])

    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Número de Iteración', rotation=270, labelpad=15)

    if save is not None:
        if os.path.dirname(save) != '':
            os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, bbox_inches='tight', dpi=300)

    return fig
 

def plot_stochastic(solution, fig=None, axes=None, save=None):
    if fig is None or axes is None: 
        fig, axes = plt.subplots(2, 2, 
                         figsize=(8, 6), 
                         gridspec_kw={'hspace':0.4, 'wspace':0.3},
                         dpi=300)
    
    epsilon_grid = solution['states']['epsilon']
    W_grid = solution['states']['W']


    # fig.suptitle('Análisis del Problema de Consumo de la Torta\n Estocastico', fontsize=16)
    label_tag = [r'$\epsilon=0.9$', r'$\epsilon=1.1$']
    colorsx= ['darkgreen', 'darkblue']
    # === Gráfico en (0, 0): Política de Consumo (cpol) ===
    ax_cpol = axes[0, 0]
    for i in range(len(epsilon_grid)):
        ax_cpol.plot(W_grid, solution['cpol'][i, :], label=label_tag[i], color=colorsx[i])

    ax_cpol.set_title('Función de Política de Consumo')
    ax_cpol.set_xlabel('Riqueza (W)')
    ax_cpol.set_ylabel('Consumo Óptimo (c)')
    ax_cpol.legend()
    ax_cpol.grid(True, linestyle='--', alpha=0.6)


    # === Gráfico en (0, 1): Política de Riqueza (wpol) ===
    ax_wpol = axes[0, 1]
    for i, sc in enumerate(solution['wpol']):
        ax_wpol.plot(W_grid, sc, color=colorsx[i], label=label_tag[i])
    ax_wpol.plot(W_grid, W_grid, color='k', linestyle='--', linewidth=1, label="45°")

    ax_wpol.set_title("Función de Política de Riqueza")
    ax_wpol.set_xlabel('Riqueza Hoy (W)')
    ax_wpol.set_ylabel("Riqueza Mañana (W')")
    ax_wpol.legend(loc='lower right')
    ax_wpol.grid(True, linestyle='--', alpha=0.6)


    # === Gráfico en Fila 1: Iteración de la Función de Valor ===
    v_shock_0, v_shock_1 = [], []
    for iter_v in solution['history']['V']:
        v_shock_0.append(iter_v[0])
        v_shock_1.append(iter_v[1])


    n_val = len(v_shock_0)
    cmap = plt.get_cmap('viridis_r')
    norm = colors.PowerNorm(gamma=0.5, vmin=0., vmax=n_val - 1)
    ax_vfi = axes[1, 0]
    ax_vfi.set_title(r'Función de Valor ($\epsilon = 0.9)$')

    for k, v in enumerate(v_shock_0):
        color = cmap(norm(k))
        ax_vfi.plot(W_grid, v, color=color, linewidth=1)
        
        ax_vfi.set_xlabel('Nivel de Riqueza (W)')
        ax_vfi.set_ylabel('Función de Valor V(W)')
        ax_vfi.legend()
        ax_vfi.grid(True, linestyle='--', alpha=0.6)
    # ax_vfi.plot(W_grid, v_shock_0[-1], linestyle=':', color='darkred', linewidth=3, label='Teórico')
    ax_vfi.legend()

    n_val = len(v_shock_1)
    ax_vfi = axes[1, 1]
    ax_vfi.set_title(r'Función de Valor ($\epsilon = 1.1)$')
    axes[1, 0].sharey(axes[1, 1])
    for k, v in enumerate(v_shock_1):
        color = cmap(norm(k))
        ax_vfi.plot(W_grid, v, color=color, linewidth=1)
        ax_vfi.set_xlabel('Nivel de Riqueza (W)')
        ax_vfi.set_ylabel('Función de Valor V(W)')

        ax_vfi.grid(True, linestyle='--', alpha=0.6)
    # ax_vfi.legend()

    # Crear el ScalarMappable que se vinculará al colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax_vfi.inset_axes([1.02, 0.05, 0.03, 0.9])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Número de Iteración', rotation=270, labelpad=15) 

    if save is not None:
        if os.path.dirname(save) != '':
            os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, bbox_inches='tight', dpi=300)