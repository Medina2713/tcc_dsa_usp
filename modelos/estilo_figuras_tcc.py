"""
Estilo padrao para figuras academicas (TCC).
- Fundo branco
- Sem moldura (spines superior e direito removidos)
- Sem grid
- Fontes maiores nos eixos
"""

import matplotlib.pyplot as plt


def aplicar_estilo_tcc(ax, fig=None, fonte_eixos=14, fonte_ticks=12):
    """
    Aplica estilo academico TCC ao eixo.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo a estilizar
    fig : matplotlib.figure.Figure, optional
        Figura (para fundo branco). Se None, usa ax.figure
    fonte_eixos : int
        Tamanho da fonte dos rotulos dos eixos (padrao 14)
    fonte_ticks : int
        Tamanho da fonte dos ticks (padrao 12)
    """
    if fig is None:
        fig = ax.figure
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    # Remove moldura (spines superior e direito)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_fontsize(fonte_eixos)
    ax.yaxis.label.set_fontsize(fonte_eixos)
    ax.tick_params(axis='both', labelsize=fonte_ticks)


def configurar_savefig_tcc(fig, path, dpi=300):
    """Salva figura com fundo branco, sem bordas extras."""
    fig.patch.set_facecolor('white')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
