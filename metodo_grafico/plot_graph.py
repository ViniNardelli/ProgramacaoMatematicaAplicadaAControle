import numpy as np

from metodo_grafico.modelo2d import Modelo2dToPlot
from programacao_linear.utils import expression_to_str, monomial_to_str, max_digits
from programacao_linear.modelo import is_in_feasible_region

import matplotlib.pyplot as plt
from typing import Tuple


def plot_feasibility_region(model: Modelo2dToPlot, save: bool = False):
    fig, ax = plt.subplots(dpi=300, figsize=(6.4, 3.6))
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(('data', 0))
    # Hide the top and right spines.
    ax.spines[["top", "right"]].set_visible(False)

    # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
    # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
    # respectively) and the other one (1) is an axes coordinate (i.e., at the very
    # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
    # actually spills out of the axes.
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlabel(f'{[*model.variables.keys()][0]}', loc='right')
    ax.set_ylabel(f'{[*model.variables.keys()][1]}', loc='top', rotation=90)

    # Plot restrictions
    cmap = color_restrictions(model.m)
    n_a = [max_digits(column) for column in np.vstack((model.alt, model.aeq, model.agt)).T]
    n_b = max_digits(np.concatenate((model.blt, model.beq, model.bgt)).T[0])
    if any((len(model.agt) > 0, len(model.aeq) > 0, len(model.alt) > 0)):
        for i, (a, b) in enumerate(zip(np.vstack((model.agt, model.alt, model.aeq)),
                                       np.vstack((model.bgt, model.blt, model.beq)))):
            ax.axline(*line_points(*a.tolist(), -b[0]), color=cmap(i),
                      label=expression_to_str(a, model.variables, n_a).strip(' ')
                            + ' = ' + monomial_to_str(b[0], None, n_b))

    # Plot extreme points
    for i, point in enumerate(model.extreme_points, 1):
        ax.plot(*point.tolist(), 'ko')
        ax.annotate(f'e{i}', tuple(point.tolist()), fontsize=5, textcoords="offset points", xytext=(2, 3))

    # Plot curve lines
    x = np.linspace(*ax.get_xlim(), 160)
    y = np.linspace(*ax.get_ylim(), 90)
    X, Y = np.meshgrid(x, y)
    mask = [[not is_in_feasible_region(model, (xx, yy)) for xx, yy in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)]
    Z = model.c.flatten()[0] * X + model.c.flatten()[1] * Y
    Z[mask] = np.nan
    CS = ax.contour(X, Y, Z, linewidths=0.5, levels=30, cmap='Greens_r' if model._minimize else 'Greens')

    # Axes configurations
    ax.legend()
    ax.legend(prop={'size': 5})
    ax.clabel(CS, CS.levels[1::2], fontsize=3)
    ax.set_xlim(*map(lambda x: x * 1.1, model.var_range[list(model.variables)[0]]))
    ax.set_ylim(*map(lambda x: x * 1.1, model.var_range[list(model.variables)[1]]))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if save:
        fig.savefig('test.png', transparent=True)
    plt.show()


def line_points(x1: int = 0, x2: int = 0, c: int = 0) -> Tuple[Tuple[float, float]]:
    if x1 == 0 and x2 == 0:
        raise Exception("linePoints: a and b cannot both be zero")
    return tuple((-c / x1, p) if x2 == 0 else (p, (-c - x1 * p) / x2) for p in [-1., 1.])


def color_restrictions(n, name='ocean'):
    return plt.cm.get_cmap(name, n + 1)


if __name__ == '__main__':
    model = Modelo2dToPlot('100*A+150*B')
    model += '1/2 * A + 1/6 * B >= 20'
    model += '3/10 * A + 1/3 * B >= 27'
    model += '1/5 * A + 1/2 * B >= 30'
    print(model)
    plot_feasibility_region(model)
