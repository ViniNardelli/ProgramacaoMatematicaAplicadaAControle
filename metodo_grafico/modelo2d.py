from __future__ import annotations

import numpy as np

from programacao_linear.modelo import Modelo
from typing import Tuple, Dict, Iterable
from sympy import symbols


class Modelo2dToPlot(Modelo):
    def __init__(self,
                 c: str,
                 variables_name: Tuple[str, str] = None,
                 minimize: bool = True,
                 boundaries: Dict[str, Tuple[float, float]] = None):
        if variables_name is not None:
            self.variables = {v: symbols(v) for v in variables_name}
        super().__init__(c, minimize, boundaries)

        if self.n != 2:
            raise ValueError('')

        self.__a_domain_str = 'A \u2208 \u211D\u207D\u00B9\u02E3\u00B2\u207E'
        self.__x_domain_str = 'x \u2208 \u211D\u207D\u00B2\u02E3\u00B9\u207E'

    @property
    def var_range(self) -> Dict[str, Tuple[np.ndarray]]:
        points = get_extreme_points(self)
        return {label: (np.min(points.T[i]), np.max(points.T[i])) for i, label in enumerate(self.variables.keys())}

    @classmethod
    def from_modelo(cls, model: Modelo, dimensions_name: Tuple[str, str]) -> Modelo2dToPlot:
        new_model = cls(''.join((f'{k:+} * {v}' for k, v in zip(model.c.flatten().tolist(), model.variables))),
                        dimensions_name,
                        model._minimize)
        for a_array, b, sign in zip(np.vstack((model.alt, model.aeq, model.agt)),
                              np.vstack((model.blt, model.beq, model.bgt)).flatten(),
                              ['<='] * len(model.alt) + ['='] * len(model.aeq) +['>='] * len(model.agt)):
            new_model += ''.join((f'{a:+} * {v}' for a, v in zip(a_array, model.variables))) + sign + f'{b}'
        return new_model


if __name__ == '__main__':
    from metodo_grafico.plot_graph import plot_feasibility_region
    model = Modelo('100*A+150*B+100*C')
    model += '1/2 * A + 1/6 * B + 1/3 * C >= 20'
    model += '3/10 * A + 1/3 * B + 2/3 * C >= 27'
    model += '1/5 * A + 1/2 * B  + 1/2 * C>= 30'
    print(model)
    model = Modelo2dToPlot.from_modelo(model, ('A', 'B'))
    print(model)
    plot_feasibility_region(model)
