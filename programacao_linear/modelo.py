from __future__ import annotations

from itertools import combinations

from numpy.linalg import solve
from scipy.optimize import LinearConstraint, milp
from sympy.parsing.sympy_parser import parse_expr
from programacao_linear.utils import expression_to_str, monomial_to_str, max_digits, is_polynomial, is_number
from typing import Tuple, Dict, Type, Iterable

import numpy as np


class Modelo:
    r"""
    .. math::
        \min_x \ & c^T x \\
        \mbox{such that} & A_{lt} x \leq b_{lt} \\
                         & A_{gt} x \geq b_{gt} \\
                         & A_{eq} x = b_{eq} \\
                         & l \leq x \leq u
    .. math::
        \mbox{where} &x \ \mbox{is a vector of decision variables};\\
                     &c, \ b_{gt}, \ b_{lt}, \ b_{eq}, \ l \ \mbox{and} \ u \ \mbox{are vectors} \ (n \times 1);\\
                     &A_{gt}, \ A_{lt} \ \mbox{and} \ A_{eq} \ \mbox{are matrices} \ (m \times n)."""
    def __init__(self,
                 c: str,
                 minimize: bool = True,
                 boundaries: Dict[str, Tuple[float, float]] = None,
                 var_type: Dict[str, Type] = None) -> None:
        self._minimize = minimize

        c_exp = parse_expr(c)
        if not hasattr(self, 'variables'):
            self.variables = {v.name: v for v in sorted(c_exp.free_symbols, key=lambda v: v.name)}
        if len(self.variables) > 0 and not is_polynomial(c, self.variables):
            raise ValueError(f'{c} não é linear')
        self.c = np.array([[float(c_exp.coeff(var_label)) for var_label in self.variables.keys()]])

        if self.c.shape[0] != 1:
            raise ValueError
        else:
            self.n = self.c.shape[1]
        self.m = 0

        self.alt = np.empty((0, self.n), float)
        self.blt = np.empty((0, 1), float)
        self.aeq = np.empty((0, self.n), float)
        self.beq = np.empty((0, 1), float)
        self.agt = np.empty((0, self.n), float)
        self.bgt = np.empty((0, 1), float)

        if boundaries is None:
            boundaries = {}
        for var in self.variables.keys():
            if var in boundaries.keys():
                if boundaries[var][0] > boundaries[var][1]:
                    raise ValueError('Os limites das variáveis devem ser (a, b), a <= b ')
                if boundaries[var][0] is not None:
                    self.__add__(f'{var} >= {boundaries[var][0]}')
                if boundaries[var][1] is not None:
                    self.__add__(f'{var} <= {boundaries[var][1]}')
            else:
                self.__add__(f'{var} >= 0')

        self.integrity = []
        self.bool_constrains = []
        if var_type is None:
            var_type = {}
        for var in self.variables.keys():
            if var in var_type.keys():
                if isinstance(var_type[var], float):
                    self.integrity.append(0)
                elif isinstance(var_type[var], int):
                    self.integrity.append(1)
                elif isinstance(var_type[var], bool):
                    self.integrity.append(1)
                    self.bool_constrains.append(var)
                else:
                    raise TypeError('As variáveis devem ser do tipo float, int ou bool')
            else:
                self.integrity.append(0)


        self.__a_domain_str = 'A \u2208 \u211D\u207D\u00B9\u02E3\u207F\u207E'
        self.__x_domain_str = 'x \u2208 \u211D\u207D\u207F\u02E3\u00B9\u207E'
        self.__b_domain_str = 'b \u2208 \u211D'

    def __add__(self, other: str) -> Modelo:
        if not isinstance(other, str):
            raise ValueError
        if not self.is_restriction(other):
            raise ValueError(f'A restrição ({other}) deve ser Ax \u2264 b, Ax = b ou Ax \u2265 b,\n' +
                             f'\t\t\t\t{self.__a_domain_str}, {self.__x_domain_str} e {self.__b_domain_str}')
        if '>=' in other:
            a, b = other.split('>=')
            exp = parse_expr(a)
            a_row = np.array([float(exp.coeff(var)) for var in self.variables.keys()])
            if not a_row.any():
                return self
            unique = np.unique(np.hstack((np.vstack((self.agt, a_row)), np.vstack((self.bgt, np.array([float(b)]))))), axis=0)
            self.agt, self.bgt = unique[:, :-1], unique[:, [-1]]
        elif '<=' in other:
            a, b = other.split('<=')
            exp = parse_expr(a)
            a_row = np.array([float(exp.coeff(var)) for var in self.variables.keys()])
            if not a_row.any():
                return self
            unique = np.unique(np.hstack((np.vstack((self.alt, a_row)), np.vstack((self.blt, np.array([float(b)]))))), axis=0)
            self.alt, self.blt = unique[:, :-1], unique[:, [-1]]
        elif '=' in other:
            a, b = other.split('=')
            exp = parse_expr(a)
            a_row = np.array([float(exp.coeff(var)) for var in self.variables.keys()])
            if not a_row.any():
                return self
            unique = np.unique(np.hstack((np.vstack((self.aeq, a_row)), np.vstack((self.beq, np.array([float(b)]))))), axis=0)
            self.aeq, self.beq = unique[:, :-1], unique[:, [-1]]
        self.m += 1
        return self

    def __str__(self) -> str:
        return ''.join(('    min' if self._minimize else '    max',
                        self.cost_function_to_str(),
                        self.constrains_to_str() if self.m > 0 else ''))

    @property
    def solution(self):
        return self.solve()

    @property
    def extreme_points(self) -> Iterable[np.ndarray]:
        return filter(lambda p: is_in_feasible_region(self, p), get_extreme_points(self))

    def print_extreme_points(self) -> None:
        print('\n'.join((f'c({p}) = {(self.c @ p)[0]}' for p in self.extreme_points)))

    def solve(self, verbose: bool = False) -> Tuple[float | int]:
        constrains_gt = LinearConstraint(self.agt, self.bgt.flatten(), np.full_like(self.bgt.flatten(), np.inf))
        constrains_lt = LinearConstraint(self.alt, np.full_like(self.blt.flatten(), -np.inf), self.blt.flatten())
        constrains_eq = LinearConstraint(self.aeq, self.beq.flatten(), self.beq.flatten())
        a_bool = np.array([[*map(lambda x: int(x == var), self.variables)] for var in self.bool_constrains]).reshape((-1, self.n))
        constrains_bool_var = LinearConstraint(a_bool, np.zeros(len(a_bool)), np.ones(len(a_bool)))
        res = milp(c=self.c.flatten() if self._minimize else -self.c.flatten(),
                   integrality=np.array(self.integrity),
                   constraints=(constrains_gt, constrains_lt, constrains_eq, constrains_bool_var))

        if not res.success:
            raise OverflowError(res.message)
        if verbose:
            print('\n\tc(' + ', '.join([f'{v}={x:.3}' for x, v in zip(res.x, self.variables)]) +
                  f') = {res.fun if self._minimize else -res.fun}')

        return res.x

    def constrains_to_str(self) -> str:
        if self.m == 0:
            return ''
        n_a = [max_digits(column) for column in np.concatenate((self.alt, self.aeq, self.agt), axis=0).T]
        n_b = max_digits(np.concatenate((self.blt, self.beq, self.bgt)).T[0])
        constrains = []
        for coefficients_row, b_row in zip(self.alt, self.blt):
            constrains.append(expression_to_str(coefficients_row, self.variables, n_a) + ' \u2264 '
                              + monomial_to_str(b_row[0], None, n_b))
        for coefficients_row, b_row in zip(self.aeq, self.beq):
            constrains.append(expression_to_str(coefficients_row, self.variables, n_a) + ' = '
                              + monomial_to_str(b_row[0], None, n_b))
        for coefficients_row, b_row in zip(self.agt, self.bgt):
            constrains.append(expression_to_str(coefficients_row, self.variables, n_a) + ' \u2265 '
                              + monomial_to_str(b_row[0], None, n_b))

        return '\nsujeito a ' + '\n          '.join(constrains)

    def cost_function_to_str(self,) -> str:
        return expression_to_str(self.c.flatten(), self.variables, [max_digits([n]) for n in self.c.flatten()])

    def is_restriction(self, exp: str):
        for sign in ('<=', '>=', '='):
            if sign in exp:
                a, b = exp.split(sign)
                return is_polynomial(a.strip(' '), self.variables) and is_number(b.strip(' '))
        else:
            return False


def get_extreme_points(model: Modelo) -> np.ndarray:
    restrictions_a = np.concatenate((model.alt, model.aeq, model.agt))
    restrictions_b = np.concatenate((model.blt, model.beq, model.bgt))

    extreme_points = []
    for a_b_tuple in combinations(zip(restrictions_a, restrictions_b), model.n):
        a_array = np.empty((0, model.n), float)
        b_array = np.empty((0, 1), float)
        for a, b in a_b_tuple:
            a_array = np.concatenate((a_array, [a]))
            b_array = np.concatenate((b_array, [b]))
        extreme_points.append(solve(a_array, b_array.flatten()))

    return np.stack(extreme_points)


def is_in_feasible_region(model: Modelo, point: Tuple[float, float] | np.ndarray, erro: float = 0.001) -> bool:
    if any((a @ point > b + erro for a, b in zip(model.alt, model.blt))):
        return False
    if any((a @ point < b - erro for a, b in zip(model.agt, model.bgt))):
        return False
    if any((not (-erro < a @ point - b < erro) for a, b in zip(model.aeq, model.beq))):
        return False
    return True


if __name__ == '__main__':
    model = Modelo('100*A+150*B')
    model += '1/2 * A + 1/6 * B >= 20'
    model += '3/10 * A + 1/3 * B >= 27'
    model += '1/5 * A + 1/2 * B >= 30'
    print(model)
    model.solve(verbose=True)
    model.print_extreme_points()
