from __future__ import annotations

from typing import Iterable, Dict

from sympy import Symbol, degree, parse_expr


def expression_to_str(coefficients: Iterable[float], variables: Iterable[str], n_list: Iterable[int]) -> str:
    expression = ''
    is_first = True
    for i, (coefficient, variable, n) in enumerate(zip(coefficients, variables, n_list)):
        if i != 0:
            if coefficient < 0 and not is_first:
                expression += ' - '
            elif coefficient > 0 and not is_first:
                expression += ' + '
            else:
                expression += '   '
        else:
            if coefficient == 0:
                expression += '  '
            elif coefficient < 0:
                expression += '- '
                is_first = False
            elif coefficient > 0:
                expression += '  '
                is_first = False

        expression += monomial_to_str(abs(coefficient), variable, n)
    return expression


def monomial_to_str(coefficient: float, variable: str | None, n: int) -> str:
    integer, decimal = str(coefficient).split('.') if '.' in str(coefficient) else (coefficient, '')
    if not coefficient.is_integer():
        return f'{str(coefficient)[:n]:>{n}}' + (f' * {variable}' if variable is not None else '')
    elif coefficient == 0:
        return ' ' * (n + 3 + len(variable)) if variable is not None else f'{integer:>{n}}'
    elif coefficient == 1:
        return (' ' * (n + 3) + f'{variable}') if variable is not None else f'{integer:>{n}}'
    return f'{integer:>{n}}' + (f' * {variable}' if variable is not None else '')


def max_digits(numbers: Iterable[float], max_d: int = 3) -> int:
    return max((len(str(abs(int(number)) if number.is_integer() else abs(round(number, max_d)))) for number in numbers))


def is_polynomial(exp: str, variables: Dict[str, Symbol]) -> bool:
    return any((degree(parse_expr(exp), var) <= 1 for var in variables.values()))

def is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


