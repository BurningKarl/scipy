"""
Configuration options for the interior-point method
"""

import dataclasses
import enum
import itertools
import json
import typing

import numpy as np


class PreconditioningMethod(enum.Enum):
    NONE = "none"
    SKETCHING_QR = "sketching_qr"
    SKETCHING_CHOLESKY = "sketching_cholesky"


@dataclasses.dataclass
class PreconditioningOptions:
    """
    Stores all options for determining left and right preconditioners for the
    m x n matrix in the normal equation.

    Attributes
    ----------
    method : PreconditioningMethod (default = PreconditioningMethod.NONE)
        Specifies how the preconditioner is determined.
        - ``none``: No preconditioning
        - ``sketching_qr``: Preconditioners are determined using a QR
          decomposition of a sketched matrix.
        - ``sketching_cholesky``: Preconditioners are determined using a
          Cholesky decomposition of a sketched matrix
        Note that sketching will only work advantage if the coefficient
        matrix has much more variables than constraints (m << n)
    sketching_factor : float (default = 2)
        Determines the size of the sketched matrix to be
            ``sketching_factor * m`` x ``m`` matrix
    sketching_sparsity : int (default = 3)
        Determines the number of nonzero entries in each column if the a sparse
        sketch is used (``sparse`` = True).
    """

    method: PreconditioningMethod
    sketching_factor: float
    sketching_sparsity: int


@dataclasses.dataclass
class LinearSolverOptions:
    """
    Stores all options for determining the solver for the normal equation.

    Attributes
    ----------
    sparse : bool (default = False)
        Set to ``True`` if the problem is to be treated as sparse after
        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,
        this option will automatically be set ``True``, and the problem
        will be treated as sparse even during presolve. If your constraint
        matrices contain mostly zeros and the problem is not very small (less
        than about 100 constraints or variables), consider setting ``True``
        or providing ``A_eq`` and ``A_ub`` as sparse matrices.
    lstsq : bool (default = False)
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left ``False`` unless severe
        numerical difficulties are encountered. Leave this at the default
        unless you receive a warning message suggesting otherwise.
    sym_pos : bool (default = True)
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix
        (almost always). Leave this at the default unless you receive
        a warning message suggesting otherwise.
    cholesky : bool (default = True)
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for problems
        that are numerically well-behaved.
    iterative : bool (default = False)
        Set to ``True`` if the normal equations are to be solved by an iterative
        method. Depending on the value of sym_pos either CG or GMRES is used.
    linear_operators : bool (default = False)
        Set to ``True`` if the iterative method should use linear operators to
        compute products with the matrix in the normal equation lazily instead
        of calculating the matrix in advance.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``, and no SuiteSparse.)
        A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    preconditioning_options:
        Stores preconditioning options.
    """

    sparse: bool
    lstsq: bool
    sym_pos: bool
    cholesky: bool
    iterative: bool
    linear_operators: bool
    permc_spec: str
    preconditioning_options: PreconditioningOptions


@dataclasses.dataclass
class SearchDirectionOptions:
    """
    Stores all options for determining the search direction in each iteration
    of the interior-point method.

    Attributes
    ----------
    pc : bool (default = True)
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool (default = False)
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. Whether this is beneficial or not
        depends on the problem.
    solver_options: LinearSolverOptions
        Stores solver options.
    """

    pc: bool
    ip: bool
    solver_options: LinearSolverOptions


@dataclasses.dataclass
class IpmOptions:
    """
    Stores all options for the interior-point method

    Attributes
    ----------
    maxiter : int (default = 1000)
        The maximum number of iterations of the algorithm.
    tol : float (default = 1e-8)
        Termination tolerance to be used for all termination criteria;
        see [4]_ Section 4.5.
    disp : bool (default = False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    alpha0 : float (default = 0.99995)
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_{3}` of [4]_ Table 8.1.
    beta : float (default = 0.1)
        The desired reduction of the path parameter :math:`\mu` (see [6]_)
        when Mehrota's predictor-corrector is not in use (uncommon).
    sparse : bool (default = False)
        Set to ``True`` if the problem is to be treated as sparse after
        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,
        this option will automatically be set ``True``, and the problem
        will be treated as sparse even during presolve. If your constraint
        matrices contain mostly zeros and the problem is not very small (less
        than about 100 constraints or variables), consider setting ``True``
        or providing ``A_eq`` and ``A_ub`` as sparse matrices.
    lstsq : bool (default = False)
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left ``False`` unless severe
        numerical difficulties are encountered. Leave this at the default
        unless you receive a warning message suggesting otherwise.
    sym_pos : bool (default = True)
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix
        (almost always). Leave this at the default unless you receive
        a warning message suggesting otherwise.
    cholesky : bool (default = True)
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for problems
        that are numerically well-behaved.
    iterative : bool (default = False)
        Set to ``True`` if the normal equations are to be solved by an iterative
        method. Depending on the value of sym_pos either CG or GMRES is used.
    linear_operators : bool (default = False)
        Set to ``True`` if the iterative method should use linear operators to
        compute products with the matrix in the normal equation lazily instead
        of calculating the matrix in advance.
    pc : bool (default = True)
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool (default = False)
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. Whether this is beneficial or not
        depends on the problem.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``, and no SuiteSparse.)
        A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    sketching_method : str (default = 'none')
        Specifies how the preconditioner is determined.
        - ``none``: No preconditioning
        - ``sketching_qr``: Preconditioners are determined using a QR
          decomposition of a sketched matrix.
        - ``sketching_cholesky``: Preconditioners are determined using a
          Cholesky decomposition of a sketched matrix
        Note that sketching will only work advantage if the coefficient
        matrix has much more variables than constraints (m << n)
    sketching_factor : float (default = 2)
        Determines the size of the sketched matrix to be
            ``sketching_factor * m`` x ``m`` matrix
    sketching_sparsity : int (default = 3)
        Determines the number of nonzero entries in each column if the a sparse
        sketch is used (i.e. if ``sparse`` = True).
    """

    maxiter: int = 1000
    tol: float = 1e-8
    disp: bool = False
    alpha0: float = 0.99995
    beta: float = 0.1
    sparse: bool = False
    lstsq: bool = False
    sym_pos: bool = True
    cholesky: bool = True
    iterative: bool = False
    linear_operators: bool = False
    pc: bool = True
    ip: bool = False
    permc_spec: str = "MMD_AT_PLUS_A"
    preconditioning_method: str = "none"
    sketching_factor: float = 2
    sketching_sparsity: int = 3

    @classmethod
    def all_options(cls):
        return set(field.name for field in dataclasses.fields(cls))

    def search_direction_options(self):
        return SearchDirectionOptions(
            pc=self.pc,
            ip=self.ip,
            solver_options=LinearSolverOptions(
                sparse=self.sparse,
                lstsq=self.lstsq,
                sym_pos=self.sym_pos,
                cholesky=self.cholesky,
                iterative=self.iterative,
                linear_operators=self.linear_operators,
                permc_spec=self.permc_spec,
                preconditioning_options=PreconditioningOptions(
                    method=PreconditioningMethod(self.preconditioning_method),
                    sketching_factor=self.sketching_factor,
                    sketching_sparsity=self.sketching_sparsity,
                ),
            ),
        )
