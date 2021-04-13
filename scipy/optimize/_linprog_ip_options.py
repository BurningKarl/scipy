"""
Configuration options for the interior-point method
"""

import enum
import warnings
from dataclasses import dataclass, field, fields

from .optimize import OptimizeWarning, _check_unknown_options


class PreconditioningMethod(enum.Enum):
    NONE = "none"
    SKETCHING_QR = "sketching_qr"
    SKETCHING_CHOLESKY = "sketching_cholesky"


@dataclass
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

    preconditioning_method: PreconditioningMethod = PreconditioningMethod.NONE
    sketching_factor: float = 2
    sketching_sparsity: int = 3

    def __post_init__(self):
        if isinstance(self.preconditioning_method, str):
            self.preconditioning_method = PreconditioningMethod(
                self.preconditioning_method
            )


@dataclass
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
    """

    sparse: bool = False
    lstsq: bool = False
    sym_pos: bool = True
    cholesky: bool = True
    iterative: bool = False
    linear_operators: bool = False
    permc_spec: str = "MMD_AT_PLUS_A"


@dataclass
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
    """

    pc: bool = True
    ip: bool = False


@dataclass
class IpmOptions:
    """
    Stores all options for the main IPM loop.

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
    """
    maxiter: int = 1000
    tol: float = 1e-8
    disp: bool = False
    alpha0: float = 0.99995
    beta: float = 0.1



@dataclass
class AllOptions:
    """
    Stores all options for the interior-point method

    Attributes
    ----------
    All of these attributes can be supplied via the from_dict method.

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

    ipm: IpmOptions = field(default_factory=IpmOptions)
    search_direction: SearchDirectionOptions = field(default_factory=SearchDirectionOptions)
    linear_solver: LinearSolverOptions = field(default_factory=LinearSolverOptions)
    preconditioning: PreconditioningOptions = field(default_factory=PreconditioningOptions)

    @staticmethod
    def from_dict(options_dict, has_umfpack=False, has_cholmod=False):
        options = AllOptions(
            **{
                suboptions.name: suboptions.default_factory(
                    **{
                        f.name: options_dict.pop(f.name)
                        for f in fields(suboptions.default_factory)
                        if f.name in options_dict
                    }
                )
                for suboptions in fields(AllOptions)
            }
        )

        # Any entries left in options_dict are unknown options
        _check_unknown_options(options_dict)

        # These should be warnings, not errors
        if (
            options.linear_solver.cholesky
            and options.linear_solver.sparse
            and not has_cholmod
        ):
            warnings.warn(
                "Sparse cholesky is only available with scikit-sparse. "
                "Dense cholesky will be used.",
                OptimizeWarning,
                stacklevel=4,
            )

        if options.linear_solver.sparse and options.linear_solver.lstsq:
            warnings.warn(
                "Option combination 'sparse':True and 'lstsq':True is not "
                "recommended.",
                OptimizeWarning,
                stacklevel=4,
            )

        if options.linear_solver.lstsq and options.linear_solver.cholesky:
            warnings.warn(
                "Invalid option combination 'lstsq':True and 'cholesky':True; "
                "option 'cholesky' has no effect when 'lstsq' is set True.",
                OptimizeWarning,
                stacklevel=4,
            )

        if options.linear_solver.iterative and options.linear_solver.cholesky:
            warnings.warn(
                "Invalid option combination 'iterative':True and "
                "'cholesky':True; option 'cholesky' has no effect when "
                "'iterative' is set True.",
                OptimizeWarning,
                stacklevel=4,
            )

        if options.linear_solver.iterative and options.linear_solver.lstsq:
            warnings.warn(
                "Invalid option combination 'iterative':True and 'lstsq':True; "
                "option 'lstsq' has no effect when 'iterative' is set True.",
                OptimizeWarning,
                stacklevel=4,
            )

        if (
            options.linear_solver.linear_operators
            and not options.linear_solver.iterative
        ):
            warnings.warn(
                "Invalid option combination 'linear_operators':True and "
                "'iterative':False; option 'linear_operators' has no effect "
                "when 'iterative' is set False.",
                OptimizeWarning,
                stacklevel=4,
            )

        valid_permc_spec = ("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD")
        if options.linear_solver.permc_spec.upper() not in valid_permc_spec:
            warnings.warn(
                "Invalid permc_spec option: '" + str(self.permc_spec) + "'. "
                "Acceptable values are 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', "
                "and 'COLAMD'. Reverting to default.",
                OptimizeWarning,
                stacklevel=4,
            )
            options.linear_solver.permc_spec = "MMD_AT_PLUS_A"

        # This can be an error
        if not options.linear_solver.sym_pos and options.linear_solver.cholesky:
            raise ValueError(
                "Invalid option combination 'sym_pos':False "
                "and 'cholesky':True: Cholesky decomposition is only possible "
                "for symmetric positive definite matrices."
            )

        options.linear_solver.cholesky = options.linear_solver.cholesky or (
            options.linear_solver.cholesky is None
            and options.linear_solver.sym_pos
            and not options.linear_solver.lstsq
        )

        return options
