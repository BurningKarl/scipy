"""Interior-point method for linear programming

The *interior-point* method uses the primal-dual path following algorithm
outlined in [1]_. This algorithm supports sparse constraint matrices and
is typically faster than the simplex methods, especially for large, sparse
problems. Note, however, that the solution returned may be slightly less
accurate than those of the simplex methods and will not, in general,
correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

References
----------
.. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
       optimizer for linear programming: an implementation of the
       homogeneous algorithm." High performance optimization. Springer US,
       2000. 197-232.
"""
# Author: Matt Haberland

import collections
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.stats as sts
import wandb
from codetiming import Timer
from logzero import logger
from warnings import warn
from scipy.linalg import LinAlgError, solve_triangular
from .optimize import OptimizeWarning, OptimizeResult
from ._linprog_util import _postsolve
from ._linprog_ip_options import (
    AllOptions,
    LinearSolverOptions,
    PreconditioningMethod,
)

has_umfpack = True
has_cholmod = True
try:
    import sksparse
    from sksparse.cholmod import cholesky as cholmod
    from sksparse.cholmod import analyze as cholmod_analyze
except ImportError:
    has_cholmod = False
try:
    import scikits.umfpack  # test whether to use factorized
except ImportError:
    has_umfpack = False

default_rng = np.random.default_rng()


def _construct_sketching_matrix(
    w, n, s=3, sparse=True, fast=False, rng=default_rng
):
    """
    Randomly construct a sketching matrix of size (w, n) where w is assumed to
    be smaller than n. If sparse is True, each column contains (approximately)
    s nonzero entries randomly drawn from +-1/sqrt(s). Otherwise all matrix
    entries are drawn from a normal distribution.

    Parameters
    ----------
    w : int
        Number of rows of the sketching matrix
    n : int
        Number columns of the sketching matrix
    s : int
        Number of nonzero entries in each row if ``sparse = True``
    sparse : bool
        True if the sketching matrix should be sparse.
    rng : np.random.Generator (default = np.random.default_rng())
        A random number generator used to construct the random matrix

    Returns
    -------
    A random sketching matrix

    """
    if sparse:
        data = rng.choice([-1 / np.sqrt(s), 1 / np.sqrt(s)], size=s * n)
        if fast:
            row_indices = rng.choice(w, size=n * s)
        else:
            row_indices = rng.choice(w, size=(int(n * 1.1), s))
            row_indices.sort()
            row_indices = row_indices[
                (row_indices[..., 1:] != row_indices[..., :-1]).all(axis=-1)
            ]
            row_indices = row_indices[:n]
            row_indices.shape = (n * s,)

        column_indices = np.repeat(np.arange(n), s)
        mat = sps.coo_matrix(
            (data, (row_indices, column_indices)), shape=(w, n)
        )
        return mat.tocsr()
    else:
        return rng.normal(size=(w, n)) / np.sqrt(w)


def _assemble_matrix(A, Dinv, options):
    """
    Given the current linear system return a (possibly preconditioned) matrix
    for which a solver is needed and a function to turn such a solver into a
    solver for the matrix ``M = A * Dinv * A.T`` as defined in [4] Equation 8.31

    Parameters
    ----------
    A : 2-D array
    Dinv : 2-D array
        As defined in [4] Equation 8.31
    options : AllOptions
        Provides the options to choose the preconditioners.

    Returns
    -------
    matrix: 2-D array
        The preconditioned matrix
    preconditioned_solver: function
        A function decorator that takes in a solver function for the
        preconditioned matrix and returns a solve function for ``M``.
    """
    method = options.preconditioning.preconditioning_method

    if options.linear_solver.linear_operators:
        A_op = sps.linalg.aslinearoperator(A)
        Dinv_op = sps.linalg.aslinearoperator(sps.diags(Dinv, 0, format="csc"))
        M = A_op * Dinv_op * A_op.T
    else:
        if options.linear_solver.sparse:
            M = A.dot(sps.diags(Dinv, 0, format="csc").dot(A.T))
        else:
            M = A.dot(Dinv.reshape(-1, 1) * A.T)

    if method is PreconditioningMethod.NONE:

        def preconditioned_solver(solve):
            def new_solve(r):
                with Timer(name="solve", logger=None):
                    x = solve(r)
                return x

            return new_solve

        return M, preconditioned_solver
    elif method is PreconditioningMethod.SKETCHING:
        Dinv_half = np.sqrt(Dinv)

        with Timer(logger=None) as generate_sketch_timer:
            sketching_matrix = _construct_sketching_matrix(
                int(options.preconditioning.sketching_factor * A.shape[0]),
                A.shape[1],
                options.preconditioning.sketching_sparsity,
                sparse=options.linear_solver.sparse,
            )

        with Timer(logger=None) as sketching_timer:
            if options.linear_solver.sparse:
                sketched_matrix = (
                    sketching_matrix @ sps.diags(Dinv_half, format="csc") @ A.T
                )
            else:
                sketched_matrix = sketching_matrix @ (
                    Dinv_half.reshape(-1, 1) * A.T
                )

        with Timer(logger=None) as decomposition_timer:
            R = np.linalg.qr(
                sketched_matrix.toarray()
                if options.linear_solver.sparse
                else sketched_matrix,
                mode="r",
            )

        with Timer(logger=None) as product_timer:
            if options.linear_solver.linear_operators:
                if options.preconditioning.triangular_solve:
                    Rinv = sps.linalg.LinearOperator(
                        R.shape,  # R is square
                        matvec=lambda x: solve_triangular(R, x, trans="N"),
                        rmatvec=lambda x: solve_triangular(R, x, trans="T"),
                    )
                else:
                    Rinv = sps.linalg.aslinearoperator(np.linalg.inv(R))
                matrix = Rinv.T @ M @ Rinv
            else:
                Rinv = np.linalg.inv(R)
                matrix = (
                    Rinv.T @ (M.toarray() if sps.isspmatrix(M) else M) @ Rinv
                )


        statistics = {
            "generate_sketch_duration": generate_sketch_timer.last,
            "sketching_duration": sketching_timer.last,
            "decomposition_duration": decomposition_timer.last,
            "product_duration": product_timer.last,
        }
        if (
            options.linear_solver.log_conditioning_and_rank
            or options.linear_solver.log_sparsity
        ):
            if isinstance(matrix, sps.linalg.LinearOperator):
                dense_matrix = matrix @ np.eye(*matrix.shape)
            elif sps.isspmatrix(matrix):
                dense_matrix = matrix.toarray()
            else:
                dense_matrix = matrix
            if isinstance(M, sps.linalg.LinearOperator):
                dense_M = M @ np.eye(*M.shape)
            elif sps.isspmatrix(M):
                dense_M = M.toarray()
            else:
                dense_M = M
        if options.linear_solver.log_conditioning_and_rank:
            statistics.update({
                "condition_number": np.linalg.cond(dense_M),
                "condition_number_sketched": np.linalg.cond(dense_matrix),
                "rank": np.linalg.matrix_rank(dense_M),
                "rank_sketched": np.linalg.matrix_rank(dense_matrix),
            })
        if options.linear_solver.log_sparsity:
            statistics.update({
                "nnz_sketched": sketched_matrix.count_nonzero(),
                "density_sketched": sketched_matrix.count_nonzero()
                / (sketched_matrix.shape[0] * sketched_matrix.shape[1]),
                "nnz_coefficient": A.count_nonzero(),
                "density_coefficient": A.count_nonzero()
                / (A.shape[0] * A.shape[1]),
                "nnz_matrix": sps.coo_matrix(dense_matrix).count_nonzero(),
                "density_matrix": sps.coo_matrix(dense_matrix).count_nonzero()
                / (matrix.shape[0] * matrix.shape[1]),
                "nnz_M": sps.coo_matrix(dense_M).count_nonzero(),
                "density_M": sps.coo_matrix(dense_M).count_nonzero()
                / (M.shape[0] * M.shape[1]),
            })

        wandb.log(statistics, commit=False)

        def preconditioned_solver(solve):
            def new_solve(r):
                new_r = Rinv.T @ r
                with Timer(name="solve", logger=None):
                    new_x = solve(new_r)
                x = Rinv @ new_x

                if not hasattr(new_solve, "statistics"):
                    new_solve.statistics = collections.defaultdict(list)
                new_solve.statistics["residual_M"].append(
                    np.linalg.norm(M @ x - r) / np.linalg.norm(r)
                )
                wandb.log(
                    {
                        f"residual_M[{i}]": v
                        for i, v in enumerate(
                            new_solve.statistics["residual_M"]
                        )
                    },
                    commit=False,
                )

                return x

            return new_solve

        return matrix, preconditioned_solver


def _get_solver(M, options):
    """
    Given a matrix and solver options, return a handle to the appropriate linear
    system solver.

    Parameters
    ----------
    M : 2-D array
        The matrix to be solved
    options : AllOptions
        Provides the options to choose the linear solver.

    Returns
    -------
    solve : function
        Handle to the appropriate solver function

    """

    try:
        if options.linear_solver.iterative:
            if options.linear_solver.sym_pos:
                logger.debug("Using scipy.sparse.linalg.cg")

                def solve(r):
                    x, info, iterations, residual = sps.linalg.cg(
                        M,
                        r,
                        maxiter=options.linear_solver.solver_maxiter,
                        tol=options.linear_solver.solver_rtol,
                        atol=options.linear_solver.solver_atol,
                    )
                    if not hasattr(solve, "statistics"):
                        solve.statistics = collections.defaultdict(list)
                    solve.statistics["inner_iterations"].append(iterations)
                    solve.statistics["residual"].append(residual)
                    wandb.log(
                        {
                            **{
                                f"inner_iterations[{i}]": v
                                for i, v in enumerate(
                                    solve.statistics["inner_iterations"]
                                )
                            },
                            **{
                                f"residual[{i}]": v
                                for i, v in enumerate(
                                    solve.statistics["residual"]
                                )
                            }
                        },
                        commit=False,
                    )
                    if info < 0:
                        raise LinAlgError(f"CG failed ({info})!")
                    else:
                        return x

            else:
                logger.debug("Using scipy.sparse.linalg.gmres")

                def solve(r):
                    x, info, iterations, residual = sps.linalg.gmres(
                        M,
                        r,
                        maxiter=options.linear_solver.solver_maxiter,
                        tol=options.linear_solver.solver_rtol,
                        atol=options.linear_solver.solver_atol,
                    )
                    if not hasattr(solve, "statistics"):
                        solve.statistics = collections.defaultdict(list)
                    solve.statistics["inner_iterations"].append(iterations)
                    solve.statistics["residual"].append(residual)
                    wandb.log(
                        {
                            **{
                                f"inner_iterations[{i}]": v
                                for i, v in enumerate(
                                    solve.statistics["inner_iterations"]
                                )
                            },
                            **{
                                f"residual[{i}]": v
                                for i, v in enumerate(
                                    solve.statistics["residual"]
                                )
                            }
                        },
                        commit=False,
                    )
                    if info < 0:
                        raise LinAlgError(f"GMRES failed ({info})!")
                    else:
                        return x

        elif options.linear_solver.sparse:
            if options.linear_solver.lstsq:
                logger.debug("Using scipy.sparse.linalg.lsqr")

                def solve(r, sym_pos=False):
                    return sps.linalg.lsqr(M, r)[0]

            elif options.linear_solver.cholesky:
                if has_cholmod:
                    logger.debug("Using sksparse.cholmod.analyze")
                    try:
                        # Will raise an exception in the first call,
                        # or when the matrix changes due to a new problem
                        _get_solver.cholmod_factor.cholesky_inplace(M)
                    except Exception:
                        _get_solver.cholmod_factor = cholmod_analyze(M)
                        _get_solver.cholmod_factor.cholesky_inplace(M)
                    solve = _get_solver.cholmod_factor
                else:
                    logger.debug("Using scipy.linalg.cho_factor")
                    L = sp.linalg.cho_factor(M.toarray())

                    def solve(r):
                        return sp.linalg.cho_solve(L, r)

            else:
                if has_umfpack and options.linear_solver.sym_pos:
                    logger.debug("Using scipy.sparse.linalg.factorized")
                    solve = sps.linalg.factorized(M)
                else:  # factorized doesn't pass permc_spec
                    logger.debug("Using scipy.sparse.linalg.splu")
                    solve = sps.linalg.splu(
                        M, permc_spec=options.linear_solver.permc_spec
                    ).solve

        else:
            if options.linear_solver.lstsq:
                # sometimes necessary as solution is approached
                logger.debug("Using scipy.linalg.lstsq")

                def solve(r):
                    return sp.linalg.lstsq(M, r)[0]

            elif options.linear_solver.cholesky:
                logger.debug("Using scipy.linalg.cho_factor")
                L = sp.linalg.cho_factor(M)

                def solve(r):
                    return sp.linalg.cho_solve(L, r)

            else:
                logger.debug("Using scipy.linalg.solve")
                # this seems to cache the matrix factorization, so solving
                # with multiple right hand sides is much faster

                def solve(r, sym_pos=options.linear_solver.sym_pos):
                    return sp.linalg.solve(M, r, sym_pos=sym_pos)

    # There are many things that can go wrong here, and it's hard to say
    # what all of them are. It doesn't really matter: if the matrix can't be
    # factorized, return None. get_solver will be called again with different
    # inputs, and a new routine will try to factorize the matrix.
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(e)
        raise
        return None
    return solve


def _get_delta(A, b, c, x, y, z, tau, kappa, gamma, eta, options):
    """
    Given standard form problem defined by ``A``, ``b``, and ``c``;
    current variable estimates ``x``, ``y``, ``z``, ``tau``, and ``kappa``;
    algorithmic parameters ``gamma and ``eta;
    and options ``sparse``, ``lstsq``, ``sym_pos``, ``cholesky``, ``pc``
    (predictor-corrector), and ``ip`` (initial point improvement),
    get the search direction for increments to the variable estimates.

    Parameters
    ----------
    As defined in [4], except:
    options : AllOptions
        Provides the options used to determine the search direction.

    Returns
    -------
    Search directions as defined in [4]

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    if A.shape[0] == 0:
        # If there are no constraints, some solvers fail (understandably)
        # rather than returning empty solution. This gets the job done.
        options.linear_solver = LinearSolverOptions()

    n_x = len(x)

    # [4] Equation 8.8
    r_P = b * tau - A.dot(x)
    r_D = c * tau - A.T.dot(y) - z
    r_G = c.dot(x) - b.transpose().dot(y) + kappa
    mu = (x.dot(z) + tau * kappa) / (n_x + 1)

    #  Assemble M from [4] Equation 8.31 inside _assemble_matrix
    Dinv = x / z
    matrix, preconditioned_solver = _assemble_matrix(A, Dinv, options)
    solve = preconditioned_solver(_get_solver(matrix, options))

    # pc: "predictor-corrector" [4] Section 4.1
    # In development this option could be turned off
    # but it always seems to improve performance substantially
    n_corrections = 1 if options.search_direction.pc else 0

    alpha, d_x, d_z, d_tau, d_kappa = 0, 0, 0, 0, 0
    for i in range(n_corrections + 1):
        # Reference [4] Eq. 8.6
        rhatp = eta(gamma) * r_P
        rhatd = eta(gamma) * r_D
        rhatg = eta(gamma) * r_G

        # Reference [4] Eq. 8.7
        rhatxs = gamma * mu - x * z
        rhattk = gamma * mu - tau * kappa

        if i == 1:
            if options.search_direction.ip:
                # if the correction is to get "initial point"
                # Reference [4] Eq. 8.23
                rhatxs = ((1 - alpha) * gamma * mu -
                          x * z - alpha**2 * d_x * d_z)
                rhattk = ((1 - alpha) * gamma * mu -
                    tau * kappa -
                    alpha**2 * d_tau * d_kappa)
            else:
                # if the correction is for "predictor-corrector"
                # Reference [4] Eq. 8.13
                rhatxs -= d_x * d_z
                rhattk -= d_tau * d_kappa

        # sometimes numerical difficulties arise as the solution is approached
        # this loop tries to solve the equations using a sequence of functions
        # for solve. For dense systems, the order is:
        # 1. scipy.linalg.cho_factor/scipy.linalg.cho_solve,
        # 2. scipy.linalg.solve w/ sym_pos = True,
        # 3. scipy.linalg.solve w/ sym_pos = False, and if all else fails
        # 4. scipy.linalg.lstsq
        # For sparse systems, the order is:
        # 1. sksparse.cholmod.cholesky (if available)
        # 2. scipy.sparse.linalg.factorized (if umfpack available)
        # 3. scipy.sparse.linalg.splu
        # 4. scipy.sparse.linalg.lsqr
        solved = False
        while not solved:
            try:
                # [4] Equation 8.28
                p, q = _sym_solve(Dinv, A, c, b, solve)
                # [4] Equation 8.29
                u, v = _sym_solve(Dinv, A, rhatd -
                                  (1 / x) * rhatxs, rhatp, solve)
                if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                    raise LinAlgError
                solved = True
            except (LinAlgError, ValueError, TypeError) as e:
                # Usually this doesn't happen. If it does, it happens when
                # there are redundant constraints or when approaching the
                # solution. If so, change solver.
                logger.error(e)
                raise

                if options.linear_solver.iterative:
                    options.linear_solver.iterative = False
                    if options.linear_solver.linear_operators:
                        options.linear_solver.linear_operators = False
                        # Non-iterative methods cant handle linear operators
                        matrix, preconditioned_solver = _assemble_matrix(
                            A, Dinv, options
                        )

                    warn(
                        "Solving system with option 'iterative':True "
                        "failed. It is normal for this to happen "
                        "occasionally, especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "consider setting option 'iterative' to False.",
                        OptimizeWarning, stacklevel=5)
                elif options.linear_solver.cholesky:
                    options.linear_solver.cholesky = False
                    warn(
                        "Solving system with option 'cholesky':True "
                        "failed. It is normal for this to happen "
                        "occasionally, especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "consider setting option 'cholesky' to False.",
                        OptimizeWarning, stacklevel=5)
                elif options.linear_solver.sym_pos:
                    options.linear_solver.sym_pos = False
                    warn(
                        "Solving system with option 'sym_pos':True "
                        "failed. It is normal for this to happen "
                        "occasionally, especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "consider setting option 'sym_pos' to False.",
                        OptimizeWarning, stacklevel=5)
                elif not options.linear_solver.lstsq:
                    options.linear_solver.lstsq = True
                    warn(
                        "Solving system with option 'sym_pos':False "
                        "failed. This may happen occasionally, "
                        "especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "your problem may be numerically challenging. "
                        "If you cannot improve the formulation, consider "
                        "setting 'lstsq' to True. Consider also setting "
                        "`presolve` to True, if it is not already.",
                        OptimizeWarning, stacklevel=5)
                else:
                    raise e
                solve = _get_solver(matrix, options)
        # [4] Results after 8.29
        d_tau = ((rhatg + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) /
                 (1 / tau * kappa + (-c.dot(p) + b.dot(q))))
        d_x = u + p * d_tau
        d_y = v + q * d_tau

        # [4] Relations between  after 8.25 and 8.26
        d_z = (1 / x) * (rhatxs - z * d_x)
        d_kappa = 1 / tau * (rhattk - kappa * d_tau)

        # [4] 8.12 and "Let alpha be the maximal possible step..." before 8.23
        alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
        if options.search_direction.ip:  # initial point - see [4] 4.4
            gamma = 10
        else:  # predictor-corrector, [4] definition after 8.12
            beta1 = 0.1  # [4] pg. 220 (Table 8.1)
            gamma = (1 - alpha)**2 * min(beta1, (1 - alpha))

    return d_x, d_y, d_z, d_tau, d_kappa


def _sym_solve(Dinv, A, r1, r2, solve):
    """
    An implementation of [4] equation 8.31 and 8.32

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # [4] 8.31
    r = r2 + A.dot(Dinv * r1)
    v = solve(r)
    # [4] 8.32
    u = Dinv * (A.T.dot(v) - r1)
    return u, v


def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    """
    An implementation of [4] equation 8.21

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # [4] 4.3 Equation 8.21, ignoring 8.20 requirement
    # same step is taken in primal and dual spaces
    # alpha0 is basically beta3 from [4] Table 8.1, but instead of beta3
    # the value 1 is used in Mehrota corrector and initial point correction
    i_x = d_x < 0
    i_z = d_z < 0
    alpha_x = alpha0 * np.min(x[i_x] / -d_x[i_x]) if np.any(i_x) else 1
    alpha_tau = alpha0 * tau / -d_tau if d_tau < 0 else 1
    alpha_z = alpha0 * np.min(z[i_z] / -d_z[i_z]) if np.any(i_z) else 1
    alpha_kappa = alpha0 * kappa / -d_kappa if d_kappa < 0 else 1
    alpha = np.min([1, alpha_x, alpha_tau, alpha_z, alpha_kappa])
    return alpha


def _get_message(status):
    """
    Given problem status code, return a more detailed message.

    Parameters
    ----------
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    Returns
    -------
    message : str
        A string descriptor of the exit status of the optimization.

    """
    messages = (
        ["Optimization terminated successfully.",
         "The iteration limit was reached before the algorithm converged.",
         "The algorithm terminated successfully and determined that the "
         "problem is infeasible.",
         "The algorithm terminated successfully and determined that the "
         "problem is unbounded.",
         "Numerical difficulties were encountered before the problem "
         "converged. Please check your problem formulation for errors, "
         "independence of linear equality constraints, and reasonable "
         "scaling and matrix condition numbers. If you continue to "
         "encounter this error, please submit a bug report."
         ])
    return messages[status]


def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    """
    An implementation of [4] Equation 8.9

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    z = z + alpha * d_z
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return x, y, z, tau, kappa


def _get_blind_start(shape):
    """
    Return the starting point from [4] 4.4

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    m, n = shape
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    """
    Implementation of several equations from [4] used as indicators of
    the status of optimization.

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """

    # residuals for termination are relative to initial values
    x0, y0, z0, tau0, kappa0 = _get_blind_start(A.shape)

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    def r_p(x, tau):
        return b * tau - A.dot(x)

    def r_d(y, z, tau):
        return c * tau - A.T.dot(y) - z

    def r_g(x, y, kappa):
        return kappa + c.dot(x) - b.dot(y)

    # np.dot unpacks if they are arrays of size one
    def mu(x, tau, z, kappa):
        return (x.dot(z) + np.dot(tau, kappa)) / (len(x) + 1)

    obj = c.dot(x / tau) + c0

    def norm(a):
        return np.linalg.norm(a)

    # See [4], Section 4.5 - The Stopping Criteria
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / max(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / max(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / max(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0

    wandb.log(
        {
            "rho_A": rho_A,
            "rho_p": rho_p,
            "rho_d": rho_d,
            "rho_g": rho_g,
            "rho_mu": rho_mu,
        },
        commit=False,
    )

    return rho_p, rho_d, rho_A, rho_g, rho_mu, obj


def _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj, header=False):
    """
    Print indicators of optimization status to the console.

    Parameters
    ----------
    rho_p : float
        The (normalized) primal feasibility, see [4] 4.5
    rho_d : float
        The (normalized) dual feasibility, see [4] 4.5
    rho_g : float
        The (normalized) duality gap, see [4] 4.5
    alpha : float
        The step size, see [4] 4.3
    rho_mu : float
        The (normalized) path parameter, see [4] 4.5
    obj : float
        The objective function value of the current iterate
    header : bool
        True if a header is to be printed

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    if header:
        print("Primal Feasibility ",
              "Dual Feasibility   ",
              "Duality Gap        ",
              "Step            ",
              "Path Parameter     ",
              "Objective          ")

    print(
        f"{rho_p:011.5E}         "
        + f"{rho_d:011.5E}         "
        + f"{rho_g:011.5E}         "
        + (
            f"{alpha:011.5E}      "
            if isinstance(alpha, float)
            else f"{alpha:<17}"
        )
        + f"{rho_mu:011.5E}         "
        + f"{obj:011.5E}         "
    )


def _ip_hsd(A, b, c, c0, callback, postsolve_args, options):
    r"""
    Solve a linear programming problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    using the interior point method of [4].

    Parameters
    ----------
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A`` (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True only when an algorithm has completed successfully,
                so this is always False as the callback function is called
                only while the algorithm is still iterating.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed. This is always
                1 for the interior-point method because it has only one phase.
            status : int
                For revised simplex, this is always 0 because if a different
                status is detected, the algorithm terminates.
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.
    options : AllOptions
        Provides all options.

    Returns
    -------
    x_hat : float
        Solution vector (for standard form problem).
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at:
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf

    """

    iteration = 0

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A.shape)

    # first iteration is special improvement of initial point
    options.search_direction.ip = (
        options.search_direction.ip if options.search_direction.pc else False
    )

    # [4] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)
    go = (
        rho_p > options.ipm.tol
        or rho_d > options.ipm.tol
        or rho_A > options.ipm.tol
    )

    best_iteration = 0
    best_indicators = {
        "rho_A": rho_A,
        "rho_p": rho_p,
        "rho_d": rho_d,
        "rho_g": rho_g,
        "rho_mu": rho_mu,
    }
    wandb.summary.update({"best_iteration": best_iteration, **best_indicators})
    if options.ipm.disp:
        _display_iter(rho_p, rho_d, rho_g, "-", rho_mu, obj, header=True)
    if callback is not None:
        x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                              'con': con, 'nit': iteration, 'phase': 1,
                              'complete': False, 'status': 0,
                              'message': "", 'success': False})
        callback(res)

    status = 0
    message = "Optimization terminated successfully."

    if options.linear_solver.sparse:
        A = sps.csc_matrix(A)
        A.T = A.transpose()  # A.T is defined for sparse matrices but is slow
        # Redefine it to avoid calculating again
        # This is fine as long as A doesn't change

    while go:

        iteration += 1

        if options.search_direction.ip:  # initial point
            # [4] Section 4.4
            gamma = 1

            def eta(g):
                return 1
        else:
            # gamma = 0 in predictor step according to [4] 4.1
            # if predictor/corrector is off, use mean of complementarity [6]
            # 5.1 / [4] Below Figure 10-4
            gamma = (
                0
                if options.search_direction.pc
                else options.ipm.beta * np.mean(z * x)
            )
            # [4] Section 4.1

            def eta(g=gamma):
                return 1 - g

        try:
            # Solve [4] 8.6 and 8.7/8.13/8.23
            d_x, d_y, d_z, d_tau, d_kappa = _get_delta(
                A, b, c, x, y, z, tau, kappa, gamma, eta, options
            )

            if options.search_direction.ip:  # initial point
                # [4] 4.4
                # Formula after 8.23 takes a full step regardless if this will
                # take it negative
                alpha = 1.0
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y,
                    d_z, d_tau, d_kappa, alpha)
                x[x < 1] = 1
                z[z < 1] = 1
                tau = max(1, tau)
                kappa = max(1, kappa)
                options.search_direction.ip = False  # done with initial point
            else:
                # [4] Section 4.3
                alpha = _get_step(
                    x, d_x, z, d_z, tau, d_tau, kappa, d_kappa,
                    options.ipm.alpha0
                )
                # [4] Equation 8.9
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)

        except (LinAlgError, FloatingPointError,
                ValueError, ZeroDivisionError) as e:
            # this can happen when sparse solver is used and presolve
            # is turned off. Also observed ValueError in AppVeyor Python 3.6
            # Win32 build (PR #8676). I've never seen it otherwise.
            logger.error(e)
            status = 4
            message = _get_message(status)
            break
        finally:
            statistics = {
                "solve_duration": Timer.timers.total("solve"),
            }
            wandb.log(statistics, commit=True)
            Timer.timers.clear()

        # [4] 4.5
        rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
            A, b, c, c0, x, y, z, tau, kappa)
        go = (
            rho_p > options.ipm.tol
            or rho_d > options.ipm.tol
            or rho_A > options.ipm.tol
        )

        if best_indicators["rho_p"] > rho_p:
            best_iteration = iteration
            best_indicators = {
                "rho_A": rho_A,
                "rho_p": rho_p,
                "rho_d": rho_d,
                "rho_g": rho_g,
                "rho_mu": rho_mu,
            }
            wandb.summary.update(
                {"best_iteration": best_iteration, **best_indicators}
            )
        if options.ipm.disp:
            _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj)
        if callback is not None:
            x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
            res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                                  'con': con, 'nit': iteration, 'phase': 1,
                                  'complete': False, 'status': 0,
                                  'message': "", 'success': False})
            callback(res)

        # [4] 4.5
        inf1 = (
            rho_p < options.ipm.tol
            and rho_d < options.ipm.tol
            and rho_g < options.ipm.tol
            and tau < options.ipm.tol * max(1, kappa)
        )
        inf2 = (
            rho_mu < options.ipm.tol
            and tau < options.ipm.tol * min(1, kappa)
        )
        if inf1 or inf2:
            # [4] Lemma 8.4 / Theorem 8.3
            if b.transpose().dot(y) > options.ipm.tol:
                status = 2
            else:  # elif c.T.dot(x) < tol: ? Probably not necessary.
                status = 3
            message = _get_message(status)
            break
        elif iteration >= options.ipm.maxiter:
            status = 1
            message = _get_message(status)
            break

    x_hat = x / tau
    # [4] Statement after Theorem 8.2
    return x_hat, status, message, iteration


def _linprog_ip(c, c0, A, b, callback, postsolve_args, **options_dict):
    r"""
    Minimize a linear objective function subject to linear
    equality and non-negativity constraints using the interior point method
    of [4]_. Linear programming is intended to solve problems
    of the following form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the right hand side of each equality
        constraint (row) in ``A``.
    callback : callable, optional
        Callback function to be executed once per iteration.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.
    options : dict, optional
        A dictionary of solver options. The following options are accepted by
        this solver:

        maxiter : int (default = 1000)
            The maximum number of iterations of the algorithm.
        tol : float (default = 1e-8)
            Termination tolerance to be used for all termination criteria;
            see [4]_ Section 4.5.
        disp : bool (default = False)
            Set to ``True`` if indicators of optimization status are to be
            printed to the console each iteration.
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
            matrices contain mostly zeros and the problem is not very small
            (less than about 100 constraints or variables), consider setting
            ``True`` or providing ``A_eq`` and ``A_ub`` as sparse matrices.
        lstsq : bool (default = False)
            Set to ``True`` if the problem is expected to be very poorly
            conditioned. This should always be left ``False`` unless severe
            numerical difficulties are encountered. Leave this at the default
            unless you receive a warning message suggesting otherwise.
        sym_pos : bool (default = True)
            Leave ``True`` if the problem is expected to yield a well
            conditioned symmetric positive definite normal equation matrix
            (almost always). Leave this at the default unless you receive
            a warning message suggesting otherwise.
        cholesky : bool (default = True)
            Set to ``True`` if the normal equations are to be solved by explicit
            Cholesky decomposition followed by explicit forward/backward
            substitution. This is typically faster for problems
            that are numerically well-behaved.
        iterative : bool (default = False)
            Set to ``True`` if the normal equations are to be solved by an
            iterative method. Depending on the value of sym_pos either CG or
            GMRES is used.
        linear_operators : bool (default = False)
            Set to ``True`` if the iterative method should use linear operators
            to compute products with the matrix in the normal equation lazily
            instead of calculating the matrix in advance.
        pc : bool (default = True)
            Leave ``True`` if the predictor-corrector method of Mehrota is to be
            used. This is almost always (if not always) beneficial.
        ip : bool (default = False)
            Set to ``True`` if the improved initial point suggestion due to [4]_
            Section 4.3 is desired. Whether this is beneficial or not
            depends on the problem.
        permc_spec : str (default = 'MMD_AT_PLUS_A')
            (Has effect only with ``sparse = True``, ``lstsq = False``,
            ``sym_pos = True``, and no SuiteSparse.)
            A matrix is factorized in each iteration of the algorithm.
            This option specifies how to permute the columns of the matrix for
            sparsity preservation. Acceptable values are:

            - ``NATURAL``: natural ordering.
            - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
            - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of
              A^T+A.
            - ``COLAMD``: approximate minimum degree column ordering.

            This option can impact the convergence of the
            interior point algorithm; test different values to determine which
            performs best for your problem. For more information, refer to
            ``scipy.sparse.linalg.splu``.
        preconditioning_method : str (default = 'none')
            Specifies how the preconditioner is determined.
            - ``none``: No preconditioning
            - ``sketching``: Preconditioners are determined using a
              decomposition of a sketched matrix.
            Note that sketching assumes that the coefficient matrix has much more
            variables than constraints (m << n)
        sketching_factor : float (default = 2)
            Determines the size of the sketched matrix to be
                ``sketching_factor * m`` x ``m`` matrix
        sketching_sparsity : int (default = 3)
            Determines the number of nonzero entries in each column if the a
            sparse sketch is used (i.e. if ``sparse`` = True).

        A warning is issued for all unused options provided by the user.

    Returns
    -------
    x : 1-D array
        Solution vector.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.

    Notes
    -----
    This method implements the algorithm outlined in [4]_ with ideas from [8]_
    and a structure inspired by the simpler methods of [6]_.

    The primal-dual path following method begins with initial 'guesses' of
    the primal and dual variables of the standard form problem and iteratively
    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the
    problem with a gradually reduced logarithmic barrier term added to the
    objective. This particular implementation uses a homogeneous self-dual
    formulation, which provides certificates of infeasibility or unboundedness
    where applicable.

    The default initial point for the primal and dual variables is that
    defined in [4]_ Section 4.4 Equation 8.22. Optionally (by setting initial
    point option ``ip=True``), an alternate (potentially improved) starting
    point can be calculated according to the additional recommendations of
    [4]_ Section 4.4.

    A search direction is calculated using the predictor-corrector method
    (single correction) proposed by Mehrota and detailed in [4]_ Section 4.1.
    (A potential improvement would be to implement the method of multiple
    corrections described in [4]_ Section 4.2.) In practice, this is
    accomplished by solving the normal equations, [4]_ Section 5.1 Equations
    8.31 and 8.32, derived from the Newton equations [4]_ Section 5 Equations
    8.25 (compare to [4]_ Section 4 Equations 8.6-8.8). The advantage of
    solving the normal equations rather than 8.25 directly is that the
    matrices involved are symmetric positive definite, so Cholesky
    decomposition can be used rather than the more expensive LU factorization.

    With default options, the solver used to perform the factorization depends
    on third-party software availability and the conditioning of the problem.

    For dense problems, solvers are tried in the following order:

    1. ``scipy.linalg.cho_factor``

    2. ``scipy.linalg.solve`` with option ``sym_pos=True``

    3. ``scipy.linalg.solve`` with option ``sym_pos=False``

    4. ``scipy.linalg.lstsq``

    For sparse problems:

    1. ``sksparse.cholmod.cholesky`` (if scikit-sparse and SuiteSparse are
       installed)

    2. ``scipy.sparse.linalg.factorized`` (if scikit-umfpack and SuiteSparse are
       installed)

    3. ``scipy.sparse.linalg.splu`` (which uses SuperLU distributed with SciPy)

    4. ``scipy.sparse.linalg.lsqr``

    If the solver fails for any reason, successively more robust (but slower)
    solvers are attempted in the order indicated. Attempting, failing, and
    re-starting factorization can be time consuming, so if the problem is
    numerically challenging, options can be set to  bypass solvers that are
    failing. Setting ``cholesky=False`` skips to solver 2,
    ``sym_pos=False`` skips to solver 3, and ``lstsq=True`` skips
    to solver 4 for both sparse and dense problems.

    Potential improvements for combatting issues associated with dense
    columns in otherwise sparse problems are outlined in [4]_ Section 5.3 and
    [10]_ Section 4.1-4.2; the latter also discusses the alleviation of
    accuracy issues associated with the substitution approach to free
    variables.

    After calculating the search direction, the maximum possible step size
    that does not activate the non-negativity constraints is calculated, and
    the smaller of this step size and unity is applied (as in [4]_ Section
    4.1.) [4]_ Section 4.3 suggests improvements for choosing the step size.

    The new point is tested according to the termination conditions of [4]_
    Section 4.5. The same tolerance, which can be set using the ``tol`` option,
    is used for all checks. (A potential improvement would be to expose
    the different tolerances to be set independently.) If optimality,
    unboundedness, or infeasibility is detected, the solve procedure
    terminates; otherwise it repeats.

    The expected problem formulation differs between the top level ``linprog``
    module and the method specific solvers. The method specific solvers expect a
    problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    Whereas the top level ``linprog`` module expects a problem of form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    The original problem contains equality, upper-bound and variable constraints
    whereas the method specific solver requires equality constraints and
    variable non-negativity.

    ``linprog`` module converts the original problem to standard form by
    converting the simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.


    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point methods
            for large scale linear programming. HEC/Universite de Geneve, 1996.

    """

    options = AllOptions.from_dict(
        options_dict=options_dict,
        has_umfpack=has_umfpack,
        has_cholmod=has_cholmod,
    )

    x, status, message, iteration = _ip_hsd(
        A, b, c, c0, callback, postsolve_args, options
    )

    return x, status, message, iteration
