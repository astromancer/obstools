


def check_bc(self, p=None):
    # check that boundary conditions are satisfied

    if self.nconstraints == 0:
        # no boundary conditions
        return True

    if p is None:
        # use random parameter vector
        p = np.random.randn(self.dof)

    coeff = self.get_block_coeff(p)
    lhs = np.polyval(coeff, 0)[1:]
    rhs = list(map(np.polyval, coeff.T, np.diff(self.breakpoints[:-1])))
    assert np.allclose(lhs, rhs)