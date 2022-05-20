
class UncertainTransform:
    def __init__(self):
        pass

    @staticmethod
    def compose(self, Tij, Pij, Tjk, Pjk):
        raise NotImplementedError()

    @staticmethod
    def inv(self, Tij, Pij):
        raise NotImplementedError()

    @staticmethod
    def between(Tij, Pij, Tik, Pik):
        raise NotImplementedError()

class LeftPerturbations(UncertainTransform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compose(Tij, Pij, Tjk, Pjk, Pij_jk=None):
        Tik = Tij * Tjk
        Ad_ij = Tij.Adj
        Pik = Pij + Ad_ij @ Pjk @ Ad_ij.T
        if Pij_jk is not None:
            Pik += Pij_jk @ Ad_ij.T + Ad_ij + Pij_jk.T

        return Tik, Pik

    @staticmethod
    def inv(Tij, Pij):
        Tji = Tij.inv()
        Ad_ji = Tji.Adj
        Pji = Ad_ji @ Pij @ Ad_ji.T

        return Tji, Pji

    @staticmethod
    def between(Tij, Pij, Tik, Pik, Pij_ik=None):
        Tji, Pji = LeftPerturbations.inv(Tij, Pij)
        Tjk = Tji * Tik
        Ad_ji = Tji.Adj
        Pjk = Pji + Ad_ji @ Pik @ Ad_ji.T
        if Pij_ik is not None:
            Pjk -= (Ad_ji @ Pij_ik @ Ad_ji.T + Ad_ji @ Pij_ik.T @ Ad_ji.T)

        return Tjk, Pjk

class RightPerturbations(UncertainTransform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compose(Tij, Pij, Tjk, Pjk, Pij_jk=None):
        Tik = Tij * Tjk
        Ad_kj = Tjk.inv().Adj
        Pik = Pjk + Ad_kj @ Pij @ Ad_kj.T
        if Pij_jk is not None:
            Pik += Pij_jk @ Ad_kj.T + Ad_kj + Pij_jk.T

        return Tik, Pik

    @staticmethod
    def inv(Tij, Pij):
        Tji = Tij.inv()
        Ad_ij = Tij.Adj
        Pji = Ad_ij @ Pij @ Ad_ij.T

        return Tji, Pji

    @staticmethod
    def between(Tij, Pij, Tik, Pik, Pij_ik=None):
        Tji, Pji = LeftPerturbations.inv(Tij, Pij)
        Tjk = Tji * Tik
        Ad_ij = Tij.Adj
        Ad_ki = Tik.inv().Adj
        Ad = Ad_ki @ Ad_ij
        Pjk = Pik + Ad @ Pij @ Ad.T
        if Pij_ik is not None:
            Pjk -= (Ad @ Pij_ik + Pij_ik.T @ Ad.T)

        return Tjk, Pjk
