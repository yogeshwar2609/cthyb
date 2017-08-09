from pytriqs.gf import *
from itertools import product
from numpy import array, conj, pi, sqrt
from scipy.special import spherical_jn

def make_T_bar(o, l_max):
    """
    Make a function object that returns modified T-coefficients for a fixed index 'o'.

    .. math:: \bar T_{o,l} = \sqrt{2l+1}i^o i^l j_l\left(\frac{o\pi}{2}\right).

    Parameters
    ----------
    g2_legendre : Block2Gf
                  Two-particle Green's function in the mixed Matsubara/Legendre representation
    n_inu : int
            Number of fermionic Matsubara frequencies in the output

    Returns
    -------
    T_bar : function
            T_bar(l) = \bar T_{o,l}

    """
    l_range = range(0, l_max+1)
    jl = spherical_jn(l_range, abs(o) * pi / 2)
    t = array([sqrt(2*l+1) * pow(1j, l) * (pow(-1, l) if o < 0 else 1.0) * jl[l] for l in l_range]) * pow(1j, o)

    return lambda l: t[l]


def G2Legendre2Matsubara(g2_legendre, n_inu):
    r"""
    Transform a two-particle Green's function from a mixed Matsubara/Legendre
    representation G^2(iomega,l,l') to the pure Matsubara representation
    G^2(iomega,inu,inu').

    .. math::
        G^2(i\omega,i\nu,i\nu') =
        \sum_{ll'} \bar T^*_{2n+m+1,l} G^2(i\omega,l,l') \bar T_{2n'+m+1,l'}

    Parameters
    ----------
    g2_legendre : Block2Gf
                  Two-particle Green's function in the mixed Matsubara/Legendre representation
    n_inu : int
            Number of fermionic Matsubara frequencies in the output

    Returns
    -------
    g2_matsubara : Block2Gf
                   Two-particle Green's function in the Matsubara representation

    """
    name_list1 = list(set(b[0] for b in g2_legendre.indices))
    name_list2 = list(set(b[1] for b in g2_legendre.indices))

    # Create output container
    def make_block(bn1, bn2):
        iw_mesh = g2_legendre[bn1, bn2].mesh.components[0]
        inu_mesh = MeshImFreq(iw_mesh.beta, 'Fermion', n_inu)
        return Gf(mesh = MeshProduct(iw_mesh, inu_mesh, inu_mesh),
                  target_shape = g2_legendre[bn1, bn2].target_shape)
    g2 = Block2Gf(name_list1, name_list2, make_block)

    # Perform transformation
    for bn1, bn2 in product(name_list1, name_list2):
        g2_in = g2_legendre[bn1, bn2]
        g2_out = g2[bn1, bn2]

        l_mesh, lp_mesh = g2_in.mesh.components[1:]
        iw_mesh, inu_mesh, inup_mesh = g2_out.mesh.components
        assert iw_mesh == g2_in.mesh.components[0]

        for iw_idx, iw in enumerate(iw_mesh):
            m = iw_idx - len(iw_mesh)/2
            for (inu_idx, inu), (inup_idx, inup) in product(enumerate(inu_mesh), enumerate(inup_mesh)):
                n = inu_idx - len(inu_mesh)/2
                np = inup_idx - len(inup_mesh)/2

                t_bar = make_T_bar(2*n + m + 1, len(l_mesh)-1)
                t_bar_p = make_T_bar(2*np + m + 1, len(lp_mesh)-1)

                for l, lp in product(l_mesh, lp_mesh):
                    l, lp = int(l.real), int(lp.real)
                    g2_out.data[iw_idx,inu_idx,inup_idx,...] += \
                        conj(t_bar(l)) * g2_in.data[iw_idx,l,lp,...] * t_bar_p(lp)

    return g2


if __name__ == '__main__':
    from pytriqs.archive import HDFArchive
    with HDFArchive('measure_g2.h5', 'r') as inp, HDFArchive('legendre2matsubara.h5', 'w') as out:
        n_inu = 10
        G2_iw_l_lp_pp = inp['G2_iw_l_lp_pp']
        G2_iw_l_lp_ph = inp['G2_iw_l_lp_ph']
        out['G2_iw_inu_inup_pp_trans'] = G2Legendre2Matsubara(G2_iw_l_lp_pp, n_inu)
        out['G2_iw_inu_inup_ph_trans'] = G2Legendre2Matsubara(G2_iw_l_lp_ph, n_inu)
