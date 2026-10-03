"""Check recursive ownership and the unit-graph hinges against actual graphs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gasket_control', ROOT/'scripts/audit_recursive_gasket_coherence.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


@pytest.mark.parametrize('s,m', [(1, 2), (2, 2), (3, 3)])
def test_cover_is_original_graph_and_ownership_is_nested(s, m):
    graph, cells = control.cell_cover(s, m)
    original = control._build_level(s+m)
    assert graph.adj == original.adj and graph.corners == original.corners
    fine, sizes = control.cell_labels(graph, cells, s)
    coarse_graph, coarse_cells = control.cell_cover(s+1, m-1)
    coarse, _ = control.cell_labels(coarse_graph, coarse_cells, s+1)
    assert np.array_equal(coarse, fine//3)
    V = (3**(s+1)+3)//2
    assert V-3 <= sizes.min() <= sizes.max() <= V


def test_unit_graph_diameter_ports_and_refinement_bounds():
    for m in range(1, 5):
        adj = control.contact_graph(m)
        unit = control.integer_distances(adj)
        assert len(adj) == 3**m and unit.max() == 2**m-1
        assert unit[0, (3**m-1)//2] == 2**m-1
        assert all(len(neighbors) <= 3 for neighbors in adj.values())
        parent = control.integer_distances(control.contact_graph(m-1))
        projection = np.arange(len(adj))//3
        assert np.abs(unit-2*parent[projection[:, None], projection]).max() <= 1
        for cell in range(3**(m-1)):
            ports = []
            for child in range(3):
                z = 3*cell+child
                outside = [y//3 for y in adj[z] if y//3 != cell]
                assert len(outside) <= 1
                ports.extend(outside)
            assert len(ports) == len(set(ports))


def test_recursive_cell_coherence_and_nonembedding_share_one_layer():
    original = control._build_level(5)
    record, _ = control.layer(3, 2, original, tau=1)
    assert record['delta'] <= .02
    assert record['escape_max'] <= .02
    assert record['inf_count'] == 0
    assert record['nonembedding']['relative_fit_error_lower_bound'] == '1/12'
    with pytest.raises(ValueError):
        control.layer(1, 2, control._build_level(3))
    with pytest.raises(ValueError):
        control.contact_graph(-1)


def test_stage_five_has_same_cell_contacts_and_exact_interface_flux():
    from geo_sbt.substrates.sierpinski import sierpinski
    from geo_sbt.packaging import make_C
    from geo_sbt.lenses.prototypes import prototypes_uniform
    from geo_sbt.geometry.metric import macro_kernel
    original = control._build_level(5)
    record, values = control.layer(3, 2, original, tau=5)
    assert record['interface_flux'] == '11905/16384'
    assert record['escape_max'] < .06
    assert record['nonembedding']['relative_fit_error_lower_bound'] == '1/12'
    _, P = sierpinski(5, lazy=.5)
    labels = values[0]
    expected = macro_kernel(P, 5, make_C(labels), prototypes_uniform(labels))
    assert np.allclose(values[2].toarray(), expected, atol=1e-13, rtol=0)


def test_unit_graph_uniform_ball_growth_and_limiting_quad_prefixes():
    for m in range(2, 6):
        unit = control.integer_distances(control.contact_graph(m))
        for R in range(1, 2**m):
            k = R.bit_length()
            counts = (unit <= R).sum(axis=1)
            assert counts.min() >= 3**(k-1)
            assert counts.max() <= 10*3**(k-1)
        if m >= 3:
            q, p = 3**(m-1), 3**(m-2)
            vertices = [(q-1)//2, 2*q+(p-1)//2, 2*p-1, 2*q-1]
            a = 2**(m-2)
            expected = {'01': 3*a-1, '02': a-1, '03': 2*a, '12': 2*a, '13': a+1, '23': 3*a-1}
            assert {f'{i}{j}': int(unit[vertices[i], vertices[j]]) for i in range(4) for j in range(i+1,4)} == expected
