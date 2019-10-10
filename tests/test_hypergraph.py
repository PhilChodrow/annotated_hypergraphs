from unittest import TestCase
import numpy as np

from data_loader import DATA, ROLE_FIELDS

from ahyper.annotated_hypergraph import AnnotatedHypergraph


class ConstructionTests(TestCase):
    """
    Basic tests for the construction of Annotated Hypergraphs.
    """

    def test_record_constructor(self):
        """
        Test building from records.
        """

        A = AnnotatedHypergraph.from_records(DATA, ROLE_FIELDS)

    def test_incidence_constructor(self):
        """
        Test building from incidence data.
        """

        A = AnnotatedHypergraph.from_incidence(
            dataset="enron", root="./data/", relabel_roles=True, add_metadata=True
        )


class MCMCTests(TestCase):
    """
    Testing the MCMC algorithm
    """

    def setUp(self):
        self.A = AnnotatedHypergraph.from_records(DATA, ROLE_FIELDS)

    def tearDown(self):
        self.A = None

    def test_degree_and_dimension_preserved(self):
        """
        Ensure degree and dimension sequence is preserved at each step.
        """

        d0 = self.A.node_degrees(by_role=True)
        k0 = self.A.edge_dimensions(by_role=True)

        self.A.MCMC(n_steps=1, avoid_degeneracy=True)

        d = self.A.node_degrees(by_role=True)
        k = self.A.edge_dimensions(by_role=True)

        self.assertEqual(d0, d)
        self.assertEqual(k0, k)

    def test_swaps_degenerate(self):
        """
        Tests that at least one swap is made in a multiple passes of the MCMC.
        Since certain swaps lead to zero change in the incidence list we cannot
        test a single swap.

        Note: This test may fail if the all steps result in zero change.
        """

        il_before = self.A.get_IL()

        self.A.MCMC(n_steps=10, avoid_degeneracy=True)

        il_after = self.A.get_IL()

        diff = sum([x.nid != y.nid for x, y in zip(il_before, il_after)])

        self.assertGreater(diff, 0)

    def test_degree_and_dimension_preserved_degenerate(self):
        """
        Ensure degree and dimension sequence is preserved at each step.

        Note: Degenerate can result in a node being in two roles in one edge.
        """

        d0 = self.A.node_degrees(by_role=True)
        k0 = self.A.edge_dimensions(by_role=True)

        self.A.MCMC(n_steps=1, avoid_degeneracy=False)

        d = self.A.node_degrees(by_role=True)
        k = self.A.edge_dimensions(by_role=True)

        self.assertEqual(d0, d)
        self.assertEqual(k0, k)

    def test_swaps_degenerate(self):
        """
        Tests that at least one swap is made in a multiple passes of the MCMC.
        Since certain swaps lead to zero change in the incidence list we cannot
        test a single swap.

        Note: Degenerate can result in a node being in two roles in one edge.
        """

        il_before = self.A.get_IL()

        self.A.MCMC(n_steps=10, avoid_degeneracy=False)

        il_after = self.A.get_IL()

        diff = sum([x.nid != y.nid for x, y in zip(il_before, il_after)])

        self.assertGreater(diff, 0)


class ConversionTests(TestCase):
    """
    Testing the conversion methods.
    """

    def setUp(self):
        self.A = AnnotatedHypergraph.from_records(DATA, ROLE_FIELDS)

    def tearDown(self):
        self.A = None

    def test_convert_to_weighted_edges(self):
        """Test projection to a weighted directed graph."""

        weighted_edges = self.A.to_weighted_projection()

        # One particular example
        self.assertEqual(weighted_edges[0][6], 8.0)

    def test_convert_to_weighted_edges_with_custom_R(self):
        """
        Test projection to a weighted directed graph, using a custom
        role-interaction matrix, R.
        """
        R = 2 * np.ones((len(ROLE_FIELDS), len(ROLE_FIELDS)))
        self.A.assign_role_interaction_matrix(R)
        weighted_edges = self.A.to_weighted_projection()

        # One particular example
        self.assertEqual(weighted_edges[0][6], 16.0)

    def test_convert_to_bipartite_graph(self):
        """Test conversion to a bipartite graph."""

        G = self.A.to_bipartite_graph()

        self.assertEqual(len(G.nodes()), self.A.n + self.A.m)
