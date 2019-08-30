from unittest import TestCase
import numpy as np

from data_loader import DATA, ROLE_FIELDS

from ahyper.annotated_hypergraph import AnnotatedHypergraph

class ConstructionTests(TestCase):
    """
    Basic tests for the construction of Annotated Hypergraphs
    """

    def test_constructor(self):
        """
        """
    
        A = AnnotatedHypergraph(DATA, ROLE_FIELDS)

class MCMCTests(TestCase):
    """
    Testing the MCMC algorithm
    """

    def setUp(self):
        self.A = AnnotatedHypergraph(DATA, ROLE_FIELDS)

    def tearDown(self):
        self.A = None
    
    def test_degree_and_dimension_preserved(self):
        """
        Ensure degree and dimension sequence is preserved at each step.
        """

        d0 = self.A.node_degrees(by_role=True)
        k0 = self.A.edge_dimensions(by_role=True)
        
        self.A.stub_labeled_MCMC(n_steps=1)

        d = self.A.node_degrees(by_role=True)
        k = self.A.edge_dimensions(by_role=True)

        self.assertEqual(d0, d)
        self.assertEqual(k0, k)

    def test_swaps(self):
        """
        Tests that at least one swap is made in a multiple passes of the MCMC.
        Since certain swaps lead to zero change in the incidence list we cannot
        test a single swap.

        Note: This test may fail if the all steps result in zero change.
        """

        il_before = self.A.get_IL()

        self.A.stub_labeled_MCMC(n_steps=10)

        il_after = self.A.get_IL()

        diff = sum([x.nid!=y.nid for x,y in zip(il_before, il_after)])
        
        self.assertGreater(diff, 0)

class ConversionTests(TestCase):
    """
    Testing the conversion methods.
    """

    def setUp(self):
        self.A = AnnotatedHypergraph(DATA, ROLE_FIELDS)

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
        R = 2*np.ones((len(ROLE_FIELDS), len(ROLE_FIELDS)))
        self.A.assign_role_interaction_matrix(R)
        weighted_edges = self.A.to_weighted_projection()
        
        # One particular example
        self.assertEqual(weighted_edges[0][6], 16.0)