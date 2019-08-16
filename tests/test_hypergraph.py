from unittest import TestCase
from data_loader import DATA, ROLE_FIELDS

from ahyper.annotated_hypergraph import annotated_hypergraph

class ConstructionTests(TestCase):
    """
    Basic tests for the construction of Annotated Hypergraphs
    """

    def test_constructor(self):
        """
        """
    
        A = annotated_hypergraph(DATA, ROLE_FIELDS)


class MCMCTests(TestCase):
    """
    Testing the MCMC algorithm
    """

    def setUp(self):
        self.A = annotated_hypergraph(DATA, ROLE_FIELDS)

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

        diff = sum([x[0]!=y[0] for x,y in zip(il_before, il_after)])
        
        self.assertGreater(diff, 0)