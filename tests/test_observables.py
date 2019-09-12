from unittest import TestCase, skip
from data_loader import DATA, ROLE_FIELDS

from ahyper.annotated_hypergraph import AnnotatedHypergraph
from ahyper.observables import (local_role_density, degree_centrality, 
                                eigenvector_centrality, pagerank_centrality)

class DensityTests(TestCase):
    """
    Basic tests for the calculation of role densities.
    """

    def setUp(self):
        self.A = AnnotatedHypergraph.from_records(DATA, ROLE_FIELDS)

    def tearDown(self):
        self.A = None

    def test_counts(self):
        """
        Test whether the function is correctly producing role counts.
        """
        density = local_role_density(self.A, include_focus=False, absolute_values=True)
    
    def test_densities(self):
        """
        Test whether the densities returned are meaningful
        """
        density = local_role_density(self.A, include_focus=False, absolute_values=False)

        for entry in density.values():
            self.assertAlmostEqual(sum(entry.values()), 1.0)

    def test_focus(self):
        """
        Test whether including focus changes counts.
        """

        density_focus = local_role_density(self.A, include_focus=True, absolute_values=True)
        density_no_focus = local_role_density(self.A, include_focus=False, absolute_values=True)

        for key, entry in density_focus.items():
            self.assertNotEqual(sum(entry.values()), sum(density_no_focus[key].values()))

class CentralityTests(TestCase):
    """
    Basic tests for the calculation of centralities on the weighted projected network.
    """
    def setUp(self):
        self.A = AnnotatedHypergraph.from_records(DATA, ROLE_FIELDS)

    def tearDown(self):
        self.A = None

    def test_degree_centrality(self):
        """ Test the degree centrality calculation. """

        degrees = degree_centrality(self.A)

        self.assertEqual(degrees[96], 2237.0)

    def test_eigenvector_centrality(self):
        """ Test the eigenvector centrality calculation. """

        eigenvector = eigenvector_centrality(self.A)

        self.assertAlmostEqual(eigenvector[67], 0.14693461555354528)

    def test_pagerank_centrality(self):
        """ Test the eigenvector centrality calculation. """

        pagerank = pagerank_centrality(self.A)

        self.assertAlmostEqual(pagerank[8], 0.01516827830191012)