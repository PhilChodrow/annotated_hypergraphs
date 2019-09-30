from unittest import TestCase, skip
from data_loader import DATA, ROLE_FIELDS

from ahyper.annotated_hypergraph import AnnotatedHypergraph
from ahyper.observables import (local_role_density, degree_centrality, 
                                eigenvector_centrality, pagerank_centrality,
                                connected_components, random_walk, random_walk_pagerank,
                                assortativity)

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

        degrees = degree_centrality(self.A, use_networkx=True)

        self.assertAlmostEqual(degrees[96], 2237.0)

    def test_eigenvector_centrality(self):
        """ Test the eigenvector centrality calculation. """

        eigenvector = eigenvector_centrality(self.A)

        self.assertAlmostEqual(eigenvector[67], 0.14693461555354528)

    def test_pagerank_centrality(self):
        """ Test the eigenvector centrality calculation. """

        pagerank = pagerank_centrality(self.A)

        self.assertAlmostEqual(pagerank[8], 0.01516827830191012)

    def test_connected_components(self):
        """ Test the number of connected components."""

        components = connected_components(self.A)

        self.assertEqual(components, 3)

class RandomWalkTests(TestCase):
    """
    Basic tests for methods including random walks.
    """
    def setUp(self):
        self.A = AnnotatedHypergraph.from_records(DATA, ROLE_FIELDS)

    def tearDown(self):
        self.A = None

    def test_random_walk(self):
        """Test random walk method. """
        G = self.A.to_bipartite_graph()
        rw = random_walk(G,
                         n_steps=1000,
                         alpha=0.1,
                         nonbacktracking=False)

    def test_pagerank(self):
        """ Test random walk PageRank. """
        pagerank = random_walk_pagerank(self.A,
                                        n_steps=1000,
                                        nonbacktracking=False,
                                        alpha_1=0.1,
                                        alpha_2=0.2,
                                        return_path=False)

    def test_sampled_assortativity(self):
        """Test sampled assortativity. """
        assort = assortativity(self.A,
                               n_samples=1000,
                               by_role=True,
                               spearman=True)