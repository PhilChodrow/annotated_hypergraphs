from unittest import TestCase
from data_loader import DATA, ROLE_FIELDS

from ahyper.annotated_hypergraph import AnnotatedHypergraph
from ahyper.observables import local_role_density

class DensityTests(TestCase):
    """
    Basic tests for the calculation of role densities.
    """

    def setUp(self):
        self.A = AnnotatedHypergraph(DATA, ROLE_FIELDS)

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