from unittest import TestCase
from data_loader import DATA, ROLE_FIELDS

from ahyper.utils import *


class UtilTests(TestCase):
    """
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_incidence_records_conversion(self):
        """
        Ensure that records can records can be reversibly
        transformed into an incidence list.
        """

        il = incidence_list_from_records(DATA, ROLE_FIELDS)
        records = records_from_incidence_list(il, ROLE_FIELDS)

        self.assertEqual(DATA, records)
