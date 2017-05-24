import unittest
from nematus.lm import KenLM
from os import remove
from os.path import exists
import logging

test_location = '/home/.../kenlm_model/kenlm_test_model.zip'


class KenLMTestCase(unittest.TestCase):
    logger = None

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(__file__)
        cls.logger.info("========================================================================================")
        cls.logger.info("Setting up the KenLM tests")
        cls.logger.info("========================================================================================")
        cls.klm = KenLM()

    @classmethod
    def tearDownClass(cls):
        cls.logger.info("========================================================================================")
        cls.logger.info("Tearing down the KenLM tests")
        cls.logger.info("========================================================================================")

    def setUp(self):
        self.klm = KenLM()
        self.zip_path = 'kenlm_test.zip'
        if exists(self.zip_path):
            remove(self.zip_path)

    def tearDown(self):
        if exists(self.zip_path):
            remove(self.zip_path)

    def test_load(self):
        self.logger.info("========================================================================================")
        self.logger.info("Attempt to load a KenLM model")
        self.logger.info("========================================================================================")
        self.klm = KenLM()
        self.klm.load(test_location)

    def test_train(self):
        self.logger.info("========================================================================================")
        self.logger.info("Attempt to train a KenLM model")
        self.logger.info("========================================================================================")
        path = '/to/do/some/text/file'
        self.klm.train(path)
        self.klm.save(self.zip_path)

    def test_save(self):
        self.logger.info("========================================================================================")
        self.logger.info("Attempt to save a loaded KenLM model")
        self.logger.info("========================================================================================")
        self.klm.load(test_location)
        self.klm.save(self.zip_path)

    def test_score(self):
        self.logger.info("========================================================================================")
        self.logger.info("Attempt to score a sentence and see if it matches the expected result.")
        self.logger.info("========================================================================================")
        self.klm.load(test_location)
        sentence = 'language modeling is fun .'
        expected_score = [-6.203308582305908, -3.873892307281494, -6.203308582305908, -6.203308582305908,
                          -6.203308582305908, -3.873892307281494, -6.203308582305908, -6.203308582305908,
                          -1.7693414688110352, -6.203308582305908, -6.203308582305908, -6.203308582305908,
                          -6.203308582305908, -6.203308582305908, -6.203308582305908, -6.203308582305908,
                          -6.203308582305908, -1.7693414688110352, -6.203308582305908, -6.203308582305908,
                          -1.7693414688110352, -6.203308582305908, -6.203308582305908, -6.203308582305908,
                          -1.7693414688110352, -4.689509868621826]
        self.assertEqual(self.klm.score(sentence), expected_score)


if __name__ == '__main__':
    unittest.main()

