#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import subprocess
import os
import random
import math

# Machine Learning imports
import cma
from sklearn import tree

class EchantillonNotDefinedException(Exception):
    pass

class NoPopulationDefinedException(Exception):
    print ('Please provide one or several populations.')
    pass

class PopulationManager():
    """Population Manager is a class that loads a population from different sources: panda dataframe, json, openfisca.
       This population can be used for automatic inference of the law, or making parametric reforms.
       It contains methods for filtering and handling this population.
       It takes as input a population with data (e.g., salaries/taxes/subsidies) this population

       First a population is given in _raw_population
       Second, concepts (for instance, parent_isole) can be built and stored in _population
    """

    def __init__(self, openfisca_results_filename=None, variables_loaded=[], echantillon=None):
        """Defining a new population

            :arg echantillon: the percentage of the population covered by the population.
                Example: for a population of 120 000  that is representative of 93% of the global population of 60 M
                        echantillon = 120000 / (.93 * 60000000) = 0.0021
        """
        self._echantillon = echantillon
        self._raw_population = None
        self._population = []

        if openfisca_results_filename is not None:
            results_openfisca = self.load_from_json(openfisca_results_filename + '-openfisca.json')
            testcases = self.load_from_json(openfisca_results_filename + '-testcases.json')
            self._raw_population = testcases[:]
            self._population = [None] * len(testcases)
            for i  in range(len(self._population)):
                self._population[i] = {}

            for variable in variables_loaded:
                for i in range(len(results_openfisca)):
                    self._raw_population[i][variable] = results_openfisca[i][variable]
                    self._population[i][variable] = results_openfisca[i][variable]
        else:
            raise NoPopulationDefinedException()

    def summarize_population_from_cerfa_values(self):
        total_people = 0
        for family in self._raw_population:
            total_people += 1
            if '0DB' in family and family['0DB'] == 1:
                total_people += 1
            if 'F' in family:
                total_people += family['F']

        # We assume that there are 2000000 people with RSA
        # TODO Put that where it belongs in the constructor
        self._echantillon =  float(total_people) / 30000000
        print 'Echantillon of ' + repr(total_people) + ' people, in percent of french population for similar revenu: ' + repr(100 * self._echantillon) + '%'


    def compute_echantillion_from_cerfa_fields(self):
        total_people = 0
        for family in self._raw_population:
            total_people += 1
            if '0DB' in family and family['0DB'] == 1:
                total_people += 1
            if 'F' in family:
                total_people += family['F']

        # We assume that there are 2000000 people with RSA
        # TODO Put that where it belongs in the constructor
        # There are 62M inhabitant, 93% of which are represented on declaration of revenu
        self._echantillon =  float(total_people) / (62000000 * 0.93)
        print 'Echantillon of ' + repr(total_people) + ' people, in percent of french population for similar revenu: ' + repr(100 * self._echantillon) + '%'

    def add_concept(self, concept, function):
        for i in range(len(self._raw_population)):
            result = function(self._raw_population[i])
            if result is not None and result is not False:
                self._population[i][concept] = float(result)

    def filter_only_likely__population(self):
        """
            Removes unlikely elements in the population
            TODO: This should be done by the population generator

        :param: raw_population_manager._population:
        :return: raw_population_manager._population without unlikely cases
        """
        new_raw__population = []
        for case in self._raw_population_manager._population:
            if (int(case['0DA']) <= 1950 or ('0DB' in case and int(case['0DA'] <= 1950))) and 'F' in case and int(case['F']) > 0:
                pass
            else:
                new_raw__population.append(case)
        self._raw_population = new_raw__population

    def filter_only_no_revenu(self):
        """
            Removes people who have a salary from the population

        :param raw_population_manager._population:
        :return: raw_population_manager._population without null salary
        """
        new_raw_population = []
        for case in self._raw_population_manager._population:
            if case.get('1AJ', 0) < 1 and case.get('1BJ', 0) < 1:
                new_raw_population._population.append(case)
        self._raw_population_manager._population = new_raw_population

    def load_from_json(self, filename):
        with open('../data/' + filename, 'r') as f:
            return json.load(f)